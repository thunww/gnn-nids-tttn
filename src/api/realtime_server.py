"""Phase F: FastAPI server -- tail conn.log, gom cua so flow, suy luan GraphSAGE, day ket qua
real-time ra trinh duyet qua WebSocket. Xem docs/graphsage/05_demo_realtime_setup.md muc 8.

Chay: venv/bin/python -m uvicorn api.realtime_server:app --host 0.0.0.0 --port 8000
(chay tu thu muc src/, hoac them src/ vao PYTHONPATH).

GIOI HAN DA BIET: neu Zeek xoay vong (archive) conn.log giua luc demo (mac dinh zeekctl xoay
theo gio), tien trinh tail dang mo se khong thay file moi -- voi 1 buoi demo ngan (~15-30 phut)
kha nang thap, nhung neu gap thi restart lai server nay la du (khong can restart Zeek).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from api.infer import RealtimeDetector
from api.window_buffer import SlidingWindowBuffer
from graph.config import WINDOW_OVERLAP, WINDOW_SIZE_BY_DATASET

CONN_LOG_PATH = Path("/opt/zeek/logs/current/conn.log")
PROCESSED_DIR = Path("data/processed")
DATASET = "nf-cse-cic-ids2018-v2"
POLL_INTERVAL = 0.5  # giay -- tan suat kiem tra dong moi khi conn.log tam thoi khong co gi moi
WINDOW_SIZE = WINDOW_SIZE_BY_DATASET[DATASET]

# So do mang demo (xem docs/graphsage/05_demo_realtime_setup.md muc 4.1) -- dung de: (1) loc bo
# traffic vao chinh trang demo nay (gay nhieu, xem _is_dashboard_traffic), (2) hien chu thich
# vai tro IP tren giao dien cho de doc ket qua.
MONITOR_IP = "192.168.207.200"
DASHBOARD_PORT = "8000"
VICTIM_IP = "192.168.207.199"
ATTACKER_IP = "192.168.207.194"

_clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _is_dashboard_traffic(row: dict[str, str]) -> bool:
    """Loai traffic ra/vao chinh cong web demo nay (vd may host mo trinh duyet xem dashboard) --
    khong phai traffic mang can phan tich, dua vao se gay nhieu (tu xem dashboard bi tinh nham
    la tan cong flood do ket noi lien tuc)."""
    return (row.get("id.resp_h") == MONITOR_IP and row.get("id.resp_p") == DASHBOARD_PORT) or (
        row.get("id.orig_h") == MONITOR_IP and row.get("id.orig_p") == DASHBOARD_PORT
    )


def _tail_conn_log(path: Path) -> Iterator[dict[str, str] | None]:
    """Doc #fields header, bo qua noi dung co san truoc do, roi lien tuc yield dong moi xuat
    hien (yield None khi tam thoi chua co dong moi, de vong lap goi ham nghi ngoi)."""
    while not path.exists():
        yield None
    with open(path, encoding="utf-8") as f:
        fields: list[str] | None = None
        for line in f:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
        f.seek(0, 2)  # bo qua toan bo noi dung cu, chi lay dong MOI tu thoi diem server khoi dong
        while True:
            pos = f.tell()
            line = f.readline()
            if not line or not line.endswith("\n"):
                f.seek(pos)
                yield None
                continue
            if line.startswith("#"):
                if line.startswith("#fields"):
                    fields = line.rstrip("\n").split("\t")[1:]
                continue
            if fields is None:
                continue
            values = line.rstrip("\n").split("\t")
            if len(values) != len(fields):
                continue
            yield dict(zip(fields, values))


def _broadcast_threadsafe(message: dict) -> None:
    assert _loop is not None
    payload = json.dumps(message, default=str, ensure_ascii=False)
    for ws in list(_clients):
        asyncio.run_coroutine_threadsafe(_safe_send(ws, payload), _loop)


async def _safe_send(ws: WebSocket, payload: str) -> None:
    try:
        await ws.send_text(payload)
    except Exception:
        _clients.discard(ws)


def _flow_records(df, cols: list[str]) -> list[dict]:
    return df[cols].to_dict("records")


def _worker() -> None:
    print(f"[realtime_server] nap model + scaler ({DATASET}) ...")
    detector = RealtimeDetector(PROCESSED_DIR, DATASET)
    print(f"[realtime_server] da nap xong. WINDOW_SIZE={WINDOW_SIZE}, overlap={WINDOW_OVERLAP}")
    buffer = SlidingWindowBuffer(WINDOW_SIZE, WINDOW_OVERLAP)
    total_flows = 0
    skipped_dashboard = 0

    for row in _tail_conn_log(CONN_LOG_PATH):
        if row is None:
            time.sleep(POLL_INTERVAL)
            continue

        if _is_dashboard_traffic(row):
            skipped_dashboard += 1
            continue

        total_flows += 1
        window = buffer.add(row)
        _broadcast_threadsafe(
            {
                "type": "progress",
                "total_flows": total_flows,
                "skipped_dashboard": skipped_dashboard,
                "flows_until_next_window": buffer.flows_until_next_window(),
                "window_size": WINDOW_SIZE,
                "last_flow": {
                    "src": f"{row.get('id.orig_h')}:{row.get('id.orig_p')}",
                    "dst": f"{row.get('id.resp_h')}:{row.get('id.resp_p')}",
                    "proto": row.get("ip_proto"),
                },
            }
        )

        if window is None:
            continue

        result = detector.predict_window(window)
        attacks = result[result["pred_label"] == 1]
        print(f"[realtime_server] cua so moi: {len(result)} flow, {len(attacks)} bi gan nhan Attack")
        cols = ["ts", "uid", "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "pred_proba"]
        _broadcast_threadsafe(
            {
                "type": "window_result",
                "num_flows": int(len(result)),
                "num_attack": int(len(attacks)),
                "attacks": _flow_records(attacks, cols),
                "recent": _flow_records(result.tail(15), cols + ["pred_label"]),
            }
        )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _loop
    _loop = asyncio.get_event_loop()
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="GraphSAGE NIDS - Demo real-time", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.discard(websocket)


_INDEX_HTML = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphSAGE NIDS &mdash; Real-time Monitor</title>
<style>
  :root {{
    color-scheme: dark;
    --page:       #0d0d0d;
    --surface:    #1a1a19;
    --surface-2:  #212120;
    --ink:        #ffffff;
    --ink-2:      #c3c2b7;
    --muted:      #898781;
    --border:     rgba(255,255,255,0.10);
    --blue:       #3987e5;   /* monitor */
    --blue-100:   #cde2fb;
    --yellow:     #c98500;   /* victim */
    --red:        #e66767;   /* attacker identity */
    --good:       #0ca30c;   /* benign status */
    --critical:   #d03b3b;   /* attack status */
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ overflow-x: hidden; }}
  body {{
    background: radial-gradient(circle at top left, #14181c 0%, var(--page) 55%);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    margin: 0;
    padding: 28px clamp(16px, 4vw, 40px) 48px;
  }}
  header {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:22px; }}
  h1 {{ font-size:21px; font-weight:700; margin:0; letter-spacing:-0.01em; }}
  h1 .accent {{ color: var(--blue); }}
  .sub {{ color: var(--muted); font-size:13px; margin-top:4px; }}
  .live {{ display:flex; align-items:center; gap:8px; font-size:13px; color: var(--ink-2); background:var(--surface); border:1px solid var(--border); padding:6px 14px; border-radius:999px; }}
  .live .dot {{ width:8px; height:8px; border-radius:50%; background:var(--good); box-shadow:0 0 0 0 rgba(12,163,12,0.6); animation:pulse 2s infinite; }}
  .reset-btn {{
    font: inherit; font-size:13px; color:var(--ink-2); background:var(--surface);
    border:1px solid var(--border); padding:6px 14px; border-radius:999px; cursor:pointer;
  }}
  .reset-btn:hover {{ color:var(--ink); border-color:rgba(255,255,255,0.22); }}
  @keyframes pulse {{
    0%   {{ box-shadow:0 0 0 0 rgba(12,163,12,0.55); }}
    70%  {{ box-shadow:0 0 0 7px rgba(12,163,12,0); }}
    100% {{ box-shadow:0 0 0 0 rgba(12,163,12,0); }}
  }}

  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; }}

  .legend {{ display:flex; gap:22px; padding:14px 18px; margin-bottom:18px; flex-wrap:wrap; font-size:13px; align-items:center; }}
  .legend .item {{ display:flex; align-items:center; gap:8px; }}
  .legend .role {{ width:9px; height:9px; border-radius:50%; flex:none; }}
  .legend .role.attacker {{ background:var(--red); }}
  .legend .role.victim {{ background:var(--yellow); }}
  .legend .role.monitor {{ background:var(--blue); }}
  .legend .ip {{ font-variant-numeric:tabular-nums; color:var(--ink); font-weight:600; }}
  .legend .note {{ color:var(--muted); margin-left:auto; }}

  .progress-card {{ padding:16px 18px; margin-bottom:20px; }}
  .progress-top {{ display:flex; justify-content:space-between; align-items:baseline; font-size:13px; color:var(--ink-2); margin-bottom:9px; gap:12px; flex-wrap:wrap; }}
  .progress-top b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
  .bar-track {{ background:var(--surface-2); border-radius:999px; height:8px; overflow:hidden; }}
  .bar-fill {{ background:linear-gradient(90deg, var(--blue-100), var(--blue)); height:100%; width:0%; transition:width 0.4s ease; border-radius:999px; }}
  .last-flow {{ color:var(--muted); font-size:12.5px; margin-top:10px; font-variant-numeric:tabular-nums; }}
  .last-flow b {{ color:var(--ink-2); font-weight:600; }}

  .stats {{ display:grid; grid-template-columns:repeat(3, minmax(160px, 1fr)); gap:16px; margin-bottom:24px; }}
  .stat {{ padding:18px 20px; }}
  .stat-label {{ font-size:12.5px; color:var(--muted); display:flex; align-items:center; gap:6px; margin-bottom:8px; }}
  .stat-label .swatch {{ width:8px; height:8px; border-radius:2px; }}
  .stat .num {{ font-size:34px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1; }}
  .stat.windows .swatch {{ background:var(--blue); }}
  .stat.attack .swatch {{ background:var(--critical); }}
  .stat.attack .num {{ color:var(--critical); }}
  .stat.benign .swatch {{ background:var(--good); }}
  .stat.benign .num {{ color:var(--good); }}

  .panels {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:20px; }}
  .panel {{ padding:16px 4px 4px; }}
  .panel-head {{ display:flex; justify-content:space-between; align-items:center; padding:0 18px 12px; }}
  .panel-head h3 {{ margin:0; font-size:14px; font-weight:600; }}
  .panel-head .count {{ font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }}
  .table-scroll {{ overflow-x:auto; max-height:420px; overflow-y:auto; }}
  table {{ width:100%; border-collapse:collapse; }}
  th, td {{ text-align:left; padding:9px 18px; font-size:13px; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  th {{ color:var(--muted); font-weight:600; font-size:11.5px; text-transform:uppercase; letter-spacing:0.04em; position:sticky; top:0; background:var(--surface); }}
  tbody tr {{ border-top:1px solid var(--border); }}
  tbody tr.new-row {{ animation: flash 1.4s ease-out; }}
  @keyframes flash {{ from {{ background:rgba(208,59,59,0.18); }} to {{ background:transparent; }} }}
  .src {{ color:var(--ink-2); }}
  .src.is-attacker {{ color:var(--red); font-weight:700; }}
  .dst {{ color:var(--muted); }}

  .badge {{ display:inline-flex; align-items:center; gap:6px; padding:3px 9px; border-radius:999px; font-size:11.5px; font-weight:600; }}
  .badge .dot {{ width:6px; height:6px; border-radius:50%; }}
  .badge.attack {{ color:var(--critical); background:rgba(208,59,59,0.14); }}
  .badge.attack .dot {{ background:var(--critical); }}
  .badge.benign {{ color:var(--good); background:rgba(12,163,12,0.12); }}
  .badge.benign .dot {{ background:var(--good); }}

  .empty {{ color:var(--muted); font-size:13px; padding:24px 18px; text-align:center; }}
</style>
</head>
<body>
  <header>
    <div>
      <h1>GraphSAGE <span class="accent">NIDS</span> &mdash; Giam sat tan cong real-time</h1>
      <div class="sub">Zeek conn.log &rarr; NetFlow V2 &rarr; E-GraphSAGE ({DATASET})</div>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
      <div class="live"><span class="dot"></span><span id="live-text">Dang ket noi...</span></div>
      <button id="reset-btn" class="reset-btn">&#8635; Reset hien thi</button>
    </div>
  </header>

  <div class="card legend">
    <div class="item"><span class="role attacker"></span>Attacker <span class="ip">{ATTACKER_IP}</span></div>
    <div class="item"><span class="role victim"></span>Victim <span class="ip">{VICTIM_IP}</span></div>
    <div class="item"><span class="role monitor"></span>Monitor <span class="ip">{MONITOR_IP}</span></div>
    <div class="note">Traffic toi cong :{DASHBOARD_PORT} (trang nay) da duoc loai khoi phan tich</div>
  </div>

  <div class="card progress-card">
    <div class="progress-top">
      <span id="progress-text">Dang cho du lieu...</span>
      <span><b id="flows-remaining">&mdash;</b> flow nua toi cua so tiep theo</span>
    </div>
    <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
    <div class="last-flow" id="last-flow">&nbsp;</div>
  </div>

  <div class="stats">
    <div class="card stat windows">
      <div class="stat-label"><span class="swatch"></span>Cua so da xu ly</div>
      <div class="num" id="total-windows">0</div>
    </div>
    <div class="card stat attack">
      <div class="stat-label"><span class="swatch"></span>Flow gan nhan Attack</div>
      <div class="num" id="total-attack">0</div>
    </div>
    <div class="card stat benign">
      <div class="stat-label"><span class="swatch"></span>Flow Benign</div>
      <div class="num" id="total-benign">0</div>
    </div>
  </div>

  <div class="panels">
    <div class="card panel">
      <div class="panel-head"><h3>Traffic gan nhat</h3><span class="count">mau cua so vua xu ly</span></div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Gio</th><th>Nguon</th><th>Dich</th><th>Ket qua</th></tr></thead>
          <tbody id="recent-body"><tr><td class="empty" colspan="4">Dang cho cua so dau tien&hellip;</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="card panel">
      <div class="panel-head"><h3>Canh bao Attack</h3><span class="count" id="attack-count">0 canh bao</span></div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Gio</th><th>Nguon</th><th>Dich</th><th>Xac suat</th></tr></thead>
          <tbody id="attack-body"><tr><td class="empty" colspan="4">Chua co canh bao nao</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

<script>
  const ATTACKER_IP = "{ATTACKER_IP}";
  let totalWindows = 0, totalAttack = 0, totalBenign = 0, attackAlerts = 0;
  const ws = new WebSocket(`ws://${{location.host}}/ws`);

  function fmtTime(ts) {{
    return new Date(parseFloat(ts) * 1000).toLocaleTimeString('vi-VN', {{ hour12:false }});
  }}
  function srcCell(ip, port) {{
    const cls = ip === ATTACKER_IP ? ' class="src is-attacker"' : ' class="src"';
    return `<td${{cls}}>${{ip}}:${{port}}</td>`;
  }}
  function badge(isAttack) {{
    return isAttack
      ? '<span class="badge attack"><span class="dot"></span>Attack</span>'
      : '<span class="badge benign"><span class="dot"></span>Benign</span>';
  }}

  ws.onopen = () => {{ document.getElementById("live-text").textContent = "Live"; }};

  document.getElementById("reset-btn").onclick = () => {{
    totalWindows = 0; totalAttack = 0; totalBenign = 0; attackAlerts = 0;
    document.getElementById("total-windows").textContent = "0";
    document.getElementById("total-attack").textContent = "0";
    document.getElementById("total-benign").textContent = "0";
    document.getElementById("attack-count").textContent = "0 canh bao";
    document.getElementById("attack-body").innerHTML = '<tr><td class="empty" colspan="4">Chua co canh bao nao</td></tr>';
    document.getElementById("recent-body").innerHTML = '<tr><td class="empty" colspan="4">Dang cho cua so tiep theo&hellip;</td></tr>';
  }};

  ws.onmessage = (event) => {{
    const msg = JSON.parse(event.data);
    if (msg.type === "progress") {{
      const pct = 100 * (msg.window_size - msg.flows_until_next_window) / msg.window_size;
      document.getElementById("bar").style.width = Math.max(0, Math.min(100, pct)) + "%";
      document.getElementById("progress-text").textContent = `Tong flow da nhan: ${{msg.total_flows.toLocaleString('vi-VN')}}`;
      document.getElementById("flows-remaining").textContent = msg.flows_until_next_window.toLocaleString('vi-VN');
      if (msg.last_flow) {{
        document.getElementById("last-flow").innerHTML =
          `Flow moi nhat: <b>${{msg.last_flow.src}}</b> &rarr; <b>${{msg.last_flow.dst}}</b> &middot; proto ${{msg.last_flow.proto}}`;
      }}
    }} else if (msg.type === "window_result") {{
      totalWindows += 1;
      totalAttack += msg.num_attack;
      totalBenign += (msg.num_flows - msg.num_attack);
      attackAlerts += msg.attacks.length;
      document.getElementById("total-windows").textContent = totalWindows.toLocaleString('vi-VN');
      document.getElementById("total-attack").textContent = totalAttack.toLocaleString('vi-VN');
      document.getElementById("total-benign").textContent = totalBenign.toLocaleString('vi-VN');
      document.getElementById("attack-count").textContent = `${{attackAlerts.toLocaleString('vi-VN')}} canh bao`;

      const attackBody = document.getElementById("attack-body");
      if (msg.attacks.length && attackBody.querySelector(".empty")) attackBody.innerHTML = "";
      for (const a of msg.attacks) {{
        const tr = document.createElement("tr");
        tr.className = "new-row";
        tr.innerHTML = `<td>${{fmtTime(a.ts)}}</td>${{srcCell(a.IPV4_SRC_ADDR, a.L4_SRC_PORT)}}` +
          `<td class="dst">${{a.IPV4_DST_ADDR}}:${{a.L4_DST_PORT}}</td><td>${{(a.pred_proba*100).toFixed(1)}}%</td>`;
        attackBody.prepend(tr);
      }}
      while (attackBody.rows.length > 40) attackBody.deleteRow(-1);

      const recentBody = document.getElementById("recent-body");
      recentBody.innerHTML = "";
      for (const r of msg.recent.slice().reverse()) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{fmtTime(r.ts)}}</td>${{srcCell(r.IPV4_SRC_ADDR, r.L4_SRC_PORT)}}` +
          `<td class="dst">${{r.IPV4_DST_ADDR}}:${{r.L4_DST_PORT}}</td><td>${{badge(r.pred_label === 1)}}</td>`;
        recentBody.appendChild(tr);
      }}
    }}
  }};
  ws.onclose = () => {{
    document.getElementById("live-text").textContent = "Mat ket noi";
    document.querySelector(".live .dot").style.background = "var(--critical)";
  }};
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _INDEX_HTML
