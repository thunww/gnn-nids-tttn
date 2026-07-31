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
<title>GraphSAGE NIDS - Demo real-time</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background:#0b0f14; color:#e6edf3; font-family: system-ui, sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h3 {{ margin-bottom:8px; }}
  .sub {{ color:#8b949e; margin-bottom:16px; }}
  .legend {{ display:flex; gap:16px; background:#161b22; border-radius:8px; padding:12px 16px; margin-bottom:20px; flex-wrap:wrap; font-size:14px; }}
  .legend span.tag {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
  .tag.attacker {{ background:#f85149; }}
  .tag.victim {{ background:#d29922; }}
  .tag.monitor {{ background:#2f81f7; }}
  .progress-wrap {{ background:#161b22; border-radius:8px; padding:16px; margin-bottom:20px; }}
  .bar-bg {{ background:#21262d; border-radius:6px; height:14px; overflow:hidden; margin-top:8px; }}
  .bar-fg {{ background:#2f81f7; height:100%; width:0%; transition:width 0.3s; }}
  .last-flow {{ color:#8b949e; font-size:13px; margin-top:6px; }}
  .stats {{ display:flex; gap:24px; margin-bottom:24px; flex-wrap:wrap; }}
  .stat {{ background:#161b22; border-radius:8px; padding:16px 24px; text-align:center; flex:1; min-width:140px; }}
  .stat .num {{ font-size:32px; font-weight:700; }}
  .stat.attack .num {{ color:#f85149; }}
  .stat.benign .num {{ color:#3fb950; }}
  .panels {{ display:flex; gap:24px; flex-wrap:wrap; }}
  .panel {{ flex:1; min-width:420px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #21262d; font-size:13px; white-space:nowrap; }}
  th {{ color:#8b949e; font-weight:600; }}
  tr.new-row {{ animation: flash 1.2s ease-out; }}
  tr.row-attack td {{ color:#f85149; }}
  tr.row-benign td {{ color:#7d8590; }}
  .src-attacker {{ color:#f85149 !important; font-weight:700; }}
  @keyframes flash {{ from {{ background:#f8514933; }} to {{ background:transparent; }} }}
</style>
</head>
<body>
  <h1>GraphSAGE NIDS &mdash; Demo phat hien tan cong real-time</h1>
  <div class="sub">Nguon: Zeek conn.log &rarr; NetFlow V2 &rarr; E-GraphSAGE (nf-cse-cic-ids2018-v2)</div>

  <div class="legend">
    <div><span class="tag attacker"></span>Attacker: <b>{ATTACKER_IP}</b></div>
    <div><span class="tag victim"></span>Victim: <b>{VICTIM_IP}</b></div>
    <div><span class="tag monitor"></span>Monitor (may nay): <b>{MONITOR_IP}</b></div>
    <div style="color:#8b949e">Traffic toi cong :8000 (chinh trang nay) da duoc loai khoi phan tich</div>
  </div>

  <div class="progress-wrap">
    <div id="progress-text">Dang cho ket noi...</div>
    <div class="bar-bg"><div class="bar-fg" id="bar"></div></div>
    <div class="last-flow" id="last-flow"></div>
  </div>

  <div class="stats">
    <div class="stat"><div class="num" id="total-windows">0</div>So cua so da xu ly</div>
    <div class="stat attack"><div class="num" id="total-attack">0</div>Flow bi gan nhan Attack</div>
    <div class="stat benign"><div class="num" id="total-benign">0</div>Flow Benign</div>
  </div>

  <div class="panels">
    <div class="panel">
      <h3>Traffic gan nhat (mau moi cua so)</h3>
      <table>
        <thead><tr><th>Gio</th><th>Nguon</th><th>Dich</th><th>Ket qua</th></tr></thead>
        <tbody id="recent-body"></tbody>
      </table>
    </div>
    <div class="panel">
      <h3>Cac flow gan nhan Attack</h3>
      <table>
        <thead><tr><th>Gio</th><th>Nguon</th><th>Dich</th><th>Xac suat</th></tr></thead>
        <tbody id="attack-body"></tbody>
      </table>
    </div>
  </div>

<script>
  const ATTACKER_IP = "{ATTACKER_IP}";
  let totalWindows = 0, totalAttack = 0, totalBenign = 0;
  const ws = new WebSocket(`ws://${{location.host}}/ws`);

  function fmtTime(ts) {{
    return new Date(parseFloat(ts) * 1000).toLocaleTimeString('vi-VN');
  }}
  function srcCell(ip, port) {{
    const cls = ip === ATTACKER_IP ? ' class="src-attacker"' : '';
    return `<td${{cls}}>${{ip}}:${{port}}</td>`;
  }}

  ws.onmessage = (event) => {{
    const msg = JSON.parse(event.data);
    if (msg.type === "progress") {{
      const pct = 100 * (msg.window_size - msg.flows_until_next_window) / msg.window_size;
      document.getElementById("bar").style.width = Math.max(0, Math.min(100, pct)) + "%";
      document.getElementById("progress-text").textContent =
        `Tong flow da nhan: ${{msg.total_flows}} -- can them ${{msg.flows_until_next_window}} flow de co cua so tiep theo (${{msg.window_size}} flow/cua so)`;
      if (msg.last_flow) {{
        document.getElementById("last-flow").textContent =
          `Flow moi nhat: ${{msg.last_flow.src}} -> ${{msg.last_flow.dst}} (proto ${{msg.last_flow.proto}})`;
      }}
    }} else if (msg.type === "window_result") {{
      totalWindows += 1;
      totalAttack += msg.num_attack;
      totalBenign += (msg.num_flows - msg.num_attack);
      document.getElementById("total-windows").textContent = totalWindows;
      document.getElementById("total-attack").textContent = totalAttack;
      document.getElementById("total-benign").textContent = totalBenign;

      const attackBody = document.getElementById("attack-body");
      for (const a of msg.attacks) {{
        const tr = document.createElement("tr");
        tr.className = "new-row";
        tr.innerHTML = `<td>${{fmtTime(a.ts)}}</td>${{srcCell(a.IPV4_SRC_ADDR, a.L4_SRC_PORT)}}` +
          `<td>${{a.IPV4_DST_ADDR}}:${{a.L4_DST_PORT}}</td><td>${{(a.pred_proba*100).toFixed(1)}}%</td>`;
        attackBody.prepend(tr);
      }}
      while (attackBody.rows.length > 30) attackBody.deleteRow(-1);

      const recentBody = document.getElementById("recent-body");
      recentBody.innerHTML = "";
      for (const r of msg.recent.slice().reverse()) {{
        const tr = document.createElement("tr");
        tr.className = r.pred_label === 1 ? "row-attack" : "row-benign";
        tr.innerHTML = `<td>${{fmtTime(r.ts)}}</td>${{srcCell(r.IPV4_SRC_ADDR, r.L4_SRC_PORT)}}` +
          `<td>${{r.IPV4_DST_ADDR}}:${{r.L4_DST_PORT}}</td><td>${{r.pred_label === 1 ? "ATTACK" : "benign"}}</td>`;
        recentBody.appendChild(tr);
      }}
    }}
  }};
  ws.onclose = () => {{ document.getElementById("progress-text").textContent = "Mat ket noi toi server."; }};
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _INDEX_HTML
