"""Phase D: chuyen 1 dong Zeek conn.log -> 39 cot dac trung NetFlow V2 + 4 cot dinh danh.

Xem docs/graphsage/05_demo_realtime_setup.md muc 6 cho boi canh day du va bang anh xa.
Zeek khong xuat het 39 cot NetFlow V2 (thieu TTL, histogram kich thuoc goi, TCP window...) --
cac cot khong the tinh duoc dung "0" (dung khi 0 la gia tri hop ly, vd RETRANSMITTED_*, DNS_*)
hoac trung binh cua tap train sau khi clip (lay truc tiep tu scaler.mean_, dung khi 0 se la
outlier phi ly, vd MIN_TTL/MAX_TTL/TCP_WIN_MAX_*) -- xem DEFAULT_MEAN_COLS/DEFAULT_ZERO_COLS.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from pathlib import Path

# Thu tu 39 cot PHAI khop dung voi scaler.feature_names_in_ (data/processed/<dataset>/scaler.joblib)
# -- day la thu tu cot da dung de fit StandardScaler luc train, sai thu tu se lam sai het suy luan.
FEATURE_COLS = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]

IDENTIFIER_COLS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"]

# Cot khong tinh duoc tu conn.log, "0" la gia tri hop ly that su (khong phai xap xi) vi hau
# het flow that su khong retransmit / khong phai DNS-FTP.
DEFAULT_ZERO_COLS = [
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]

# Cot ap dung cho GAN NHU MOI flow (TTL/TCP window luon co gia tri that trong goi tin thuc)
# nhung Zeek khong xuat mac dinh -- dung "0" se la outlier phi ly sau khi chuan hoa, nen dung
# trung binh tap train (sau clip 99th percentile) lay truc tiep tu scaler da fit.
DEFAULT_MEAN_COLS = ["MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]

# TCP flag bit -> xap xi tu ky tu "history" cua Zeek (khong phai gia tri co bit chinh xac tu
# goi tin that, vi Zeek khong xuat flag byte truc tiep). Quy uoc history: CHU HOA = ben goi ket
# noi (originator), chu thuong = ben nhan (responder) -- kiem chung tu du lieu that thu duoc
# tren server (vd "Sr" cho REJ = originator gui SYN, responder tra loi RST).
_TCP_FLAG_BITS = {
    "F": 0x01,  # FIN
    "S": 0x02,  # SYN
    "R": 0x04,  # RST
    "D": 0x08,  # xap xi PSH (goi co du lieu thuong kem PSH)
    "A": 0x10,  # ACK
    "H": 0x12,  # SYN+ACK = SYN|ACK
}

# Nguong kich thuoc goi (bytes) cho 5 cot histogram NUM_PKTS_*_BYTES cua NetFlow V2.
_PKT_SIZE_BUCKETS = [128, 256, 512, 1024, 1514]
_HISTOGRAM_COLS = [
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
]


def _num(value: str, cast=float) -> float:
    """Zeek dung '-' cho truong khong co gia tri; parse an toan, mac dinh 0."""
    if value in ("-", "", None):
        return 0
    return cast(value)


def _parse_tcp_flags(history: str) -> tuple[int, int, int]:
    """Tra ve (TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS) xap xi tu chuoi history."""
    flags = client_flags = server_flags = 0
    for ch in history:
        bit = _TCP_FLAG_BITS.get(ch.upper())
        if not bit:
            continue
        flags |= bit
        if ch.isupper():
            client_flags |= bit
        else:
            server_flags |= bit
    return flags, client_flags, server_flags


def _pkt_size_histogram(avg_pkt_len: float, total_pkts: int) -> dict[str, int]:
    """Xap xi: khong co do dai tung goi rieng le, gia dinh moi goi trong flow xap xi bang
    kich thuoc trung binh (tong byte IP / tong so goi) va don het vao 1 bucket duy nhat."""
    histogram = dict.fromkeys(_HISTOGRAM_COLS, 0)
    if total_pkts <= 0:
        return histogram
    for threshold, col in zip(_PKT_SIZE_BUCKETS, _HISTOGRAM_COLS):
        if avg_pkt_len <= threshold:
            histogram[col] = total_pkts
            break
    else:
        histogram[_HISTOGRAM_COLS[-1]] = total_pkts
    return histogram


def convert_row(row: dict[str, str], scaler_mean: dict[str, float]) -> dict[str, float | str]:
    """Chuyen 1 dong conn.log (dict tu #fields header, xem read_conn_log()) thanh dict day du
    4 cot dinh danh + 39 cot dac trung NetFlow V2 (chua chuan hoa -- ap dung apply_scale() sau).

    scaler_mean: dict ten_cot -> trung binh tap train (StandardScaler.mean_), dung lam gia tri
    mac dinh cho DEFAULT_MEAN_COLS (xem module docstring).
    """
    proto = int(_num(row["ip_proto"], int))
    orig_bytes = _num(row.get("orig_bytes", "-"))
    resp_bytes = _num(row.get("resp_bytes", "-"))
    orig_pkts = int(_num(row.get("orig_pkts", "-"), int))
    resp_pkts = int(_num(row.get("resp_pkts", "-"), int))
    orig_ip_bytes = _num(row.get("orig_ip_bytes", "-"))
    resp_ip_bytes = _num(row.get("resp_ip_bytes", "-"))
    duration_sec = _num(row.get("duration", "-"))
    duration_ms = duration_sec * 1000
    history = row.get("history", "-")
    history = "" if history == "-" else history

    total_pkts = orig_pkts + resp_pkts
    total_ip_bytes = orig_ip_bytes + resp_ip_bytes
    avg_pkt_len = total_ip_bytes / total_pkts if total_pkts > 0 else 0.0

    tcp_flags, client_flags, server_flags = _parse_tcp_flags(history) if proto == 6 else (0, 0, 0)

    if proto == 1:  # icmp: Zeek dung lai id.orig_p/id.resp_p de chua type/code (xac nhan tren du lieu that)
        icmp_type = int(_num(row["id.orig_p"], int))
        icmp_code = int(_num(row["id.resp_p"], int))
    else:
        icmp_type = icmp_code = 0

    out = {
        "IPV4_SRC_ADDR": row["id.orig_h"],
        "L4_SRC_PORT": int(_num(row["id.orig_p"], int)),
        "IPV4_DST_ADDR": row["id.resp_h"],
        "L4_DST_PORT": int(_num(row["id.resp_p"], int)),
        "PROTOCOL": proto,
        "L7_PROTO": 0,  # khong co bang tra nDPI protocol ID dang tin cay -- xem docs muc 6.3
        "IN_BYTES": orig_bytes,
        "IN_PKTS": orig_pkts,
        "OUT_BYTES": resp_bytes,
        "OUT_PKTS": resp_pkts,
        "TCP_FLAGS": tcp_flags,
        "CLIENT_TCP_FLAGS": client_flags,
        "SERVER_TCP_FLAGS": server_flags,
        "FLOW_DURATION_MILLISECONDS": duration_ms,
        "DURATION_IN": duration_ms,  # Zeek khong tach duration rieng tung chieu -- xap xi bang ca flow
        "DURATION_OUT": duration_ms,
        "LONGEST_FLOW_PKT": avg_pkt_len,  # khong co do dai tung goi -- xap xi bang trung binh flow
        "SHORTEST_FLOW_PKT": avg_pkt_len,
        "MIN_IP_PKT_LEN": avg_pkt_len,
        "MAX_IP_PKT_LEN": avg_pkt_len,
        "SRC_TO_DST_SECOND_BYTES": orig_bytes / duration_sec if duration_sec > 0 else orig_bytes,
        "DST_TO_SRC_SECOND_BYTES": resp_bytes / duration_sec if duration_sec > 0 else resp_bytes,
        "SRC_TO_DST_AVG_THROUGHPUT": (orig_bytes * 8) / duration_sec if duration_sec > 0 else 0,
        "DST_TO_SRC_AVG_THROUGHPUT": (resp_bytes * 8) / duration_sec if duration_sec > 0 else 0,
        "ICMP_TYPE": icmp_type * 256 + icmp_code,
        "ICMP_IPV4_TYPE": icmp_type,
    }
    out.update(_pkt_size_histogram(avg_pkt_len, total_pkts))
    for col in DEFAULT_ZERO_COLS:
        out[col] = 0
    for col in DEFAULT_MEAN_COLS:
        out[col] = scaler_mean[col]

    missing = (set(IDENTIFIER_COLS) | set(FEATURE_COLS)) - out.keys()
    assert not missing, f"Thieu cot: {missing}"
    return out


def convert_rows_to_dataframe(rows: list[dict[str, str]], scaler_mean: dict[str, float]) -> pd.DataFrame:
    """Chuyen 1 danh sach dong conn.log tho thanh DataFrame dung thu tu cot IDENTIFIER_COLS +
    FEATURE_COLS -- dau vao truc tiep cho etl.scale.apply_scale() roi graph.build_graph()."""
    converted = [convert_row(row, scaler_mean) for row in rows]
    return pd.DataFrame(converted, columns=IDENTIFIER_COLS + FEATURE_COLS)


def get_scaler_mean(scaler) -> dict[str, float]:
    """Lay dict ten_cot -> trung binh tap train tu 1 StandardScaler da fit (dung cho DEFAULT_MEAN_COLS)."""
    return dict(zip(scaler.feature_names_in_, scaler.mean_))


def read_conn_log(path: Path) -> Iterator[dict[str, str]]:
    """Doc conn.log dang TSV cua Zeek, tu dong lay dung ten cot tu dong '#fields'."""
    with open(path, encoding="utf-8") as f:
        fields: list[str] | None = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                continue
            if line.startswith("#"):
                continue
            values = line.split("\t")
            if fields is None or len(values) != len(fields):
                continue
            yield dict(zip(fields, values))
