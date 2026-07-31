# Dựng demo real-time — tài liệu tự chứa cho agent/người thực hiện trên server

**Đọc file này là đủ để bắt tay vào làm, không cần xem lại các file `docs/decisions.md`/`docs/phases/*.md` (đó là nhật ký làm việc, không phải hướng dẫn thực thi).**

## 1. Bối cảnh dự án (tóm tắt)

Đồ án: hệ thống phát hiện xâm nhập mạng (NIDS) dùng mạng nơ-ron đồ thị (GNN). Bài toán: **phân loại nhị phân** 1 luồng mạng (flow) là **Benign** (bình thường) hay **Attack** (bị tấn công). Kiến trúc dùng: **E-GraphSAGE** (chi tiết công thức xem `docs/graphsage/02_kien_truc_mo_hinh.md`). Đã huấn luyện xong và có kết quả đầy đủ trên 2 bộ dữ liệu công khai (xem `docs/graphsage/03_ket_qua.md`) — **KHÔNG cần train lại gì**, việc còn lại là dựng demo phát hiện tấn công **real-time** để trình chiếu trước hội đồng bảo vệ đồ án.

## 2. Đã có sẵn trong `src/` — không cần viết lại

```
src/etl/          -- doc CSV tho (NetFlow V2) -> lam sach -> chia train/val/test -> chuan hoa (StandardScaler)
  config.py       -- ten cot: IDENTIFIER_COLS, LABEL_COL="Label", ATTACK_COL, DATASETS
  load.py         -- doc CSV
  clean.py        -- lam sach + ma hoa Attack_encoded (khong dung cho model nhi phan, chi de tham khao)
  split.py        -- chia 70/15/15
  scale.py        -- fit_scale() / apply_scale() -- StandardScaler + clip outlier 99th percentile
  run_etl.py       -- chay toan bo, luu scaler.joblib + upper_bound.joblib (can cho demo, xem muc 4)

src/graph/        -- bien du lieu bang -> do thi PyTorch Geometric
  nodes.py        -- gan ID cho tung dinh (cap IP:port)
  edges.py        -- xay dung canh (edge_index) + dac trung canh (edge_attr, 39 chieu)
  node_features.py -- tinh dac trung node (43 chieu = 4 cau truc + 39 tong hop tu canh)
  build_graph.py  -- ghep thanh 1 object Data hoan chinh, nhan y = cot Label
  windowing.py    -- cat du lieu thanh cua so truot (WINDOW_SIZE flow/cua so)
  run_graph_builder.py -- chay toan bo Graph Builder

src/models/
  gnn_config.py   -- SIEU THAM SO THAT (xem muc 5 duoi day de tai dung dung)
  sage_layer.py   -- EGraphSAGEConv (MessagePassing tuy chinh, PyG SAGEConv khong ho tro edge_dim)
  graphsage.py    -- GraphSAGEEdgeClassifier (kien truc chinh, DANG DUNG)
  gcn.py, gat.py  -- KHONG DUNG NUA (lich su cu, da bo -- xem docs/decisions.md)
  baselines.py    -- Random Forest + XGBoost
  metrics.py      -- compute_full_metrics() -- Accuracy/Precision/Recall/F1-macro/AUC-ROC/MCC
  train_gnn.py    -- vong lap train GraphSAGE (KHONG can chay lai, model da train xong)
  train_baseline.py -- train RF/XGBoost (KHONG can chay lai)
  evaluate_test.py -- **DUNG LAM KHUNG THAM KHAO cho demo** -- nap model da train, chay suy luan,
                       tinh metrics. Xem ham evaluate_baseline(), evaluate_graphsage() de biet cach
                       nap dung model + gia usthiet ve shape dau vao.
  evaluate_cross_dataset.py -- danh gia cheo bo du lieu (khong lien quan demo real-time)
```

## 3. Model đã train xong — vị trí file cần copy sang server

```
data/processed/nf-cse-cic-ids2018-v2/models/random_forest.joblib
data/processed/nf-cse-cic-ids2018-v2/models/xgboost.joblib
data/processed/nf-cse-cic-ids2018-v2/models/graphsage_best.pt
data/processed/nf-cse-cic-ids2018-v2/scaler.joblib
data/processed/nf-cse-cic-ids2018-v2/upper_bound.joblib

data/processed/nf-unsw-nb15-v2/models/random_forest.joblib
data/processed/nf-unsw-nb15-v2/models/xgboost.joblib
data/processed/nf-unsw-nb15-v2/models/graphsage_best.pt
data/processed/nf-unsw-nb15-v2/scaler.joblib
data/processed/nf-unsw-nb15-v2/upper_bound.joblib
```

**Demo real-time nên dùng model của bộ `nf-cse-cic-ids2018-v2`** (kết quả tốt hơn, F1-macro 0.988 so với 0.978 — xem `03_ket_qua.md`), trừ khi có lý do khác.

## 4. Kế hoạch tổng thể — 7 phase (A→G)

| Phase | Việc | Ai làm |
|---|---|---|
| A | Dựng 3 máy ảo VMware (tấn công/nạn nhân/giám sát), mạng host-only cô lập khỏi Internet, bật Promiscuous Mode máy giám sát, cài Zeek + Suricata + ET Open Rules | **Người dùng** (không thể làm qua code) |
| B | Chốt nội dung cụ thể 5 kịch bản tấn công + công cụ dùng | **Người dùng** |
| C | Chạy Zeek + Suricata song song, thực hiện 5 kịch bản, ghi lại chính xác mốc thời gian từng kịch bản | **Người dùng** |
| D | Viết code chuyển log Zeek → đúng 43 cột đặc trưng NetFlow V2 (khớp schema đã train) | **Agent/code** — chi tiết mục 6 |
| E | Script suy luận offline (kiểm chứng đúng trước khi làm live) + script so sánh với Suricata | **Agent/code** — chi tiết mục 7 |
| F | API real-time (FastAPI, đã có sẵn trong `requirements.txt` nhưng CHƯA có code) — tail log Zeek liên tục, chuyển đổi, gọi model, hiển thị kết quả ngay | **Agent/code** — chi tiết mục 8 |
| G | Cập nhật docs với kết quả TN6 | **Agent/code** |

**Thứ tự bắt buộc:** D → E (kiểm chứng offline đúng trước) → F (mới làm live). Không nhảy thẳng vào F vì nếu logic chuyển đổi đặc trưng (D) sai, demo live sẽ cho kết quả vô nghĩa mà không biết ngay.

## 5. Siêu tham số/hằng số cần dùng đúng khi viết code (Phase D-F)

Từ `src/models/gnn_config.py` (KHÔNG được đoán/tự đổi, phải khớp đúng lúc train):

```python
NODE_FEATURE_DIM = 43
EDGE_FEATURE_DIM = 39
NUM_LAYERS = 2
HIDDEN_DIM = 128            # dung cho nf-cse-cic-ids2018-v2
HIDDEN_DIM_BY_DATASET = {"nf-unsw-nb15-v2": 32}   # UNSW-NB15 dung 32, CSE-CIC dung HIDDEN_DIM=128
```

Cách nạp đúng model GraphSAGE (xem `evaluate_test.py` hàm `evaluate_graphsage` để chép chính xác pattern):

```python
from models.graphsage import GraphSAGEEdgeClassifier
model = GraphSAGEEdgeClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim, 2, NUM_LAYERS)
model.load_state_dict(torch.load(".../graphsage_best.pt", map_location=device, weights_only=True))
model.eval()
```

## 6. Phase D — Chuyển đổi log Zeek → đặc trưng NetFlow V2 (chi tiết kỹ thuật)

### 6.1. Schema đích (39 cột đặc trưng cạnh, đúng thứ tự đã dùng để fit scaler)

```
PROTOCOL, L7_PROTO, IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS, TCP_FLAGS, CLIENT_TCP_FLAGS,
SERVER_TCP_FLAGS, FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT, MIN_TTL, MAX_TTL,
LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, MAX_IP_PKT_LEN, SRC_TO_DST_SECOND_BYTES,
DST_TO_SRC_SECOND_BYTES, RETRANSMITTED_IN_BYTES, RETRANSMITTED_IN_PKTS, RETRANSMITTED_OUT_BYTES,
RETRANSMITTED_OUT_PKTS, SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT,
NUM_PKTS_UP_TO_128_BYTES, NUM_PKTS_128_TO_256_BYTES, NUM_PKTS_256_TO_512_BYTES,
NUM_PKTS_512_TO_1024_BYTES, NUM_PKTS_1024_TO_1514_BYTES, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT,
ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE
```

Cộng thêm 4 cột định danh: `IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT` (không đưa vào model, chỉ dùng để dựng đồ thị — xem `src/graph/nodes.py`).

### 6.2. Nguồn dữ liệu Zeek — `conn.log` là chính

Các trường chuẩn của Zeek `conn.log`: `ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto, service, duration, orig_bytes, resp_bytes, conn_state, missed_bytes, history, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes`.

### 6.3. Bảng ánh xạ đề xuất (CẦN NGƯỜI VIẾT CODE KIỂM CHỨNG LẠI, không phải tuyệt đối chính xác — Zeek và NetFlow V2 không có mapping 1:1 hoàn hảo)

| Cột NetFlow V2 | Nguồn từ Zeek | Ghi chú |
|---|---|---|
| `IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IPV4_DST_ADDR`, `L4_DST_PORT` | `id.orig_h`, `id.orig_p`, `id.resp_h`, `id.resp_p` | Trực tiếp |
| `PROTOCOL` | `proto` | Cần map chuỗi ("tcp"/"udp"/"icmp") → số hiệu IANA protocol (tcp=6, udp=17, icmp=1) |
| `L7_PROTO` | `service` | Zeek dùng tên dịch vụ (chuỗi), NF-v2 dùng mã số — cần bảng tra riêng hoặc gán giá trị mặc định nếu không map được |
| `IN_BYTES`, `OUT_BYTES` | `orig_bytes`, `resp_bytes` | Trực tiếp (chú ý chiều: orig=máy khởi tạo kết nối) |
| `IN_PKTS`, `OUT_PKTS` | `orig_pkts`, `resp_pkts` | Trực tiếp |
| `FLOW_DURATION_MILLISECONDS` | `duration` × 1000 | Zeek tính bằng giây |
| `MIN_TTL`, `MAX_TTL` | **Không có sẵn trong `conn.log`** | Cần bật thêm Zeek script ghi TTL (không mặc định có), hoặc chấp nhận để giá trị mặc định/xấp xỉ, ghi rõ hạn chế này trong báo cáo |
| `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS` | Suy ra từ `history` (chuỗi ký tự mã hoá trạng thái TCP, vd "ShADadFf") | Cần viết hàm parse riêng — Zeek không xuất cờ TCP dạng số trực tiếp |
| `RETRANSMITTED_*` | Không có sẵn | Xấp xỉ = 0 nếu không tính được, ghi rõ hạn chế |
| `NUM_PKTS_*_BYTES` (histogram kích thước gói) | Không có sẵn trong `conn.log` chuẩn | Cần Zeek script riêng ghi lại phân bố kích thước gói, hoặc chấp nhận bỏ qua/xấp xỉ 0 |
| `DNS_*` | Cần join thêm với `dns.log` (theo `uid` chung) | Chỉ áp dụng cho flow là truy vấn DNS |
| `FTP_COMMAND_RET_CODE` | Cần join thêm với `ftp.log` | Chỉ áp dụng cho flow FTP |
| Các cột còn lại (`MIN_IP_PKT_LEN`, `TCP_WIN_MAX_*`, `ICMP_*`...) | Không có sẵn, cần log Zeek bổ sung hoặc xấp xỉ | Xem xét viết Zeek script (`.zeek`) tuỳ chỉnh nếu cần độ chính xác cao hơn |

**Khuyến nghị thực tế:** không cần ánh xạ hoàn hảo 100% — với các cột không lấy được trực tiếp từ Zeek mặc định, điền giá trị mặc định hợp lý (0 hoặc giá trị trung vị đã thấy lúc train) và **ghi rõ trong báo cáo** đây là giới hạn thực tế khi triển khai thời gian thực (khác với dữ liệu tĩnh đã qua xử lý kỹ của NF-v2). Ưu tiên làm đúng các cột quan trọng nhất trước: `IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS, PROTOCOL, FLOW_DURATION_MILLISECONDS, TCP_FLAGS`.

### 6.4. Sau khi có bảng đặc trưng thô, áp dụng ĐÚNG pipeline đã dùng lúc train

```python
import joblib
scaler = joblib.load(".../nf-cse-cic-ids2018-v2/scaler.joblib")
upper_bound = joblib.load(".../nf-cse-cic-ids2018-v2/upper_bound.joblib")
# clip roi scale -- dung ham apply_scale() co san trong src/etl/scale.py, KHONG viet lai
from etl.scale import apply_scale
df_scaled = apply_scale(df_raw, feature_cols, scaler, upper_bound)
```

## 7. Phase E — Suy luận offline + so sánh Suricata

- Suy luận: chép logic từ `src/models/evaluate_test.py` (hàm `evaluate_baseline`, `evaluate_graphsage`) — input là DataFrame/Data đã qua bước 6.4, không phải viết lại từ đầu.
- So sánh Suricata: đọc `eve.json` (Suricata xuất JSON có trường `alert` khi phát hiện) theo cùng khung thời gian với dữ liệu Zeek đã gán nhãn, tính TP/FP/FN cho cả 2 hệ thống trên cùng traffic.

## 8. Phase F — API real-time

**Chưa có code nào cho phần này** (`fastapi`/`uvicorn` mới chỉ khai báo trong `requirements.txt`, ghi chú "giai đoạn 5").

Thiết kế đề xuất:
1. Script nền đọc file `conn.log` kiểu `tail -f` (mỗi dòng mới xuất hiện = 1 flow mới kết thúc).
2. Với mỗi dòng mới: áp bước chuyển đổi (Phase D) → áp scaler (mục 6.4) → gọi model (đã nạp sẵn 1 lần lúc khởi động, không nạp lại mỗi lần) → lấy nhãn dự đoán + xác suất.
3. FastAPI expose 1 endpoint (vd `GET /predictions/recent`) hoặc dùng WebSocket để đẩy kết quả real-time ra giao diện hiển thị lúc demo — chọn WebSocket nếu muốn cập nhật tức thời không cần refresh.
4. Giao diện hiển thị: có thể chỉ cần in ra console/terminal lớn dễ nhìn lúc demo (đơn giản nhất), hoặc 1 trang HTML tối giản nếu muốn trực quan hơn.

## 9. Phase G — Cập nhật docs

Sau khi có kết quả thật, thêm 1 file mới `docs/graphsage/06_ket_qua_realtime.md` (không sửa các file 00-05 hiện có) — ghi: số liệu TP/FP/FN model vs Suricata, ảnh chụp màn hình demo (nếu có), hạn chế thực tế gặp phải ở bước chuyển đổi đặc trưng (mục 6.3).
