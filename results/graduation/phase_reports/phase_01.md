# Báo cáo Phase 01 — Dữ liệu, nhãn và đồ thị dùng chung

Ngày: **07/11/2026**. Trạng thái: **đã triển khai và chạy thật trên UNSW-NB15-v2; CSE-CIC chưa chạy; baseline RF/XGBoost chưa đọc từ manifest mới.**

## 1. Đã làm

| Việc | File |
|---|---|
| Protocol dùng chung: flow_id, block split + purge gap, mapping nhãn khóa theo task, fit clip/scaler chỉ trên train, hàm audit | `src/etl/protocol.py` |
| ETL mới (không ghi đè artifact thực tập, xuất ra `data/processed_v2/`) | `src/etl/run_etl_v2.py` |
| `build_graph` nhận `target_col`, `flow_id_col` (tương thích ngược với `infer.py` và pipeline cũ) | `src/graph/build_graph.py` |
| Dựng đồ thị trong từng split, ghi shard tăng dần (256 đồ thị/shard) | `src/graph/run_graph_builder_v2.py` |
| 10 test mới (có fixture overlap, lớp vắng, đảo thứ tự feature, scaler lẫn outlier test) | `src/etl/tests/test_protocol.py` |

Lệnh đã chạy:

```powershell
$env:PYTHONPATH = "src"
python -m pytest src/etl/tests src/graph/tests -q            # 13 passed
python src/etl/run_etl_v2.py --dataset nf-unsw-nb15-v2
python src/graph/run_graph_builder_v2.py --dataset nf-unsw-nb15-v2 --task multiclass --workers 4
```

## 2. Kết quả đo (UNSW-NB15-v2, chạy thật)

- Raw 2.390.275 flow; trùng 0, thiếu 0. Bỏ 1.000 flow ở 2 ranh giới (purge gap = 500 = 1 cửa sổ).
- Flow mỗi split: train 1.673.192 / val 358.541 / test 357.542.
- **Cả 10 lớp đều có mặt ở train, val, test** (ít nhất: Worms 71 / 42 / 51). Bảng đầy đủ: `data/processed_v2/nf-unsw-nb15-v2/class_support_multiclass.csv` (thư mục data bị Git ignore).
- Đồ thị multiclass: 6.691 / 1.433 / 1.429 (train/val/test). Nạp lại toàn bộ shard: nhãn `y` khớp nhãn gốc tra theo `flow_id` ở mọi cạnh; **giao flow_id giữa các split = 0**.
- Khoảng 100–200 flow cuối mỗi split không vào đồ thị vì không đủ một cửa sổ đầy (cách `sliding_windows` đã có).

## 3. Quyết định thiết kế

- Split theo **khối liên tục theo thứ tự nguồn**, không gọi là time-based (V2 không có timestamp tin cậy).
- Cửa sổ dựng riêng trong từng split nên không flow nào nằm ở hai tập, kể cả khi cửa sổ chồng lấp 50%.
- Mapping nhãn lấy từ tên nhãn, không từ thống kê split. UNSW multiclass `benign_class_id = 2`, binary `= 0`; mapping lưu trong `contract.json`.
- Scaler/clip fit trên **flow train của chính split chung** (`scaler_fit_rows` = số flow train); bản `raw_test.parquet` giữ nguyên chưa clip.

## 4. Còn thiếu / giới hạn

- [ ] Chạy ETL + graph cho **CSE-CIC** (3,2 GB CSV; cần ước lượng RAM, ~19 triệu dòng; nên chạy trên Colab hoặc máy đủ RAM). Với CSE-CIC cần xem `classes_missing_by_split` vì các tấn công tập trung theo ngày.
- [ ] Baseline RF/XGBoost phải đọc `flows.parquet` theo cột `split` (chưa viết).
- [ ] Chưa audit trùng/phiên xuyên split ngoài phép kiểm flow_id (UNSW không có dòng trùng).
- [ ] Feature parity Zeek ↔ NetFlow chưa làm (thuộc nhánh live).
- [ ] `train_gnn.py` chưa đọc shard mới / `meta.json` (Phase 02).
- Việc loại trùng chỉ xét các dòng giống hệt toàn bộ cột trừ `flow_id`.
