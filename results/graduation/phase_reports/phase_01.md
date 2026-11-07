# Báo cáo Phase 01 — Dữ liệu, nhãn và đồ thị dùng chung

Ngày: **07/11/2026**. Trạng thái: **hoàn thành phần mã nguồn và kiểm chứng trên UNSW-NB15-v2. CSE-CIC chưa chạy (cần Colab; notebook đã chuẩn bị).** Baseline RF/XGBoost đã đọc split chung.

## 1. Đã làm

| Việc | File |
|---|---|
| Protocol dùng chung: `flow_id`, block split + purge gap, mapping nhãn khóa theo task, fit clip/scaler chỉ trên train (tính từng cột), audit chồng lấp, tập "flow mới" | `src/etl/protocol.py` |
| ETL mới, ghi `data/processed_v2/<dataset>/` (không ghi đè artifact thực tập); đọc CSV theo khối vào mảng float32 cấp phát trước | `src/etl/run_etl_v2.py` |
| `build_graph` nhận `target_col`, `flow_id_col` (tương thích ngược với `infer.py`) | `src/graph/build_graph.py` |
| Dựng đồ thị trong từng split, ghi shard tăng dần (256 đồ thị/shard) | `src/graph/run_graph_builder_v2.py` |
| RF/XGBoost đọc cùng split; metric theo task, lớp vắng không làm hỏng; báo cáo thêm trên tập "flow mới" | `src/models/train_baseline_v2.py` |
| 18 test (10 cho protocol, 1 cho scaler tương đương sklearn, 1 cho bộ đọc CSV, 1 cho audit chồng lấp, 2 cho baseline, 3 test graph cũ) | `src/etl/tests/`, `src/models/tests/`, `src/graph/tests/` |
| Notebook Colab chạy ETL + graph + kiểm chứng + baseline cho CSE-CIC | `notebooks/02_phase01_cse_cic_etl_graph.ipynb` |

Lệnh đã chạy:

```powershell
$env:PYTHONPATH = "src"
python -m pytest src -q -p no:cacheprovider                 # 18 passed
python src/etl/run_etl_v2.py --dataset nf-unsw-nb15-v2
python src/graph/run_graph_builder_v2.py --dataset nf-unsw-nb15-v2 --task binary     --workers 4
python src/graph/run_graph_builder_v2.py --dataset nf-unsw-nb15-v2 --task multiclass --workers 4
python src/models/train_baseline_v2.py --dataset nf-unsw-nb15-v2 --task binary     --max-train-rows 300000
python src/models/train_baseline_v2.py --dataset nf-unsw-nb15-v2 --task multiclass --max-train-rows 300000
```

## 2. Kết quả đo (UNSW-NB15-v2, chạy thật)

**Split và đồ thị**
- Raw 2.390.275 flow; trùng 0, thiếu 0. 1.000 flow ở hai ranh giới mang nhãn `purged` (purge gap = 500).
- Flow mỗi split: train 1.673.192 / val 358.541 / test 357.542.
- **Cả 10 lớp đều có mặt ở train, val và test** (ít nhất: Worms 71 / 42 / 51).
- Đồ thị: 6.691 / 1.433 / 1.429 (train/val/test), cho cả binary và multiclass. Nạp lại toàn bộ shard: nhãn `y` khớp nhãn gốc tra theo `flow_id` ở mọi cạnh; **giao `flow_id` giữa các split = 0**.
- Khoảng 100–200 flow cuối mỗi split không vào đồ thị vì không đủ một cửa sổ đầy.

**Phát hiện quan trọng về dữ liệu: flow lặp lại rất nhiều**
- Train có 1.673.192 flow nhưng chỉ **183.190 vector đặc trưng khác nhau** (11%). Không có vector nào mang hai nhãn binary khác nhau.
- 92,4% flow val và 92,9% flow test có vector đặc trưng **giống hệt** một flow train. Benign: ~95%; Attack: ~62–64%.
- Cặp (IP nguồn, IP đích) của val/test gần như 100% đã xuất hiện ở train.
- Tập "flow mới" (vector không có trong train): 27.147 flow val và 25.396 flow test.

**Baseline trên split chung** (train subsample 300.000 flow cho chạy thử; đánh giá trên val 358.541 flow):

| Task | Mô hình | F1-macro (toàn val) | F1-macro (chỉ flow mới) |
|---|---|---:|---:|
| binary | Random Forest | 0,9811 | 0,9415 |
| binary | XGBoost | 0,9821 | 0,9452 |
| multiclass (10 lớp) | Random Forest | 0,6106 | 0,4600 |
| multiclass (10 lớp) | XGBoost | 0,6078 | 0,4533 |

Cách đọc: F1 cao ở TN1 phần lớn đến từ việc nhận ra flow đã thấy. Tập "flow mới" cho con số gần với khả năng tổng quát hóa thật. **Từ Phase 02, mọi bảng kết quả cần báo cả hai.** Đây là số chạy thử trên mẫu train 300 nghìn flow, không phải kết quả cuối; số cuối cần train toàn bộ train trên Colab.

## 3. Quyết định thiết kế

- Split theo **khối liên tục theo thứ tự nguồn**, không gọi là time-based (V2 không có timestamp tin cậy).
- Cửa sổ dựng riêng trong từng split nên không flow nào nằm ở hai tập, kể cả khi cửa sổ chồng lấp 50%.
- Mapping nhãn lấy từ tên nhãn, không từ thống kê split. UNSW multiclass `benign_class_id = 2`, binary `= 0`; lưu trong `contract.json`.
- Scaler/clip fit trên flow train của chính split chung (`scaler_fit_rows`); `raw_test.parquet` giữ bản chưa clip.
- Dòng purged vẫn nằm trong `flows.parquet` (cột `split = "purged"`) để truy vết và tiết kiệm RAM; mọi bước đọc đều lọc theo split train/val/test.
- Việc loại trùng chỉ xét các dòng giống hệt toàn bộ cột trừ `flow_id`, bằng hash 64-bit từng dòng. Đặc trưng lưu float32 nên số nguyên > 2^24 bị làm tròn, không ảnh hưởng sau khi clip và chuẩn hóa.

## 4. RAM cho CSE-CIC (18,9 triệu dòng)

Ban đầu ETL tốn khoảng 1,3 GB mỗi triệu dòng (đo 4 triệu dòng: 5,27 GB), ngoại suy ~25 GB. Sau khi đọc theo khối vào mảng float32, hash dòng thay cho `duplicated()`, scaler tính từng cột và bỏ các bản sao: **4 triệu dòng đỉnh 2,35 GB** (giảm 56%, đo thật, kết quả UNSW không đổi). Ngoại suy tuyến tính cho toàn CSE-CIC khoảng 9–10 GB, vừa trong Colab miễn phí 12,7 GB nhưng sát. **Đây là ước tính, chưa đo trên toàn bộ dữ liệu**; nếu thiếu RAM, dùng runtime High-RAM.

## 5. Còn thiếu / giới hạn

- [ ] Chạy ETL + graph **CSE-CIC** trên Colab (notebook `02_phase01_cse_cic_etl_graph.ipynb`) và đọc `classes_missing_by_split`: tấn công CSE-CIC tập trung theo ngày nên chia khối có thể làm một số lớp vắng ở một split.
- [ ] Baseline đầy đủ (không subsample) chưa chạy, nên là việc của Colab.
- [ ] Không có session id trong V2: chỉ đo chồng lấp qua `flow_id`, vector đặc trưng và cặp IP.
- [ ] Feature parity Zeek ↔ NetFlow chưa làm (thuộc nhánh live, Phase 04–05).
- [ ] `train_gnn.py` chưa đọc shard và `meta.json` mới (Phase 02).
