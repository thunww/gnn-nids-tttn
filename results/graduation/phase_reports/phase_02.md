# Báo cáo Phase 02 — Huấn luyện binary / multiclass

Ngày: **07/11/2026**. Trạng thái: **mã nguồn và kiểm thử hoàn thành; chưa train đầy đủ (cần Colab GPU); chưa có kết quả cuối cho đồ án.**

## 1. Đã làm

| Việc | File |
|---|---|
| Trình huấn luyện E-GraphSAGE theo protocol Phase 01: task `binary`/`multiclass`, số lớp và `benign_class_id` lấy từ mapping khóa, kích thước đầu vào lấy từ dữ liệu | `src/models/train_gnn_v2.py` |
| Seed cố định và reseed theo epoch, **resume thật** (model + optimizer + scheduler + best + epoch), `last.pt` tách khỏi `best_bundle.pt` | cùng file |
| Loss `ce` (mốc), `cb` (Class-Balanced, mặc định), `weighted`; trọng số tính trên **flow train duy nhất**, lớp vắng train vẫn hữu hạn | cùng file |
| Validation/test **theo flow duy nhất**: flow nằm trong 2 cửa sổ chồng lấp thì lấy trung bình xác suất | `predict_flows` |
| Model bundle (state, cấu hình mô hình, mapping, `feature_hash`, seed, loss) và `load_bundle` | cùng file |
| Báo cáo: metric theo task, per-class P/R/F1/support, confusion matrix raw + normalized, du doan theo `flow_id`, tách riêng tập "flow mới" | `evaluate_test`, `task_metrics` |
| 5 test mới: trọng số hữu hạn + dedup flow, dedup cửa sổ chồng lấp, **resume bằng chạy liên tục (khớp trọng số)**, bundle reload + mô hình học được, đường đánh giá test | `src/models/tests/test_train_gnn_v2.py` |
| Notebook Colab (pilot, train có resume lên Drive, eval test khóa sẵn) | `notebooks/03_phase02_train_gnn.ipynb` |

23 test đạt (`python -m pytest src`).

## 2. Đã chạy thật

Smoke trên UNSW-NB15-v2 multiclass (CPU, 2 shard train = 512 đồ thị, 2 epoch, val đầy đủ 358.541 flow): chạy hết trong 33 giây, val F1-macro 0,10 sau 2 epoch. Đây **chỉ chứng minh đường chạy đúng** (train → đánh giá theo flow → lưu bundle), không phải kết quả mô hình. Không chạm tập test thật.

## 3. Chưa làm / còn lại

- [ ] Train đầy đủ binary và multiclass (UNSW trước, CSE-CIC sau), 3 seed nếu đủ ngân sách; pilot 3 epoch để đo thời gian/VRAM trước.
- [ ] Baseline RF/XGBoost không subsample trên Colab, đánh giá test **một lần** cùng tập với GNN.
- [ ] So sánh CE / CB / weighted trên val, rồi khóa cấu hình trước khi mở test.
- [ ] Ablation nhỏ (chỉ edge features / bỏ nhóm feature).
- [ ] `src/api/infer.py` còn giả định 2 lớp: chuyển sang `load_bundle` khi làm Phase 04.
- [ ] `metrics.py`, `evaluate_test.py`, `train_gnn.py` của thực tập giữ nguyên (tái lập số liệu PDF); đường mới là `*_v2`.
- Lưu ý đọc kết quả: báo cả F1 toàn test và F1 trên tập "flow mới" (Phase 01 cho thấy ~92% flow UNSW test trùng vector với train).
