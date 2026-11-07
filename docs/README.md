# GNN-NIDS — Mục tiêu và luồng xử lý

**Kế hoạch mở rộng cho đồ án tốt nghiệp trong 2 tháng:** xem [các phase, cách giao việc cho AI và tài liệu nghiên cứu](do_an/README.md). Phần bên dưới ghi lại phạm vi đã thực hiện trong báo cáo thực tập.

Tóm tắt phần đã thực hiện trong [báo cáo PDF](../NguyenHoangThanh_A05_BCCK.pdf), Chương 2–4, đối chiếu với mã nguồn trong `src/`. Số liệu dưới đây lấy từ báo cáo, không phải kết quả chạy lại.

## 1. Mục tiêu

Phát hiện xâm nhập mạng bằng cách phân loại từng luồng kết nối thành **Benign (0)** hoặc **Attack (1)** với **E-GraphSAGE**. Đánh giá mô hình so với **Random Forest** và **XGBoost**, kiểm tra khả năng tổng quát hóa giữa hai bộ dữ liệu, và minh họa phát hiện brute-force FTP/SSH trên mạng máy ảo.

## 2. Luồng dữ liệu và huấn luyện

```text
CSV NetFlow V2 → Làm sạch
  ├─ Chia bảng 70/15/15 → Fit clip + scaler trên train → Chuẩn hóa các tập
  │                    → Random Forest, XGBoost → Model đã lưu
  └─ Giữ thứ tự gốc → Áp clip + scaler đã fit → Cửa sổ trượt → Đồ thị
                     → Chia tập → Chia shard train → E-GraphSAGE → Checkpoint tốt nhất
Model đã lưu → TN1: kiểm thử cùng bộ / TN2: kiểm thử chéo bộ → Bảng chỉ số
```

1. **Dữ liệu:** NF-CSE-CIC-IDS2018-v2 (~18,9 triệu dòng) và NF-UNSW-NB15-v2 (~2,39 triệu dòng), cùng định dạng NetFlow V2. Dùng 39 đặc trưng số; IP/cổng chỉ dùng để định danh đỉnh; nhãn lấy từ cột `Label`.
2. **ETL — `src/etl/`:** loại dòng trùng và thiếu dữ liệu. Chia bảng cho baseline theo tỷ lệ 70/15/15, stratify theo loại tấn công gốc (`Attack`). Clip ngưỡng trên ở phân vị 99% rồi dùng `StandardScaler`; ngưỡng clip và scaler được fit từ tập train của nhánh bảng và lưu để dùng lại. Xuất thêm `full_chronological.parquet` giữ thứ tự gốc cho nhánh đồ thị.
3. **Đồ thị — `src/graph/`:** mỗi đỉnh là một cặp **IP:port**, mỗi cạnh có hướng là một flow. Cửa sổ gồm **2.000 flow** với CSE-CIC hoặc **500 flow** với UNSW, chồng lấp **50%**. Chia danh sách đồ thị thành các khối liên tục 70/15/15 và bỏ một cửa sổ tại mỗi ranh giới để tránh flow chồng lấp giữa các tập.
4. **Đặc trưng:** cạnh có **39 chiều**; đỉnh có **43 chiều**, gồm bậc vào, bậc ra, PageRank, hệ số phân cụm và trung bình 39 đặc trưng của các cạnh vào/ra.
5. **E-GraphSAGE — `src/models/`:** hai lớp truyền thông điệp kết hợp đặc trưng đỉnh lân cận và cạnh, tổng hợp bằng phép cộng. Ghép embedding đỉnh nguồn, đỉnh đích và đặc trưng cạnh, đưa qua MLP để phân loại flow. Hidden dimension là 128 cho CSE-CIC và 32 cho UNSW.
6. **Huấn luyện:** chia shard tập đồ thị train để giảm RAM; dùng Adam, Class-Balanced Cross-Entropy, giảm learning rate khi validation chững lại và early stopping. Lưu checkpoint tốt nhất theo **F1-macro trên validation**. RF/XGBoost học trực tiếp từ bảng 39 đặc trưng.

Các đầu ra nằm trong `data/processed/<dataset>/`: bảng Parquet, `scaler.joblib`, `upper_bound.joblib`, đồ thị `*_graphs.pt`, shard train và thư mục `models/` chứa các mô hình đã huấn luyện.

## 3. Đánh giá và kết quả

**TN1 — cùng bộ dữ liệu:** nạp mô hình đã lưu, suy luận trên tập test của từng bộ. Tính Accuracy, Precision-macro, Recall-macro, **F1-macro**, AUC-ROC và MCC bằng `src/models/metrics.py`.

**TN2 — chéo bộ dữ liệu:** huấn luyện trên bộ nguồn, kiểm thử trên bộ đích theo cả hai chiều, không huấn luyện hay tinh chỉnh lại. Dữ liệu đích được áp dụng scaler và ngưỡng clip của **bộ nguồn**.

F1-macro theo Bảng 4.1–4.4 của báo cáo:

| Thí nghiệm | Random Forest | XGBoost | E-GraphSAGE |
|---|---:|---:|---:|
| TN1: CSE-CIC-IDS2018-v2 | 0.9856 | 0.9901 | 0.9879 |
| TN1: UNSW-NB15-v2 | 0.9851 | 0.9838 | 0.9789 |
| TN2: CSE-CIC → UNSW | 0.4899 | 0.5065 | 0.4436 |
| TN2: UNSW → CSE-CIC | 0.4465 | 0.4701 | 0.3398 |

E-GraphSAGE đạt hiệu quả gần các baseline trong cùng bộ dữ liệu. Cả ba mô hình suy giảm mạnh khi đổi bộ dữ liệu; thí nghiệm này chưa cho thấy ưu thế tổng quát hóa của E-GraphSAGE.

## 4. Luồng demo thời gian thực — TN3

```text
Lưu lượng mạng máy ảo → Zeek conn.log → Đọc dòng mới, lọc traffic dashboard
→ Gom 2.000 flow, bước trượt 1.000 → Chuyển sang 39 đặc trưng NetFlow V2
→ Clip + Scale bằng tham số đã lưu → Dựng đồ thị → E-GraphSAGE
→ Nhãn và xác suất Attack → FastAPI / WebSocket → Dashboard
```

| Vai trò | Địa chỉ / chức năng |
|---|---|
| Máy tấn công | `192.168.207.194`, thực hiện brute-force bằng Hydra |
| Máy nạn nhân | `192.168.207.199`, dịch vụ FTP cổng 21 và SSH cổng 22 |
| Máy giám sát | `192.168.207.200`, chạy Zeek ở promiscuous mode và dashboard cổng 8000 |

- `realtime_server.py` đọc các dòng mới của `/opt/zeek/logs/current/conn.log`; `window_buffer.py` gom cửa sổ.
- `zeek_convert.py` chuyển log sang đặc trưng đầu vào. Một số đặc trưng được xấp xỉ hoặc điền giá trị mặc định do `conn.log` không cung cấp đầy đủ dữ liệu NetFlow V2.
- `infer.py` nạp một lần checkpoint CSE-CIC, scaler và ngưỡng clip, rồi dùng lại để dự đoán mỗi cửa sổ.
- Dashboard hiển thị tiến độ, flow gần nhất, nhãn dự đoán và cảnh báo Attack qua WebSocket `/ws`.

**Kết quả trong PDF (mục 4.4):** đã chạy brute-force FTP và SSH. Dashboard ghi nhận 25 lượt gán nhãn Attack cộng dồn: 16 ở cổng 21, 9 ở cổng 22, khớp IP nguồn/đích của kịch bản. Cửa sổ chồng lấp có thể đếm lại cùng một flow, nên số này không đại diện cho số flow duy nhất. Đánh giá demo bằng đối chiếu IP, cổng và thời điểm; chưa có nhãn thật cho toàn bộ lưu lượng để tính các chỉ số như TN1/TN2.

## 5. Thứ tự chạy mã nguồn

Chạy từ thư mục gốc repo, trong môi trường đã cài PyTorch và các thư viện của `requirements.txt`. Đặt CSV tại `data/raw/<dataset>/`, với `<dataset>` là `nf-cse-cic-ids2018-v2` hoặc `nf-unsw-nb15-v2`; tên file tương ứng là `NF-CSE-CIC-IDS2018-v2.csv` hoặc `NF-UNSW-NB15-v2.csv`.

```bash
python src/etl/run_etl.py
python src/graph/run_graph_builder.py
python src/models/shard_graphs.py
python src/models/train_baseline.py
python src/models/train_gnn.py
python src/models/evaluate_test.py
python src/models/evaluate_cross_dataset.py
```

Kết quả đánh giá ghi tại `data/processed/test_metrics.csv` và `data/processed/cross_dataset_metrics.csv`.

Trên máy giám sát đã cấu hình Zeek và có đủ ba file `scaler.joblib`, `upper_bound.joblib`, `models/graphsage_best.pt` trong `data/processed/nf-cse-cic-ids2018-v2/`, chạy từ gốc repo:

```bash
python -m uvicorn api.realtime_server:app --app-dir src --host 0.0.0.0 --port 8000
```

Kiểm tra các hằng `CONN_LOG_PATH`, `PROCESSED_DIR`, `DATASET` và IP trong `src/api/realtime_server.py` theo môi trường lab. Mở `http://192.168.207.200:8000` để xem dashboard; kết quả đầu tiên xuất hiện khi nhận đủ 2.000 flow mới hợp lệ.
