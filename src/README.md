# Mã nguồn GNN-NIDS

Mục tiêu, luồng xử lý, kết quả và lệnh chạy được tổng hợp tại [docs/README.md](../docs/README.md), theo báo cáo PDF.

| Module | Vai trò trong luồng đã triển khai |
|---|---|
| `etl/run_etl.py` | CSV → bảng đã làm sạch/chuẩn hóa, scaler và ngưỡng clip |
| `graph/run_graph_builder.py` | Bảng giữ thứ tự gốc → đồ thị cửa sổ trượt → train/val/test |
| `models/shard_graphs.py` | Chia tập đồ thị train thành shard để giảm RAM |
| `models/train_baseline.py` | Huấn luyện Random Forest và XGBoost trên bảng |
| `models/train_gnn.py` | Huấn luyện E-GraphSAGE và lưu checkpoint tốt nhất |
| `models/evaluate_test.py` | TN1: đánh giá trên tập test cùng bộ dữ liệu |
| `models/evaluate_cross_dataset.py` | TN2: đánh giá chéo hai bộ dữ liệu |
| `models/metrics.py` | Tính sáu chỉ số đánh giá dùng chung |
| `api/zeek_convert.py` | Chuyển Zeek conn.log sang đặc trưng đầu vào |
| `api/window_buffer.py` | Gom flow thành cửa sổ chồng lấp |
| `api/infer.py` | Nạp mô hình và dự đoán nhãn/xác suất Attack |
| `api/realtime_server.py` | Đọc log mới, điều phối suy luận, phục vụ dashboard/WebSocket |
