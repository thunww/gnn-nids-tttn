# Giai đoạn 2 — Xây dựng biểu diễn đồ thị và quyết định về môi trường lab

**Thời gian dự kiến:** Tuần 2
**Trạng thái:** Đã sửa lỗi kiến trúc quan trọng (đồ thị bị dựng từ dữ liệu đã xáo trộn thứ tự) — cần chạy lại Graph Builder trước khi coi là hoàn tất. Nhánh lab dời sau Giai đoạn 3 (xem bên dưới).

## Mục tiêu
- Xây dựng module Graph Builder (bảng → đồ thị), dùng chung cho cả hai bộ dữ liệu, kèm unit test.
- Chốt quyết định có thiết lập môi trường lab VMware hay không (xem khung quyết định tại `docs/00_research_plan.md` mục 4.3).
- Nếu có lab: thiết lập mạng ảo, Zeek/Suricata, thực hiện các kịch bản tấn công.

## Quyết định môi trường lab
- **Phương án chọn:** **A — có lab** (2026-07-12)
- **Lý do:** Nhóm xác nhận có nhu cầu trình diễn tấn công real-time trước hội đồng. Theo khung quyết định ở `docs/00_research_plan.md` mục 4.3, đây là điều kiện bắt buộc phải thiết lập lab (2 bộ dữ liệu công khai chỉ ở dạng tĩnh, không demo trực tiếp được).
- **Hệ quả:** phạm vi thí nghiệm đầy đủ TN1, TN2, TN6; cần thêm: dựng VMware, cấu hình mạng host-only cô lập, cài Zeek + Suricata (ET Open Rules), thực hiện 5 kịch bản tấn công, thu thập + gán nhãn dữ liệu lab. Tuân thủ ranh giới bắt buộc tại mục 4.4 (mọi tấn công chỉ trong mạng ảo cục bộ, không được thực hiện trên hạ tầng cloud).
- **Thứ tự triển khai:** dời việc thiết lập VMware/lab lại **sau Giai đoạn 3** (huấn luyện xong, kiểm thử mô hình ổn định) — ưu tiên làm xong pipeline khoa học cốt lõi (Graph Builder → train GCN/GAT → đánh giá TN1/TN2) trước, tránh vừa lo hạ tầng lab vừa lo mô hình cùng lúc. Giai đoạn 2 hiện tại chỉ tập trung nhánh Graph Builder.

## Đầu ra kiểm chứng được
- [x] Graph Builder chạy đúng trên cả hai bộ dữ liệu, unit test pass.
- [ ] Chạy lại Graph Builder với dữ liệu giữ đúng thứ tự thời gian (xem 2026-07-19 bên dưới) — *_graphs.pt hiện tại đã lỗi thời.
- [ ] (Nếu Phương án A) Bộ dữ liệu lab thu thập đầy đủ nhãn, đã kiểm tra độ chính xác gán nhãn. (Dời sau Giai đoạn 3.)

## Nhật ký cập nhật
- 2026-07-12: Viết Graph Builder (`src/graph/`: `nodes.py`, `edges.py`, `node_features.py`, `windowing.py`, `build_graph.py`, `run_graph_builder.py`) — chuyển bảng flow đã xử lý thành đồ thị: IP:port → node, flow → cạnh có hướng mang đặc trưng, node có thêm đặc trưng cấu trúc (bậc vào/ra, PageRank, hệ số phân cụm qua `networkx`). Sliding window 10.000 dòng/cửa sổ, overlap 50%.
- 2026-07-12: 3 unit test (`src/graph/tests/test_graph_builder.py`) kiểm tra đánh ID node, dựng cạnh, tính đặc trưng node — pass.
- 2026-07-12: Chạy full trên toàn bộ dữ liệu thật (local, 8 nhân CPU song song — ban đầu chạy tuần tự ước tính mất 5-6 tiếng, sau khi dùng `multiprocessing` giảm còn ~30 phút). Kết quả (`*_graphs.pt`, tổng ~8.1GB):
  - `nf-cse-cic-ids2018-v2`: train 2644 đồ thị con, val 565, test 565.
  - `nf-unsw-nb15-v2`: train 333 đồ thị con, val 70, test 70.
  - Đã kiểm tra: mỗi đồ thị 10.000 cạnh, 39 đặc trưng/cạnh, đặc trưng node shape đúng (N, 4).

## Vấn đề phát sinh / quyết định
- **Hiệu năng:** `run_graph_builder.py` chạy tuần tự (1 nhân) sẽ mất 5-6 tiếng cho toàn bộ dữ liệu — chuyển sang `multiprocessing.Pool` (8 worker) đưa xuống ~30 phút. Cần giữ nguyên cách chạy trực tiếp file script (`python src/graph/run_graph_builder.py`) khi có multiprocessing trên Windows, không chạy qua `python -c "..."` (spawn method trên Windows yêu cầu entry point từ 1 file thật).
- **Dung lượng:** `*_graphs.pt` tổng ~8.1GB (riêng `nf-cse-cic-ids2018-v2/train_graphs.pt` = 5GB), hiện chỉ có ở local, **chưa upload lên Drive** — cần trước khi train GNN ở Giai đoạn 3 trên Colab.
- **2026-07-18 — sửa nhãn cạnh từ nhị phân sang đa lớp**, ảnh hưởng trực tiếp `build_graph.py` — cần chạy lại `run_graph_builder.py`. Chi tiết đầy đủ xem [`docs/decisions.md`](../decisions.md).
- **2026-07-19 — lỗi kiến trúc nghiêm trọng: đồ thị bị dựng từ dữ liệu đã xáo trộn thứ tự dòng** (do đọc trực tiếp từ `train/val/test.parquet` vốn đã bị `stratified_split` xáo ngẫu nhiên) — khiến mỗi "cửa sổ" 10.000 dòng không phải lát cắt thời gian thực, cấu trúc đồ thị gần như vô nghĩa. Đã sửa: `run_etl.py` xuất thêm `full_chronological.parquet` (giữ nguyên thứ tự), `run_graph_builder.py` đọc từ file này, chia đồ thị (không phải dòng) thành train/val/test sau khi đã dựng xong. Chi tiết đầy đủ + bằng chứng xem [`docs/decisions.md`](../decisions.md). **Cần chạy lại toàn bộ Graph Builder.**
