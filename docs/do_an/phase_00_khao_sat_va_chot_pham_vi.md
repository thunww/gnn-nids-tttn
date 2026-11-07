# Phase 00 — Khảo sát mã nguồn và chốt phạm vi

**Trạng thái 24/10/2026:** đã kiểm tra local; GPU Colab chưa thực chạy; kiểm tra máy lab bỏ qua theo yêu cầu người dùng. [Báo cáo và bằng chứng](../../results/graduation/phase_reports/phase_00.md). **Thời gian dự kiến:** khoảng 2 ngày đầu tuần 1.  
**Đọc trước:** [kế hoạch tổng](README.md), [quy ước dữ liệu](quy_uoc_du_lieu.md).

## Mục tiêu và cách hiểu

Chốt một bài toán có thể làm xong trước khi chạy train dài. Multiclass không chỉ đổi số neuron output: dữ liệu, nhãn, metric, model loader và dashboard đều cần hiểu cùng một định nghĩa lớp.

## Điểm xuất phát đã thấy trong repo

| Vị trí | Hiện trạng / việc phải kiểm tra |
|---|---|
| `src/etl/clean.py` | Có mã hóa `Attack_encoded`, nhưng cần khóa và lưu mapping đúng cho từng task |
| `src/models/config.py`, `src/graph/build_graph.py` | Nhãn mục tiêu đang cố định binary |
| `src/models/graphsage.py` | Đã nhận `num_classes`; có thể tái sử dụng kiến trúc |
| `src/models/train_gnn.py` | Có class weights, shards và checkpoint; cần audit seed/resume và cách tính validation |
| `src/etl/run_etl.py`, `src/graph/run_graph_builder.py` | Baseline chia dòng ngẫu nhiên; GNN chia block đồ thị. Chưa phải cùng một split để so sánh trực tiếp |
| Scaler trong ETL | Fit từ train của baseline rồi áp toàn bảng; cần đo giao với graph-val/test và sửa theo split chung |
| `src/models/evaluate_cross_dataset.py` | Đảo scale của dữ liệu đã clip; giá trị vượt ngưỡng cũ không thể khôi phục đầy đủ |
| `src/models/metrics.py`, `src/api/infer.py` | Metric AUC/probability và inference đang có giả định 2 lớp |
| `src/api/realtime_server.py` | Có worker, WebSocket; đọc log chưa xử lý rotation, bộ đệm phải đủ số flow, API chưa có graph/XAI |
| Dữ liệu/mô hình local | Đã xác nhận hai CSV, Parquet, graph shards và checkpoint trong thư mục bị Git ignore; checkpoint GNN nạp được; local dùng CPU |

Đây là nhận xét tĩnh từ mã nguồn, chưa phải kết quả đo mức rò rỉ hay hiệu năng mới.

## Công việc

1. Lập inventory raw CSV, version, schema, mapping, checkpoint và dung lượng trên Drive/local. Kiểm tra các artifact thực tập còn đọc được; không xóa hoặc ghi đè.
2. Lập bảng phân bố lớp từ CSV thật, trước/sau clean và từng split. Chọn CSE-CIC làm bộ chính theo nền sẵn có; kiểm tra UNSW làm bộ kiểm chứng thứ hai sau khi đo ngân sách Colab.
3. Kiểm tra timestamp: V2 theo số dòng không chứng minh trật tự thời gian. V3 có flow start/end và IAT, nhưng chỉ chuyển nếu tải/chuyển đổi khả thi trong tuần 1. Khi giữ V2, ghi rõ đây là cửa sổ số lượng trên thứ tự nguồn.
4. Chọn nhãn offline: ưu tiên taxonomy gốc cho mỗi dataset. Chọn taxonomy lab riêng có ngữ nghĩa rõ, nếu cần gom nhãn phải lập mapping trước train.
5. Chốt hai profile `netflow_full_v2` và `zeek_shared_v1`; profile sau chỉ bao gồm đặc trưng đã kiểm tra được ý nghĩa/đơn vị giữa nguồn train và Zeek.
6. Thử nhỏ trên Colab: dựng vài chục đồ thị, train 1–2 epoch, reload và predict. Đo peak RAM/VRAM, thời gian/epoch và kích thước shard; chưa suy rộng thời gian toàn bộ từ một batch.
7. Spike XAI: chạy explainer trên một đồ thị nhỏ và một output edge; xác nhận mask/gradient thực sự tác động đến score. Spike khoảng nửa ngày, lưu vấn đề tương thích để phase 03 xử lý.
8. Spike lab: thu một PCAP nhỏ, sinh conn.log, xác nhận nguồn/đích và số flow có thể đọc. Kiểm tra nền FTP/SSH; lên lịch thu các phiên lab-train, lab-val và lab-test khác nhau.

## Đầu ra dự kiến

- `results/graduation/phase_reports/phase_00.md`: inventory, các nguy cơ đã kiểm tra và cấu hình chốt.
- Cấu hình task/dataset/profile và danh sách nguồn dữ liệu; kế hoạch ngân sách train theo số lần chạy.
- Quyết định bằng văn bản về V2/V3, cách chia tập, taxonomy và 4 kịch bản.
- Notebook nhỏ/lệnh kiểm tra, số đo tài nguyên thực và ghi nhận spike XAI/live.

## Nghiệm thu

- [x] Có thể giải thích flow là cạnh, vì sao cần ngữ cảnh đỉnh và vì sao output phải khớp flow ID.
- [x] Có ít nhất một dataset thực truy cập được, label inventory và kế hoạch split khả thi.
- [x] Chốt phần bắt buộc và phần loại khỏi lịch; chưa hứa hẹn một mức F1 cụ thể.
- [ ] Có đường train → save → reload → predict trên mẫu nhỏ bằng Colab.
- [x] Đã thử rủi ro tích hợp XAI và parse/window log bằng fixture trên local; chưa đánh giá XAI trên checkpoint thật.
- [ ] Thu log thực trên máy lab: bỏ qua trong P0 theo yêu cầu người dùng.

Đã chạy train → save → reload → predict trên mẫu dữ liệu thật bằng **CPU local**. [Notebook và cách kiểm tra Colab](phase_00_huong_dan_colab.md) đã chuẩn bị; không đánh dấu Colab hoàn thành bằng kết quả CPU.

Nếu Colab chưa cấp GPU, vẫn thực hiện inventory, kiểm tra schema và smoke test nhỏ trên CPU, ghi rõ chưa có benchmark GPU.

## Giao việc cho AI

> Thực hiện phase 00: đọc các file mã nguồn trong bảng, lập inventory từ dữ liệu truy cập được và tạo báo cáo khảo sát. Chuẩn bị smoke test Colab/XAI/live, không train full và không sửa bài toán để bỏ multiclass. Chỉ ghi số đo đã thực chạy; nếu artifact nằm trên Drive chưa truy cập được, cung cấp cell/lệnh để người dùng chạy và ghi rõ phần còn chờ.

## Nguồn đọc

[P02 E-GraphSAGE](tai_lieu_tham_khao.md#p02), [P04 NetFlow V3](tai_lieu_tham_khao.md#p04), [P12 phương pháp đánh giá ML bảo mật](tai_lieu_tham_khao.md#p12), [T01 dataset](tai_lieu_tham_khao.md#t01), [T13 Colab](tai_lieu_tham_khao.md#t13).
