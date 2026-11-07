# Phase 01 — Dữ liệu, nhãn và đồ thị dùng chung

**Trạng thái:** mã nguồn hoàn thành và kiểm chứng trên UNSW (07/11/2026); CSE-CIC chờ chạy trên Colab — xem [báo cáo](../../results/graduation/phase_reports/phase_01.md). **Thời gian:** cuối tuần 1 đến đầu tuần 2.  
**Phụ thuộc:** phase 00 chốt nguồn dữ liệu; đọc [quy ước](quy_uoc_du_lieu.md).

## Mục tiêu và cách hiểu

Tạo dữ liệu để RF, XGBoost và GNN giải cùng một bài toán trên cùng flow test. “Không rò rỉ dữ liệu” bao gồm thống kê tiền xử lý, flow trùng và cửa sổ overlap; chỉ bỏ overlap ở ranh giới chưa đủ nếu scaler đã học từ vùng test.

## Lựa chọn dữ liệu

- Dùng NF-CSE-CIC-IDS2018-v2 làm bộ chính ban đầu; giữ nhãn đa lớp thật trong `Attack`. NF-UNSW-NB15-v2 là bộ thứ hai nếu ngân sách phase 00 cho phép.
- Với V2 không có thời điểm đáng tin cậy, gọi split là **block theo thứ tự nguồn**, không khẳng định time-based split. Ưu tiên metadata PCAP/session nếu truy xuất được; nếu không, công bố giới hạn này.
- V3 là nhánh lựa chọn chốt trong tuần 1, không thêm bắt buộc cả V2 lẫn V3. Dùng timestamp để chia/cắt cửa sổ; không tự đưa timestamp tuyệt đối vào feature nhằm tránh học lịch tấn công.
- Không tải thêm toàn bộ dataset hợp nhất chỉ để có nhiều lớp. Profile live có thể cần lab-train bổ sung, được quản lý riêng ở phase 05.

## Các bước triển khai

1. **Inventory và ID:** đọc CSV theo chunk nếu cần; kiểm tra tên/kiểu cột, NaN/Inf, IP/cổng, nhãn. Tạo flow ID trước khi lọc và lưu thống kê mọi dòng bị bỏ.
2. **Split chung:** tạo manifest ghi `flow_id → split` ở cấp source/session hoặc block đã chốt. Không dùng lại một split ngẫu nhiên riêng cho baseline.
3. **Kiểm tra trùng:** audit dòng/flow trùng xuyên split, session rơi vào nhiều tập và các cửa sổ giao nhau. Quyết định khử trùng có lưu vết, tránh làm biến mất mẫu lặp hợp lệ của DoS mà không giải thích.
4. **Nhãn:** hỗ trợ `binary` và `multiclass` bằng cấu hình. Mapping được khóa, lưu `benign_class_id`, phân bố và support từng lớp. Label/Attack/ID/session không được đi vào feature.
5. **Tiền xử lý:** fit clip/scaler từ flow train duy nhất; dùng lại cho val/test. Giữ raw test chưa clip để đánh giá chéo nếu cần sau này.
6. **Dựng đồ thị trong từng split:** IP:port là node, flow có hướng là edge. Cửa sổ không đi qua ranh giới; giữ danh sách flow ID theo thứ tự edge. Cấu hình số flow/overlap được lưu cùng artifact.
7. **Đặc trưng node:** giữ 4 đặc trưng cấu trúc + trung bình đặc trưng cạnh làm điểm xuất phát. Node dimension phải tự suy ra từ profile; ghi rõ DiGraph đơn dùng cho cấu trúc có thể gộp các cạnh song song trong khi tensor edge vẫn giữ flow riêng.
8. **Sharding:** ghi shard tăng dần, tránh dựng toàn bộ danh sách graph rồi giữ trong RAM. Baseline xuất bảng từ manifest, không lấy những flow khác với GNN test.
9. **Bộ kiểm tra:** tạo audit JSON/CSV về số flow, graph, class support, duplicate, giao tập, thứ tự feature và checksum cấu hình.

## Feature parity cho mô hình live

Lập bảng cho từng cột: tên, đơn vị, định nghĩa byte payload/IP, chiều originator/responder, nguồn NetFlow/Zeek, cách tính, missing và sai khác đã đo. Trùng tên cột không đủ chứng minh trùng ngữ nghĩa.

Profile `zeek_shared_v1` bắt đầu từ giao thức, số byte/gói hai chiều, thời lượng và các tốc độ suy ra **sau khi kiểm chứng**. Những đặc trưng thiếu như TTL/TCP window hoặc cờ TCP xấp xỉ không được coi là số đo thực. Nếu loại khỏi profile live, phải train lại đúng profile; không giữ model 39 chiều rồi âm thầm điền mean cho toàn bộ phần thiếu.

Kiểm tra offline/live trên cùng PCAP và các flow match được. Khác exporter có thể tách flow khác nhau, do đó báo cả tỷ lệ match và phần không đối chiếu được. Nếu không chứng minh được tương đương giữa NetFlow công khai và Zeek, dùng lab-train cùng Zeek cho mô hình triển khai, công bố tách biệt với benchmark NetFlow. NFStream chỉ là lựa chọn extractor cần đánh giá, không mặc định tương thích NetFlow V2.

## File cần xem hoặc sửa

| File hiện có | Thay đổi cần thiết |
|---|---|
| `src/etl/config.py`, `clean.py`, `load.py` | Schema, flow ID, nhãn và thống kê clean |
| `src/etl/split.py`, `run_etl.py`, `scale.py` | Split manifest chung, fit train-only, giữ raw test |
| `src/graph/build_graph.py`, `run_graph_builder.py` | Chọn target theo task, ID cạnh, dựng theo split, ghi shard tăng dần |
| `src/graph/config.py`, `windowing.py`, `node_features.py` | Profile động và metadata cửa sổ |
| `src/models/config.py` | Loại mọi label/metadata khỏi feature và đọc task/profile |
| `src/api/zeek_convert.py` | Đối chiếu converter với feature contract; chưa viết lại dashboard |

Module/schema mới được đặt theo cấu trúc repo khi triển khai; chưa có CLI mới nào được bảo đảm tồn tại.

## Đầu ra và nghiệm thu

- [ ] Có manifest dùng chung cho cả ba mô hình; giao flow ID giữa các split bằng 0 theo quy tắc đã chốt.
- [ ] Có chứng cứ scaler/clip chỉ fit flow train, không fit full dataset hoặc test.
- [ ] Có bảng class support; lớp vắng train/val/test được ghi rõ và xử lý theo thiết lập đánh giá.
- [ ] Graph labels khớp task; edge order khớp flow IDs; không mất nhãn khi shard/reload.
- [ ] Một graph dựng offline và từ converter cùng input đạt sai số đã định nghĩa, hoặc có báo cáo sai khác và hướng xử lý.
- [ ] Có audit dữ liệu và báo cáo phase; mô tả đúng giới hạn thời gian của V2.

Kiểm tra bắt buộc dùng fixture có overlap, nhãn nhiều lớp, lớp vắng và thứ tự feature bị đảo. Test phải phát hiện được một split giao nhau có chủ ý, không chỉ xác nhận số shape.

## Giao việc cho AI

> Thực hiện phase 01 trên cấu hình đã chốt ở phase 00. Tạo split manifest chung, target binary/multiclass và train-only preprocessing; lưu flow ID và raw test. Bổ sung kiểm tra ngăn leakage và sai edge-label alignment. Đừng chạy ETL full nếu chưa đo được mẫu nhỏ; chuẩn bị lệnh hoặc notebook Colab tái lập với artifact tách khỏi kết quả thực tập.

## Nguồn đọc

[P03 chuẩn đặc trưng](tai_lieu_tham_khao.md#p03), [P04 timestamp V3](tai_lieu_tham_khao.md#p04), [P12 đánh giá ML bảo mật](tai_lieu_tham_khao.md#p12), [T05 Zeek conn.log](tai_lieu_tham_khao.md#t05), [T11 NFStream](tai_lieu_tham_khao.md#t11).
