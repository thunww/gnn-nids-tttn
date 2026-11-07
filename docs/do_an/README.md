# Kế hoạch đồ án tốt nghiệp — GNN-NIDS

**Đề tài:** Xây dựng hệ thống phát hiện xâm nhập mạng dựa trên mạng nơ-ron đồ thị.  
**Điều kiện đã xác nhận:** thời gian khoảng 2 tháng, huấn luyện trên Google Colab.  
**Trạng thái:** kế hoạch triển khai, chưa phải báo cáo kết quả mới. Ngày lập: 22/10/2026.

**Cập nhật 24/10/2026:** [P0 đã khảo sát local và lưu bằng chứng](../../results/graduation/phase_reports/phase_00.md); Colab chưa thực chạy, kiểm tra lab P0 bỏ qua theo yêu cầu người dùng. Có thể bắt đầu phase 01 theo [cấu hình đã chốt](phase_00_quyet_dinh.json).

## 1. Có làm được trong hai tháng không?

**Có thể làm được nếu giữ phạm vi dưới đây và hoàn thành từng mốc nghiệm thu.** Đây là đánh giá kỹ thuật dựa trên mã nguồn và tài liệu nghiên cứu, chưa phải cam kết về điểm số hay thời gian huấn luyện. Số người thực hiện và tài nguyên lab hiện chưa được xác nhận.

Nền đã có gồm ETL, dựng đồ thị, E-GraphSAGE nhị phân, hai baseline, đánh giá và demo Zeek/FastAPI. Phần cần đầu tư nhất là dữ liệu đa lớp đúng phương pháp, giải thích cho bài toán phân loại **cạnh**, sự thống nhất đặc trưng lúc train/live và đánh giá lab có nhãn.

Phần đã hoàn thành trong báo cáo thực tập nằm tại [docs/README.md](../README.md). Tài liệu này mô tả **phần mở rộng của đồ án tốt nghiệp**, không thay thế lịch sử kết quả thực tập.

## 2. Phạm vi bắt buộc và phạm vi bổ sung

| Bắt buộc trong 8 tuần | Bổ sung khi phần bắt buộc đã đạt |
|---|---|
| Giữ chế độ binary và bổ sung multiclass cho E-GraphSAGE, RF, XGBoost | Thêm GCN/GAT/Transformer hoặc kiến trúc mới |
| Một bộ dữ liệu chính; chạy bộ thứ hai nếu đủ ngân sách đã đo | Đánh giá chéo bộ dữ liệu đa lớp với hệ nhãn đã hài hòa |
| Giải thích đồ thị bằng GNNExplainer hoặc phương pháp thay thế có kiểm chứng; giải thích đặc trưng cạnh bằng IG nếu khả thi | PGExplainer, self-supervised Anomal-E, domain adaptation |
| Pipeline liên tục, dashboard cảnh báo, đồ thị liên quan, đo hiệu năng | Kafka, Kubernetes, cloud, SIEM hoặc tự động chặn tấn công |
| Ít nhất 4 kịch bản: FTP brute-force, SSH brute-force và 2 kịch bản bổ sung có nhãn | So sánh Suricata, phát hiện zero-day, DDoS nhiều nguồn |
| Accuracy, Precision, Recall, F1, confusion matrix; đánh giá vận hành | McNemar hoặc thêm các kiểm định thống kê |

Không đặt điều kiện E-GraphSAGE phải thắng baseline. Một so sánh đúng, có phân tích lớp yếu và giới hạn, vẫn đáp ứng mục tiêu nghiên cứu ứng dụng.

## 3. Các quyết định thiết kế đề xuất

1. **Tiếp tục E-GraphSAGE hiện có.** Bài báo gốc đánh giá cả binary và multiclass; mã nguồn hiện tại là biến thể cần mô tả đúng, không mặc định tái lập y hệt bài báo. Xem [P02](tai_lieu_tham_khao.md#p02).
2. **Giữ NetFlow V2 để giảm rủi ro lịch.** Kiểm tra V3 trong phase 00 nếu cần timestamp thật. V3 bổ sung thông tin thời gian, nhưng đổi phiên bản sẽ phải dựng lại dữ liệu và mô hình; không phải yêu cầu bắt buộc. Xem [P04](tai_lieu_tham_khao.md#p04).
3. **Tách cấu hình nghiên cứu và cấu hình triển khai.** Nghiên cứu có thể dùng bộ đặc trưng đầy đủ; mô hình live phải được train với đúng tập đặc trưng thực sự tính được. Mọi kết quả phải ghi `feature_profile`, không lấy kết quả offline đầy đủ để đại diện cho mô hình live rút gọn.
4. **Một hệ nhãn rõ cho mỗi mô hình.** Không ghép class ID của hai dataset; không suy ra loại tấn công chỉ từ cổng 21/22.
5. **Đánh giá bằng flow duy nhất.** Cửa sổ chồng lấp có thể dự đoán lặp một flow; chỉ số chính dùng chính sách khử lặp đã chốt.
6. **Đồ thị theo các cửa sổ liên tiếp là đủ phạm vi.** Chưa cần mô hình temporal GNN có bộ nhớ xuyên cửa sổ.

## 4. Lịch 8 tuần và thứ tự đọc

| Phase | Tuần / ngân sách dự kiến | Đọc và giao AI thực hiện | Mốc đạt |
|---|---|---|---|
| 00 | Tuần 1, khoảng 2 ngày | [Khảo sát và chốt phạm vi](phase_00_khao_sat_va_chot_pham_vi.md) | Chốt dataset, nhãn, feature profile, tài nguyên; thử sớm XAI/live |
| 01 | Phần còn lại tuần 1 đến đầu tuần 2 | [Dữ liệu và đồ thị](phase_01_du_lieu_va_do_thi.md) | Cùng split, scaler chỉ fit train, lưu flow ID và raw dữ liệu test |
| 02 | Tuần 2–3 | [Binary, multiclass và Colab](phase_02_huan_luyen_da_lop_colab.md) | RF/XGB/GNN có checkpoint và báo cáo từng lớp |
| 03 | Tuần 4 | [Giải thích GNN](phase_03_giai_thich_gnn.md) | Giải thích đúng flow; có kiểm tra fidelity, kích thước và thời gian |
| 04 | Tuần 5–6 | [Real-time và dashboard](phase_04_realtime_dashboard.md) | Chạy liên tục, chịu rotation/restart, xem được cảnh báo và đồ thị |
| 05 | Tuần 7 | [Bốn kịch bản kiểm thử](phase_05_kiem_thu_4_kich_ban.md) | Có PCAP/log, nhãn đối chứng và bảng kết quả từng kịch bản |
| 06 | Tuần 8 | [Đánh giá và báo cáo](phase_06_danh_gia_va_bao_cao.md) | Số liệu đóng băng, báo cáo, hướng dẫn chạy và demo dự phòng |

Phải giữ những ngày cuối tuần 8 cho sửa lỗi và diễn tập; không dành toàn bộ hai tháng cho tinh chỉnh mô hình. Bắt đầu thu traffic lab nhỏ từ tuần 1, không chờ đến tuần 7 mới kiểm tra khả năng thu dữ liệu.

Nếu làm một mình, thực hiện tuần tự theo bảng. Nếu có thêm người, phần đọc lý thuyết, chuẩn bị dịch vụ lab và viết báo cáo có thể làm trong lúc chờ Colab; không thay đổi giao diện dữ liệu đã chốt.

## 5. Dấu hiệu cần giảm phạm vi

| Tình huống | Cách xử lý vẫn giữ yêu cầu cốt lõi |
|---|---|
| Colab thiếu RAM hoặc phiên bị ngắt | Shard dữ liệu, giảm batch, resume checkpoint; dùng subset theo session/cửa sổ và công bố rõ quy mô |
| Đến cuối tuần 3 chưa có multiclass chạy được | Dừng thử kiến trúc mới; tập trung một dataset chính và E-GraphSAGE + hai baseline |
| GNNExplainer không tương thích edge classifier | Sửa adapter/mask trong spike có giới hạn; chuyển phương pháp giải thích đồ thị có kiểm chứng, ghi rõ phương pháp thực tế |
| Chỉ IG hoạt động, chưa giải thích được cấu trúc | Chưa coi là hoàn thành toàn bộ mục tiêu XAI đã chọn; phải bổ sung kiểm chứng cấu trúc hoặc thống nhất thu hẹp với người hướng dẫn |
| Model offline không nhận đúng lab | Kiểm tra feature parity, capture loss, nhãn; có thể train/fine-tune bằng lab-train riêng, công bố rõ và giữ lab-test độc lập |
| Live quá ít flow để đầy cửa sổ | Dùng timeout/partial window đã kiểm tra offline hoặc replay đúng thời gian; không chỉ đo thời gian forward |
| Một loại tấn công không nằm trong nhãn train | Chọn kịch bản được hỗ trợ; hoặc dùng kiểm thử binary ngoài phân phối và báo đúng giới hạn, không gán lớp giả |

Tài nguyên Colab không được bảo đảm cố định; kế hoạch cần checkpoint và chạy tiếp. Colab dùng để xử lý dữ liệu, train và đánh giá; lab/replay mạng chạy trên máy được phép của bạn. Xem [T13](tai_lieu_tham_khao.md#t13).

## 6. Cách dùng bộ tài liệu với AI

Trước mỗi phase, cho AI đọc file này, [quy ước dữ liệu](quy_uoc_du_lieu.md), file phase và các nguồn được chỉ định. Các đường dẫn ghi **dự kiến** là đầu ra cần tạo, không phải chức năng đã có.

Mẫu giao việc:

```text
Hãy đọc docs/do_an/README.md, docs/do_an/quy_uoc_du_lieu.md và file phase được giao.
Đối chiếu với mã nguồn hiện tại trước khi sửa. Chỉ thực hiện phase này và các phụ thuộc
cần thiết; không tự thêm kiến trúc hoặc thay bài toán để lấy điểm cao hơn.
Giữ chế độ binary, thêm multiclass bằng cấu hình; tách artifact của từng task/profile/run.
Hoàn thành mã nguồn và những kiểm tra nghiệm thu phù hợp. Nếu thiếu dữ liệu/GPU/lab,
chuẩn bị đầy đủ phần chạy được và ghi rõ lệnh người dùng cần chạy, không bịa kết quả.
Cập nhật báo cáo phase với file đã sửa, lệnh đã chạy, artifact và vấn đề còn lại.
Giải thích ngắn các khái niệm mới để tôi hiểu được vì sao phải làm bước đó.
```

Mỗi phase có trạng thái ban đầu **chưa triển khai**. Chỉ chuyển sang hoàn thành khi có bằng chứng nghiệm thu; chạy thử trên dữ liệu giả chỉ chứng minh kỹ thuật, không chứng minh hiệu quả phát hiện.

## 7. Đối chiếu với yêu cầu đồ án

| Yêu cầu | Phase chịu trách nhiệm | Bằng chứng cuối |
|---|---|---|
| Lý thuyết an toàn mạng, IDS, GNN, biểu diễn đồ thị | 00, 01, 06 | Tổng quan có trích dẫn, lựa chọn thiết kế và sơ đồ hệ thống |
| Giải thích kết quả GNN | 03 | Giải thuật, case đúng/sai, đánh giá chất lượng giải thích |
| Luồng liên tục và sliding window | 01, 04 | Định nghĩa cửa sổ, kiểm thử và số đo pipeline |
| Dữ liệu, cân bằng, chỉ số | 01, 02, 06 | Phân bố lớp, split, baseline và báo cáo từng lớp |
| Mạng ảo, thu dữ liệu | 00, 05 | Topology, cấu hình, PCAP và log |
| Binary rồi multiclass | 02 | Hai task có cấu hình và artifact riêng |
| Dashboard cảnh báo và đồ thị | 04 | Luồng cảnh báo → chi tiết flow → đồ thị → giải thích |
| Tối thiểu 4 kịch bản | 05 | Bốn hồ sơ chạy có ground truth, không chỉ bốn ảnh |
| Đánh giá, báo cáo tốt nghiệp | 06 | Bảng số liệu truy vết được và hướng dẫn tái lập |

## 8. Tài liệu đi kèm

- [Quy ước dữ liệu và artifact dùng chung](quy_uoc_du_lieu.md): tránh mỗi AI triển khai một schema khác nhau.
- [Bài báo và tài liệu tham khảo](tai_lieu_tham_khao.md): 15 công trình nghiên cứu và 14 tài liệu chính thức, kèm mục đích sử dụng và mức đã kiểm tra.
- Các file phase ở bảng lịch: mô tả công việc, vị trí sửa mã nguồn, đầu ra và điều kiện hoàn thành.
