# Phase 05 — Lab và tối thiểu bốn kịch bản có nhãn

**Trạng thái:** chưa triển khai theo protocol đồ án. **Thời gian chính:** tuần 7; chuẩn bị và thu train/val từ tuần 1–3.  
**Phụ thuộc:** feature contract phase 01, mô hình được đóng băng trước lab-test, pipeline phase 04.

## Mục tiêu và cách hiểu

Chứng minh hệ thống hoạt động từ lưu lượng thật/replay tới cảnh báo và giải thích, với ground truth độc lập. Hai ảnh demo FTP/SSH trong PDF chứng minh đã có demo, nhưng không thay thế PCAP/log/nhãn cho đánh giá tốt nghiệp.

**Bốn kịch bản khác nhau không bắt buộc là bốn lớp khác nhau.** FTP và SSH brute-force có thể cùng nhãn BruteForce; phải ghi rõ taxonomy khi báo cáo.

## Topology và bộ kịch bản đề xuất

Giữ mạng ảo gồm máy tạo traffic/tấn công, máy dịch vụ và máy giám sát. Địa chỉ `.194/.199/.200` trong mã là cấu hình lab cũ, kiểm tra lại trước khi dùng. Thực hiện trong mạng lab được phép; Colab chỉ dùng train/đánh giá dữ liệu, không chạy thử tấn công mạng.

| ID | Kịch bản | Mục tiêu đánh giá | Nhãn/điều kiện |
|---|---|---|---|
| B0 | Traffic bình thường | Đo false positive và kiểm tra dịch vụ | Benign: đăng nhập hợp lệ, truyền tệp, truy cập web có ghi nhận |
| S1 | FTP brute-force | Nhiều lần xác thực không hợp lệ trên FTP | BruteForce hoặc nhãn FTP riêng theo mapping train |
| S2 | SSH brute-force | Nhiều lần xác thực không hợp lệ trên SSH | BruteForce hoặc nhãn SSH riêng theo mapping train |
| S3 | Quét cổng có kiểm soát | Đồ thị fan-out và nhiều kết nối ngắn | Reconnaissance/Scanning nếu model thực sự có lớp tương ứng |
| S4 | HTTP DoS hoặc replay DoS có nhãn | Lưu lượng tăng, backlog, mất capture và độ trễ | DoS theo định nghĩa và cấu hình đã ghi; không gọi một nguồn là DDoS |

S4 dùng workload có giới hạn trong máy ảo hoặc PCAP được phép; cần chứng cứ hành vi/kịch bản, không gán mọi HTTP request tải cao là DoS. Ghi giới hạn tốc độ, thời lượng và capture loss để kết quả không chỉ phản ánh máy giám sát quá tải.

Nếu S3 không nằm trong taxonomy của checkpoint CSE-CIC, **không đổi tên dự đoán của lớp khác thành Reconnaissance**. Hướng chính là bổ sung dữ liệu lab-train có Recon vào mô hình triển khai. Nếu chưa đủ dữ liệu, báo S3 là bài kiểm tra binary ngoài phân phối, đồng thời nêu rõ multiclass chỉ đánh giá trên các lớp được hỗ trợ; không tuyên bố đã nhận diện đúng loại scan.

## Dữ liệu cho mô hình triển khai

1. Thu nhiều phiên B0/S1–S4 khác nhau; dùng cùng Zeek/extractor cho train và live. Lưu cấu hình dịch vụ, thời điểm và phiên bản công cụ.
2. Tách theo **phiên thu** thành lab-train/lab-val/lab-test từ đầu; không chia ngẫu nhiên flow từ cùng một lần tấn công vào nhiều tập.
3. Taxonomy live khởi đầu có thể là `Benign, BruteForce, Reconnaissance, DoS`. Đây là taxonomy riêng, không đồng nhất ID với các dataset công khai.
4. Huấn luyện mô hình triển khai bằng lab-train, chọn theo lab-val. Có thể dùng pretrain công khai với profile đã kiểm chứng, nhưng báo rõ nguồn train/fine-tune; không gọi là zero-shot transfer.
5. Lưu lab-test, chạy cuối khi model/window/threshold đã khóa. Nếu dùng kết quả lab-test để sửa model, phải có phiên test mới độc lập.

Số phiên là rủi ro chính của model lab. Mỗi lớp cần xuất hiện ở train và có support test đủ phân tích; một phiên thu duy nhất cho cả train/test không chứng minh tổng quát hóa.

## Protocol thu và chạy một kịch bản

1. Lập manifest: scenario/run/capture ID, nguồn/đích, protocol/cổng, cấu hình, giới hạn tải, model và window/profile.
2. Đồng bộ thời gian các máy; ghi offset. Bắt PCAP và log Zeek, lưu log dịch vụ để đối chiếu xác thực/yêu cầu.
3. Chạy đoạn nền trước, kịch bản, nền sau; ghi mốc bắt đầu/kết thúc và sự kiện thật. Giữ traffic hợp lệ trên chính cổng mục tiêu để kiểm tra model có chỉ học cổng hay không.
4. Đối chiếu flow bằng UID/5-tuple và khoảng thời gian/capture. Nhãn phản ánh luồng thử tấn công, không nhất thiết đăng nhập thành công; định nghĩa phải ghi trước.
5. Flow không đủ bằng chứng để `unknown/unlabeled`. Không coi mọi flow có IP máy tấn công là Attack và mọi flow còn lại là Benign.
6. Lưu prediction gốc của hệ thống, so với ground truth ngoài model. Khử lặp theo flow và tách detection đúng/sai loại.
7. Lặp ít nhất 3 lần lab-test cho mỗi S1–S4 nếu đủ thời gian; thay thời điểm, tốc độ và nền. Báo số lần thực tế, không giả định các flow trong cùng phiên là các lần lặp độc lập.

## Replay: hai phép thử khác nhau

- **PCAP → Zeek offline → log replay theo timestamp:** phù hợp kiểm tra tái lập input, nhãn và logic window. Chạy Zeek đọc file nhanh không đo được độ trễ mạng thời gian thực.
- **Replay packet lên mạng lab với tốc độ ghi nhận:** phù hợp kiểm tra ingest/capture/throughput. Packet replay không mặc định tạo lại phiên TCP tương tác hay tác động dịch vụ như tấn công trực tiếp; phải nói đúng điều đang đo.

Nếu rewrite IP khi replay, cập nhật mapping ground truth và source metadata. Lặp cùng PCAP nhiều lần không tạo ra nhiều phiên kiểm thử độc lập.

## Chỉ số và đầu ra

| Nhóm | Báo cáo |
|---|---|
| Phân loại | Confusion matrix binary và multiclass; P/R/F1 theo lớp trên phần flow có nhãn |
| Cảnh báo | TP/FP/FN flow duy nhất; false positive trên nền; số cảnh báo trùng bị khử |
| Kịch bản | Có phát hiện ít nhất một cảnh báo đúng trong run; thời gian tới cảnh báo đầu tiên |
| Vận hành | p50/p95 ingest-to-alert, graph/inference time, throughput, RAM/CPU, drop/capture loss |
| Chất lượng nhãn | Số/tỷ lệ flow có nhãn và chưa gán nhãn; phương pháp và sai số matching |

Lưu dưới đường dẫn dự kiến `data/lab/<capture_id>/` và `results/graduation/lab/<run_id>/`: manifest, PCAP hoặc checksum/vị trí lưu, Zeek log, log dịch vụ, ground truth, predictions, metrics và ảnh dashboard/XAI liên quan.

Không cần commit PCAP lớn hoặc dữ liệu nhạy cảm vào Git. Cần có chỉ dẫn vị trí và hash để tái lập.

## File cần xem / tạo

Xem `scripts/victim_setup.sh`, `scripts/run_scenario_portscan.sh`, `scripts/background_traffic_generator.py` trước khi tái sử dụng. Script cũ không phải bằng chứng kịch bản đã chạy. Bổ sung công cụ tạo manifest, ghép nhãn, replay và đánh giá tại `scripts/` hoặc module riêng khi triển khai.

## Nghiệm thu

- [ ] Có ít nhất 4 hồ sơ kịch bản S1–S4 và nền B0, mỗi hồ sơ có nguồn nhãn, cấu hình, log và prediction.
- [ ] Những kịch bản chỉ replay được ghi rõ; không tuyên bố có tấn công live nếu chỉ đọc log offline.
- [ ] Lab-train/val/test khác phiên; model đã khóa trước test.
- [ ] Có nhãn đối chứng, coverage và thống kê flow duy nhất; không chỉ có số cảnh báo/ảnh màn hình.
- [ ] Không tự suy ra lớp từ cổng dịch vụ; lớp không hỗ trợ được báo đúng giới hạn.
- [ ] Có đánh giá đúng/sai loại, báo động nhầm và độ trễ, kể cả kịch bản phát hiện kém.

## Giao việc cho AI

> Chuẩn bị phase 05: kiểm tra script lab hiện có, tạo manifest/ground-truth/replay/evaluation và hướng dẫn cho từng S1–S4. Phân biệt thao tác cần chạy trên máy lab và xử lý dữ liệu trong repo/Colab. Chỉ ghi “đã thực hiện” khi có log/PCAP/prediction; tách phiên train/val/test và báo rõ lớp ngoài taxonomy. Không tự thực hiện hoạt động mạng ngoài phạm vi lab được cho phép.

## Nguồn đọc

[T05 Zeek](tai_lieu_tham_khao.md#t05), [T07 tcpreplay](tai_lieu_tham_khao.md#t07), [P12 phương pháp đánh giá](tai_lieu_tham_khao.md#p12), [P13 domain shift](tai_lieu_tham_khao.md#p13). Bộ dataset/nhãn công bố được kiểm tra qua [T01](tai_lieu_tham_khao.md#t01); không suy ra scan được hỗ trợ chỉ từ tên dataset.
