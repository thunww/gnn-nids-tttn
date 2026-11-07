# Phase 04 — Pipeline liên tục và dashboard

**Trạng thái:** chưa triển khai. **Thời gian:** tuần 5–6; thu mẫu lab từ tuần 1.  
**Phụ thuộc:** schema phase 01, model bundle phase 02; XAI phase 03 có thể tích hợp sau luồng dự đoán cơ bản.

## Mục tiêu và cách hiểu

Hệ thống nhận log mới, gom flow, dự đoán, lưu cảnh báo và hiển thị đồ thị liên quan liên tục. Real-time ở đây là **near-real-time theo flow/cửa sổ**, không phải phân tích từng packet ngay lập tức.

```text
Zeek / replay → Reader → Hàng đợi hữu hạn → Window builder → Feature/Graph builder
→ Model bundle → Prediction store → WebSocket / Dashboard
                               → Yêu cầu giải thích → Worker XAI → Snapshot kết quả
```

## Công việc pipeline

1. Đưa đường dẫn log, model bundle, cấu hình window và IP lab vào CLI/config. Khi khởi động, kiểm tra schema, feature order, số lớp, scaler và model thay vì chạy đến lúc gặp lỗi tensor.
2. Reader xử lý file chưa có, dòng ghi dở, TSV header, dòng lỗi, truncate, rotation/đổi file và restart. Lưu cursor/offset cùng định danh file và chống replay trùng bằng flow ID; replay mode phải cho phép đọc từ đầu.
3. Tách ingest và inference bằng queue hữu hạn. Chốt chính sách khi queue đầy: backpressure hoặc drop có đếm; không tạo vô hạn future gửi WebSocket cho từng flow.
4. Window builder hỗ trợ số flow + timeout flush theo thời gian xử lý để không chờ vô hạn ở mạng ít traffic. Partial window phải được đánh giá và đưa vào cấu hình huấn luyện/validation tương thích trước khi dùng chính thức; cửa sổ quá nhỏ có thể chờ thêm và báo trạng thái.
5. Giữ định nghĩa event time/ingest time và late events rõ. Phiên bản đầu xử lý thứ tự nhận log, ghi đây là cửa sổ processing-time nếu không sắp xếp event time với watermark; không cần triển khai framework streaming lớn.
6. Nạp model một lần, `eval()`; inference dùng no-grad. Multiclass trả class name/proba và `is_attack` theo `benign_class_id`, không lọc cứng `pred_label==1`.
7. Lưu prediction theo flow và window; khử lặp alert theo chính sách chung. Lưu snapshot graph/feature trước khi bộ đệm trượt để XAI có đúng input gốc.
8. Worker báo lỗi/heartbeat; tiến trình shutdown đóng queue, file, WebSocket và worker. Có health endpoint báo model loaded, last ingest/inference, queue depth, drop/error counts.
9. Persist tối thiểu bằng SQLite hoặc JSONL có kiểm soát; lưu cursor và prediction đủ phục hồi sau restart. Không bắt buộc hạ tầng broker/database phân tán.

Zeek conn.log là bản tóm tắt kết nối, có thể chỉ đến sau khi kết nối kết thúc hoặc timeout. Thời gian forward thấp không đại diện cho thời gian phát hiện từ lúc tấn công bắt đầu. Xem [T05](tai_lieu_tham_khao.md#t05).

## Dashboard phải giúp người xem hiểu được gì?

| Thành phần | Nội dung cần hiển thị |
|---|---|
| Tổng quan | Flow duy nhất, lượt suy luận, cảnh báo duy nhất, trạng thái worker và độ trễ |
| Danh sách cảnh báo | Thời điểm, nguồn/đích, protocol, lớp dự đoán, score, model/profile |
| Chi tiết flow | Giá trị đặc trưng quan trọng và thông tin mapping lớp |
| Đồ thị liên quan | IP:port/nút, flow/cạnh có hướng, highlight target; lọc subgraph/top-k để trình duyệt không treo |
| Giải thích | Nút/cạnh/feature quan trọng, phương pháp, reference, thời gian và giới hạn |
| Kiểm soát kết nối | WebSocket reconnect; khi mở lại lấy snapshot gần nhất từ server |

Giao diện phải phân biệt dự đoán và nhãn thật khi đang ở chế độ đánh giá. Không đưa các chỉ số thí nghiệm vào thẳng màn hình giám sát nếu người dùng không cần để quyết định.

## Đo độ trễ và thông lượng

Thu mốc `flow_start`, `flow_end` khi có, `ingested`, `window_ready`, `graph_done`, `inference_done`, `alert_persisted`, `browser_received` khi đo được. Dùng đồng hồ monotonic để đo duration trong cùng máy; đồng bộ clock khi so mốc giữa máy.

- `model_latency`: chỉ forward model, ghi warm-up/device/batch.
- `processing_latency`: feature + graph + inference + lưu/phát cảnh báo sau khi cửa sổ sẵn sàng.
- `ingest_to_alert`: từ log được nhận đến cảnh báo, gồm thời gian chờ window và queue.
- `attack_to_first_alert`: đo riêng ở phase 05; bao gồm thời gian traffic/Zeek/log nếu đồng hồ và ground truth đủ.
- Báo p50/p95, flow/s đầu vào, unique flow/s xử lý, queue depth, CPU/RAM và tỷ lệ drop. XAI có bảng thời gian riêng.

Chốt tải mục tiêu sau pilot phần cứng thực, chưa đặt một con số ms cam kết. Dưới tải đó cần queue ổn định, số mất dòng giải thích được và không dừng sau rotation.

## File cần sửa / tạo

| Hiện có | Phần việc |
|---|---|
| `src/api/realtime_server.py` | Reader/worker lifecycle, config, health, WebSocket, UI, graph/XAI payload |
| `src/api/window_buffer.py` | Timeout/partial window, IDs, boundary và trạng thái chờ |
| `src/api/infer.py` | Model bundle loader, multiclass và profile compatibility |
| `src/api/zeek_convert.py` | Cùng feature contract đã kiểm chứng |
| `src/explain/` dự kiến | Job giải thích từ prediction snapshot |
| Module API/storage mới dự kiến | Tách logic để kiểm tra rotation, persistence và dedup |

Có thể giữ FastAPI/HTML hiện tại và bổ sung thư viện vẽ đồ thị phù hợp. Chưa cần đổi framework frontend chỉ để đạt yêu cầu dashboard.

## Nghiệm thu

- [ ] Replay cùng log qua offline và streaming cho dự đoán tương ứng theo window contract.
- [ ] Qua kiểm thử rotation, truncate, restart, dòng dở và dòng sai; không im lặng dừng worker.
- [ ] Không nhân đôi unique alert khi window overlap, restart hoặc reconnect trình duyệt.
- [ ] Có graph của cảnh báo và XAI đúng snapshot/flow đã chọn.
- [ ] Có log p50/p95 của các phần đo được; không gọi forward time là end-to-end latency.
- [ ] Chạy soak test ít nhất 60 phút trên lab hoặc replay, chủ động rotation/restart và lưu health/memory theo thời gian.
- [ ] Có cấu hình mẫu và lệnh chạy đã thực thử; giao diện báo lỗi/cạn dữ liệu rõ.

## Giao việc cho AI

> Thực hiện phase 04, tái sử dụng FastAPI và model bundle. Làm reader chống rotation/restart, queue hữu hạn, window timeout có kiểm chứng, multiclass, dedup và snapshot graph trước. Tích hợp dashboard và XAI on-demand. Viết kiểm tra các lỗi vận hành và lưu số đo thật; không mở rộng sang Kafka/cloud hay chạy XAI đồng bộ trên mọi flow.

## Nguồn đọc

[T05 Zeek](tai_lieu_tham_khao.md#t05), [T06 rotation](tai_lieu_tham_khao.md#t06), [T07 replay](tai_lieu_tham_khao.md#t07), [T14 WebSocket](tai_lieu_tham_khao.md#t14).
