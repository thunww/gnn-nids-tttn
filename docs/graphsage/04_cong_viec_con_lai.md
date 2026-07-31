# Công việc còn lại (chưa làm)

## GNNExplainer

Tích hợp module giải thích mô hình cho E-GraphSAGE (kiến trúc duy nhất đang dùng) — minh hoạ model "chú ý" vào đặc trưng/cạnh nào khi đưa ra quyết định phân loại. Chưa triển khai.

## Kiểm định thống kê (McNemar)

So sánh có ý nghĩa thống kê giữa E-GraphSAGE và baseline (Random Forest/XGBoost) — ngưỡng p-value < 0,05. Chưa triển khai.

## Thí nghiệm 6 — Mô phỏng real-time (VMware + Zeek + Suricata)

**Đã xác nhận sẽ làm, chưa bắt đầu. Đã xác nhận demo kiểu TRỰC TIẾP** (model phản ứng gần thời gian thực ngay khi tấn công diễn ra trước hội đồng, không phải chạy trước rồi phân tích sau) — quyết định 2026-07-27, xem `docs/decisions.md`. Mục đích: trình diễn phát hiện tấn công trực quan theo thời gian thực, đồng thời so sánh trực tiếp với hệ thống rule-based (Suricata).

### Checklist các bước (7 phase, A→G)

| Phase | Việc | Ai làm |
|---|---|---|
| A | Dựng VMware: máy tấn công + máy nạn nhân + máy giám sát, mạng **host-only, cô lập hoàn toàn khỏi Internet** (bắt buộc — kể cả kịch bản DoS cũng chỉ chạy trong mạng ảo cục bộ), bật Promiscuous Mode cho máy giám sát, cài Zeek + Suricata + ET Open Rules | Người thực hiện |
| B | Chốt rõ nội dung cụ thể của 5 kịch bản tấn công (chưa thấy liệt kê chi tiết trong tài liệu hiện có), chuẩn bị cách ghi lại chính xác mốc thời gian từng kịch bản | Người thực hiện |
| C | Chạy Zeek + Suricata song song, thực hiện lần lượt 5 kịch bản, ghi lại chính xác thời điểm/loại tấn công (để gán nhãn đúng sau này) | Người thực hiện |
| D | Viết code chuyển log Zeek → đúng định dạng đặc trưng NetFlow V2 (43 cột, khớp schema đã dùng để train) — **bước kỹ thuật khó nhất**, log Zeek không giống định dạng NF-v2 (khác tên cột, khác cách tính 1 số chỉ số), sai bước này model sẽ dự đoán vô nghĩa | Cần viết code hỗ trợ |
| E | Viết script nạp dữ liệu đã chuyển đổi qua model đã train (E-GraphSAGE + baseline) để phân loại (offline trước, kiểm chứng đúng trước khi làm live) + script so sánh với Suricata (TP/FP/FN mỗi bên) | Cần viết code hỗ trợ (tái sử dụng khung `evaluate_test.py`/`evaluate_cross_dataset.py`) |
| F | **(Mới — do đã xác nhận demo trực tiếp)** Dựng API (`fastapi`/`uvicorn` đã có sẵn trong `requirements.txt`, ghi chú "giai đoạn 5" nhưng **chưa có code nào**) nhận traffic gần thời gian thực từ Zeek, chuyển đổi đặc trưng, gọi model, hiển thị kết quả ngay khi có tấn công — dùng để trình chiếu trực tiếp trước hội đồng | Cần viết code hỗ trợ |
| G | Tổng hợp kết quả vào `docs/graphsage/` | — |

**Rủi ro kỹ thuật cần lưu ý:** cấu hình VMware/Promiscuous Mode, độ chính xác gán nhãn thủ công, độ trễ xử lý real-time (Phase F).

---

*Ghi chú: nội dung đầy đủ, chi tiết hơn (bao gồm cả quá trình ra quyết định, các hướng đã thử và loại bỏ) nằm ở `docs/decisions.md` và `docs/phases/phase3_model_training.md` — chỉ nên tham khảo thêm nếu cần hiểu rõ lý do đằng sau 1 quyết định cụ thể, không dùng làm nguồn số liệu chính cho báo cáo.*
