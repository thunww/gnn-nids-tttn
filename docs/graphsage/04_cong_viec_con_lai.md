# Công việc còn lại (chưa làm)

## GNNExplainer

Tích hợp module giải thích mô hình cho E-GraphSAGE (kiến trúc duy nhất đang dùng) — minh hoạ model "chú ý" vào đặc trưng/cạnh nào khi đưa ra quyết định phân loại. Chưa triển khai.

## Kiểm định thống kê (McNemar)

So sánh có ý nghĩa thống kê giữa E-GraphSAGE và baseline (Random Forest/XGBoost) — ngưỡng p-value < 0,05. Chưa triển khai.

## Thí nghiệm 6 — Mô phỏng real-time (VMware + Zeek + Suricata)

**Đã xác nhận sẽ làm, chưa bắt đầu.** Mục đích: trình diễn phát hiện tấn công trực quan theo thời gian thực trước hội đồng, đồng thời so sánh trực tiếp với hệ thống rule-based (Suricata).

### Checklist các bước

| # | Việc | Ai làm |
|---|---|---|
| 1 | Dựng VMware: máy tấn công + máy nạn nhân + máy giám sát, mạng **host-only, cô lập hoàn toàn khỏi Internet** (bắt buộc — kể cả kịch bản DoS cũng chỉ chạy trong mạng ảo cục bộ) | Người thực hiện |
| 2 | Cài Zeek trên máy giám sát, bật Promiscuous Mode để bắt toàn bộ traffic | Người thực hiện |
| 3 | Chạy lần lượt 5 kịch bản tấn công đã hoạch định, ghi lại chính xác thời điểm/loại tấn công (để gán nhãn đúng sau này) | Người thực hiện |
| 4 | Viết code chuyển log Zeek → đúng định dạng đặc trưng NetFlow V2 (43-49 cột, khớp schema đã dùng để train) — **bước kỹ thuật khó nhất**, log Zeek không giống định dạng NF-v2 (khác tên cột, khác cách tính 1 số chỉ số), sai bước này model sẽ dự đoán vô nghĩa | Cần viết code hỗ trợ |
| 5 | Viết script nạp dữ liệu đã chuyển đổi qua model đã train (E-GraphSAGE + baseline) để phân loại | Cần viết code hỗ trợ (tái sử dụng khung `evaluate_test.py`/`evaluate_cross_dataset.py`) |
| 6 | Cài Suricata + ET Open Rules trên cùng traffic | Người thực hiện |
| 7 | Viết script so sánh kết quả model vs Suricata (True Positive/False Positive/False Negative từng bên) trên traffic đã gán nhãn thủ công ở bước 3 | Cần viết code hỗ trợ |

**Rủi ro kỹ thuật cần lưu ý:** cấu hình VMware/Promiscuous Mode, độ chính xác gán nhãn thủ công.

---

*Ghi chú: nội dung đầy đủ, chi tiết hơn (bao gồm cả quá trình ra quyết định, các hướng đã thử và loại bỏ) nằm ở `docs/decisions.md` và `docs/phases/phase3_model_training.md` — chỉ nên tham khảo thêm nếu cần hiểu rõ lý do đằng sau 1 quyết định cụ thể, không dùng làm nguồn số liệu chính cho báo cáo.*
