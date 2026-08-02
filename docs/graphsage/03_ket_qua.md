# Kết quả thực nghiệm — số liệu chính thức (dùng viết Chương 4)

## Chỉ số đánh giá

Tính đủ 6 chỉ số bằng `src/models/metrics.py` (`sklearn.metrics`):

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | % mẫu đoán đúng trên tổng số — chỉ tham khảo, dễ gây hiểu lầm khi dữ liệu mất cân bằng |
| Precision (macro) | Trong số cảnh báo đưa ra, bao nhiêu % đúng thật |
| Recall (macro) | Trong số tấn công thật, bắt được bao nhiêu % — **quan trọng nhất về bảo mật**, bỏ sót tấn công nguy hiểm hơn báo động nhầm |
| **F1-macro** | Trung bình hài hoà Precision & Recall — **chỉ số chính, dùng để chọn/so sánh model** |
| AUC-ROC | Khả năng phân biệt lớp tổng quát, không phụ thuộc 1 ngưỡng quyết định cụ thể |
| MCC | Tương quan dự đoán/nhãn thật, dùng cả 4 thành phần TP/TN/FP/FN — đáng tin cậy nhất khi mất cân bằng nghiêm trọng |

---

## Thí nghiệm 1 (TN1) — Đánh giá trong-cùng-bộ dữ liệu, trên tập TEST

**Trả lời RQ1.** Script: `src/models/evaluate_test.py` — nạp lại model đã train xong, chạy suy luận **đúng 1 lần** trên tập test (chưa từng dùng để train/chọn checkpoint).

**Lưu ý phương pháp luận (đã kiểm tra và sửa):** phát hiện + sửa 1 lỗi rò rỉ dữ liệu trong bước dựng đồ thị (cửa sổ trượt chồng lấp 50% từng bị chia ngẫu nhiên vào train/test, đo được 46,2% cặp cửa sổ liền kề bị ảnh hưởng) — đã sửa (chia theo khối liên tục theo thời gian), train lại toàn bộ, và **kiểm chứng thực tế xác nhận việc sửa lỗi không làm thay đổi đáng kể kết quả** (chênh lệch F1-macro trước/sau sửa: CSE-CIC 0,0001; UNSW-NB15 nhích lên 0,0013) — tăng độ tin cậy cho số liệu dưới đây. Chi tiết đầy đủ: `docs/decisions.md` mục 2026-07-31/2026-08-02.

### CSE-CIC-IDS2018-v2

| Model | Accuracy | Precision | Recall | **F1-macro** | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.9940 | 0.9879 | 0.9832 | 0.9856 | 0.9862 | 0.9711 |
| XGBoost | 0.9959 | **0.9975** | 0.9829 | **0.9901** | **0.9931** | **0.9804** |
| **E-GraphSAGE** | 0.9950 | 0.9969 | 0.9793 | 0.9879 | 0.9888 | 0.9760 |

### UNSW-NB15-v2

| Model | Accuracy | Precision | Recall | **F1-macro** | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.9977 | 0.9835 | 0.9867 | **0.9851** | 0.9995 | **0.9702** |
| XGBoost | 0.9975 | 0.9798 | 0.9878 | 0.9838 | **0.9998** | 0.9676 |
| **E-GraphSAGE** | 0.9941 | 0.9659 | **0.9928** | 0.9789 | 0.9991 | 0.9583 |

### Nhận xét chính

1. **Xếp hạng:** CSE-CIC → XGBoost > E-GraphSAGE > RF. UNSW-NB15 → RF > XGBoost > E-GraphSAGE. Chênh lệch F1-macro giữa E-GraphSAGE và baseline tốt nhất chỉ **0.005–0.02** — không đáng kể.
2. **Val và test gần như giống hệt nhau** (chênh lệch ≤ 0.001) — chứng minh việc chọn checkpoint theo tập val không bị overfit lên tập val, kết quả tổng quát hoá tốt sang dữ liệu hoàn toàn chưa từng thấy.
3. **E-GraphSAGE đánh đổi Precision lấy Recall ở UNSW-NB15:** đạt Recall cao nhất (0.9928) trong 3 model — bắt được nhiều tấn công thật hơn, đổi lại báo động nhầm nhiều hơn 1 chút (Precision 0.9659, thấp nhất). Đây là điểm mạnh đáng nêu, vì Recall cao quan trọng hơn trong bối cảnh an ninh mạng.

**Kết luận RQ1:** E-GraphSAGE đạt hiệu quả **tương đương, cạnh tranh được** với ML truyền thống — không vượt trội nhưng cũng không thua kém đáng kể.

---

## Thí nghiệm 2 (TN2) — Đánh giá chéo bộ dữ liệu (Cross-Dataset)

**Trả lời RQ2 — kiểm tra khả năng tổng quát hoá sang môi trường mạng khác.** Script: `src/models/evaluate_cross_dataset.py` — model đã train trên bộ NGUỒN, **không train/tinh chỉnh lại gì**, chạy suy luận trên tập test của bộ ĐÍCH. Áp dụng đúng scaler + ngưỡng clip outlier của bộ NGUỒN lên dữ liệu bộ ĐÍCH (mô phỏng đúng tình huống triển khai thực tế: đem model đã train đi dùng ở môi trường mới, không có sẵn thống kê của môi trường đó).

### Train CSE-CIC-IDS2018 → Test UNSW-NB15

| Model | Accuracy | Precision | Recall | **F1-macro** | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.9602 | 0.4801 | 0.5000 | 0.4899 | 0.2772 | 0.0000 |
| XGBoost | 0.9607 | 0.8501 | 0.5083 | 0.5065 | 0.3348 | 0.1080 |
| E-GraphSAGE | 0.7962 | 0.4582 | 0.4299 | 0.4436 | 0.0350 | -0.1082 |

### Train UNSW-NB15 → Test CSE-CIC-IDS2018

| Model | Accuracy | Precision | Recall | **F1-macro** | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.7940 | 0.4396 | 0.4538 | 0.4465 | 0.2239 | -0.1056 |
| XGBoost | 0.8605 | 0.4638 | 0.4925 | 0.4701 | 0.1689 | -0.0329 |
| E-GraphSAGE | 0.5079 | 0.4079 | 0.2943 | 0.3398 | 0.4549 | -0.2753 |

*(Số liệu GraphSAGE cập nhật 2026-08-02 sau khi sửa lỗi rò rỉ dữ liệu ở TN1 và train lại — baseline không đổi (không bị ảnh hưởng bởi lỗi). Kết luận không đổi, model "sạch" hơn còn thể hiện kém hơn 1 chút khi tổng quát hoá, càng củng cố kết luận RQ2 bên dưới.)*

### Nhận xét chính

**Cả 3 model đều sụp đổ nghiêm trọng khi đổi môi trường mạng** — MCC quanh 0 hoặc âm (3/4 trường hợp GraphSAGE âm), AUC-ROC nhiều trường hợp dưới 0.5 (có nơi chỉ 0.035) — tệ hơn cả đoán ngẫu nhiên theo nghĩa thống kê. E-GraphSAGE **không** thể hiện ưu thế tổng quát hoá tốt hơn baseline như giả thuyết ban đầu, thậm chí kém hơn ở cả 2 chiều.

**Nguyên nhân đã điều tra và xác nhận (không phải lỗi code):**
- Đo trực tiếp: **69,2% số mẫu (hàng)** dữ liệu UNSW-NB15 có **ít nhất 1** đặc trưng, sau khi quy đổi đúng cách sang thang đo CSE-CIC, rơi vào vùng giá trị cực đoan (|z-score| > 5) — ngoài phạm vi model từng học lúc train (bình thường tỷ lệ này chỉ ~0,0001%). Hai môi trường mạng khác biệt tới mức việc quy đổi đúng thang đo vẫn đẩy hầu hết dữ liệu ra ngoài phân phối train. *(Nếu tính theo từng giá trị đặc trưng đơn lẻ — % trên tổng số ô dữ liệu, không gộp theo hàng — tỷ lệ là **6,2%**; xem `report_figures/zscore_distribution.png`, minh hoạ theo cách đo này.)*
- Xác nhận độc lập bằng y văn: 1 nghiên cứu đo lường đúng vấn đề trôi dạt đặc trưng (feature drift) giữa các bộ dữ liệu NIDS khác — 36/45 đặc trưng chung vượt ngưỡng "trôi dạt nghiêm trọng" (PSI ≥ 0,25). 1 nghiên cứu khác ghi nhận thẳng hiện tượng mô hình "collapse xuống mức ngang đoán ngẫu nhiên" khi đổi bộ dữ liệu — đúng hiện tượng quan sát được ở đây.
- **Giải thích tại sao E-GraphSAGE không có ưu thế tổng quát hoá:** kiến trúc chỉ có 4/43 chiều đặc trưng node là cấu trúc thuần (bất biến quy mô mạng — bậc, PageRank, clustering); 39/43 chiều còn lại + `edge_attr` đưa thẳng vào message passing đều là đặc trưng thô **nhạy quy mô mạng** (byte, throughput...) — cùng loại thông tin gây hại cho baseline. Ưu thế lý thuyết "học cấu trúc, bất biến quy mô" của GNN bị pha loãng gần hết.

**Kết luận RQ2:** mô hình huấn luyện trên 1 môi trường mạng **không tổng quát hoá được** sang môi trường khác nếu không tinh chỉnh lại — hạn chế thật, có bằng chứng định lượng và xác nhận độc lập từ y văn. Giả thuyết "GNN tổng quát hoá tốt hơn nhờ học cấu trúc" **không được xác nhận** với kiến trúc E-GraphSAGE cụ thể đã triển khai.

**Hướng cải thiện tiềm năng (chưa triển khai, ghi nhận cho phần "Hướng phát triển"):** self-supervised learning (kiểu Anomal-E, arXiv:2207.06819) — học biểu diễn không dùng nhãn trước, tránh "ghi nhớ" ngưỡng đặc trưng cụ thể của 1 bộ dữ liệu. Lưu ý: chưa có nghiên cứu nào công bố số liệu self-supervised GNN cho đúng kiểu đánh giá cross-dataset trên đúng cặp dữ liệu này — đây là khoảng trống nghiên cứu thật, không phải hướng đã được chứng minh.
