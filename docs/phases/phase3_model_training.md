# Giai đoạn 3 — Xây dựng và huấn luyện mô hình

**Thời gian dự kiến:** Tuần 3
**Trạng thái:** Phát hiện + sửa lỗi kiến trúc nghiêm trọng ở Giai đoạn 2 (đồ thị bị dựng từ dữ liệu xáo trộn thứ tự — xem `docs/decisions.md` 2026-07-19) — 2 lượt train GCN/GAT trước đó (dưới đây) đều dựa trên đồ thị lỗi, **cần train lại từ đầu sau khi Graph Builder chạy lại**

## Mục tiêu
- Hiện thực và huấn luyện GCN, GAT (tối ưu siêu tham số bằng Optuna).
- Hiện thực hai baseline Random Forest, XGBoost.
- Theo dõi thực nghiệm bằng MLflow (tracking dir trỏ vào Google Drive).
- Checkpoint mô hình sau mỗi epoch.

## Đầu ra kiểm chứng được
- [x] Trọng số mô hình Random Forest + XGBoost cho cả 2 bộ dữ liệu.
- [x] Trọng số mô hình GCN + GAT cho cả 2 bộ dữ liệu (lượt đầu, chưa tối ưu siêu tham số).
- [ ] Nhật ký MLflow với số lượt chạy tối thiểu theo kế hoạch.
- [ ] Chương lý thuyết + kiến trúc mô hình trong báo cáo cập nhật với tham số thực tế.
- [ ] Cải tiến GNN (xem "Vấn đề phát sinh" bên dưới) trước khi dùng làm kết quả chính thức ở Giai đoạn 4.
- [ ] Thêm Precision (macro), Recall (macro), AUC-ROC (macro, one-vs-rest), MCC vào code đánh giá (`evaluate()`/`compute_confusion()` trong `train_gnn.py`) — hiện chỉ có Accuracy + F1-macro + Confusion matrix. Làm ở Giai đoạn 4, ngay trước khi chạy Thí nghiệm 1 chính thức trên tập test (xem giải thích mục "Nhật ký cập nhật" 2026-07-24).

## Nhật ký cập nhật
- 2026-07-18: Viết `src/models/` (`config.py`, `baselines.py`, `train_baseline.py`), huấn luyện Random Forest + XGBoost cho cả 2 bộ dữ liệu tại local (không cần GPU). Target: `Attack_encoded` (đa lớp). Kết quả trên tập val:

  | Bộ dữ liệu | Mô hình | Accuracy | F1-macro |
  |---|---|---|---|
  | nf-cse-cic-ids2018-v2 | Random Forest | 0.9765 | 0.7479 |
  | nf-cse-cic-ids2018-v2 | XGBoost | 0.9957 | 0.8115 |
  | nf-unsw-nb15-v2 | Random Forest | 0.9899 | 0.6694 |
  | nf-unsw-nb15-v2 | XGBoost | 0.9901 | 0.6483 |

  Model lưu tại `data/processed/<bộ>/models/{random_forest,xgboost}.joblib` (local, chưa upload Drive).

- 2026-07-18: Viết `src/models/gnn_config.py`, `gcn.py`, `gat.py`, `train_gnn.py` — kiến trúc GCN/GAT phân loại cạnh (concat embedding node u, v + đặc trưng cạnh, đưa qua lớp phân loại đa lớp). Train trên Colab (GPU T4), 20 epoch, checkpoint mỗi epoch.

  **⚠️ Kết quả bên dưới (lượt 1 và lượt 2) đã lỗi thời** — train trên đồ thị dựng từ dữ liệu bị xáo trộn thứ tự (lỗi phát hiện 2026-07-19, xem `docs/decisions.md`). Giữ lại để đối chiếu/rút kinh nghiệm, không dùng làm kết quả chính thức.

  Kết quả `val_f1_macro` cuối cùng (epoch 20):

  | Bộ dữ liệu | Random Forest | XGBoost | GCN | GAT |
  |---|---|---|---|---|
  | nf-cse-cic-ids2018-v2 | 0.7479 | 0.8115 | 0.7171 | 0.4087 |
  | nf-unsw-nb15-v2 | 0.6694 | 0.6483 | 0.2552 | 0.2631 |

  **Kết luận sơ bộ RQ1: baseline vẫn đang thắng GNN ở lượt chạy đầu này** — ngược kỳ vọng ban đầu. Nguyên nhân xác định được (xem mục dưới), cần cải tiến trước khi lấy làm kết quả chính thức cho Giai đoạn 4. Model + checkpoint lưu tại `data/processed/<bộ>/{models,checkpoints}/` trên Drive.

## Vấn đề phát sinh / quyết định
- Trên `nf-unsw-nb15-v2`, XGBoost có accuracy cao hơn Random Forest nhưng **F1-macro lại thấp hơn** (0.6483 so với 0.6694) — minh chứng thực tế cho nguyên tắc đã đặt ra ở Giai đoạn 1: accuracy không phản ánh đúng hiệu quả trên các lớp tấn công hiếm khi dữ liệu mất cân bằng, F1-macro mới là chỉ số quyết định khi so sánh mô hình.
- F1-macro của cả 2 mô hình trên `nf-unsw-nb15-v2` (0.65-0.67) thấp hơn rõ rệt so với `nf-cse-cic-ids2018-v2` (0.75-0.81) — do bộ UNSW-NB15-v2 nhỏ hơn nhiều (2.39 triệu so với 18.9 triệu dòng), các lớp tấn công hiếm có ít mẫu huấn luyện hơn. Cần lưu ý khi phân tích kết quả ở Giai đoạn 4.
- Đây là kết quả **sơ bộ trên tập val**, dùng để kiểm tra pipeline train đúng — đánh giá chính thức (so sánh với GCN/GAT, McNemar test...) thuộc về Giai đoạn 4, dùng tập test.
- **2026-07-18 — xác nhận bài toán đa lớp cho cả baseline lẫn GNN** (không phải nhị phân), để so sánh công bằng ở RQ1. Chi tiết đầy đủ xem [`docs/decisions.md`](../decisions.md).
- **2026-07-18 — GAT bị overfitting trên `nf-cse-cic-ids2018-v2`**: `val_f1_macro` đạt đỉnh 0.4329 ở epoch 9 rồi giảm dần còn 0.4087 ở epoch 20, trong khi `loss` tập train vẫn tiếp tục giảm đều — dấu hiệu học tủ kinh điển. Ngược kỳ vọng ban đầu (GAT dự kiến > GCN nhờ attention), thực tế GAT (0.41) kém hơn hẳn GCN (0.72) trên bộ này.
- **2026-07-18 — cả GCN lẫn GAT rất yếu trên `nf-unsw-nb15-v2`** (0.26 và 0.25, thua xa baseline 0.65-0.67). Nguyên nhân nhiều khả năng nhất: bộ này chỉ có 333 đồ thị con để train (so với 2644 của CSE-CIC — ít hơn 8 lần) — GNN vốn cần nhiều dữ liệu hơn cây quyết định để học tốt. Cả 2 mô hình vẫn đang tăng dần ở epoch 20, chưa bão hoà (khác GCN trên CSE-CIC đã gần chững).
- **Hướng cải tiến đề xuất cho lượt chạy sau:**
  1. Dùng checkpoint tốt nhất theo `val_f1_macro` thay vì luôn lấy epoch cuối (xử lý ngay được vấn đề overfitting của GAT, không cần train lại — checkpoint mỗi epoch đã có sẵn).
  2. Thêm early stopping (dừng khi `val_f1_macro` không cải thiện sau N epoch liên tiếp) + regularization (weight decay, tăng dropout) cho GAT.
  3. Với `nf-unsw-nb15-v2`: giảm `WINDOW_SIZE` hoặc tăng `WINDOW_OVERLAP` (`src/graph/config.py`) để tạo nhiều đồ thị con hơn từ cùng lượng dữ liệu, hoặc tăng số epoch (vì cả 2 mô hình chưa bão hoà).
  4. Tối ưu siêu tham số bằng Optuna (đã có trong kế hoạch ban đầu, `docs/00_research_plan.md`) thay vì để mặc định thủ công.

- **2026-07-18 — lượt 2: áp dụng cải tiến có trích dẫn tài liệu** (dropout riêng GCN=0.4/GAT=0.5, weight_decay=5e-4, class-weighted CrossEntropyLoss, early stopping patience=5, tự lưu checkpoint tốt nhất theo `val_f1_macro`, max 40 epoch). Kết quả `val_f1_macro` tốt nhất (không phải epoch cuối):

  | Bộ dữ liệu | Random Forest | XGBoost | GCN | GAT |
  |---|---|---|---|---|
  | nf-cse-cic-ids2018-v2 | 0.7479 | 0.8115 | 0.6714 (epoch 30) | 0.5921 (epoch 29) |
  | nf-unsw-nb15-v2 | 0.6694 | 0.6483 | 0.4339 (epoch 38) | 0.3452 (epoch 31) |

  So với lượt 1: **GAT tăng mạnh cả 2 bộ** (CSE-CIC 0.41→0.59, UNSW 0.26→0.35 — đúng mục tiêu chữa overfitting), **GCN/UNSW-NB15 tăng vọt** (0.26→0.43, nhờ class weight + train dài hơn), nhưng **GCN/CSE-CIC giảm nhẹ** (0.72→0.67 — nghi do class weight quá mạnh gây mất ổn định, `val_acc` giảm từ 0.99 xuống ~0.55 và dao động mạnh giữa các epoch thay vì tăng mượt như lượt 1). **GNN vẫn chưa vượt baseline ở cả 2 bộ**, dù khoảng cách đã thu hẹp đáng kể so với lượt 1.

- **2026-07-19 — lượt 3: train lại trên đồ thị đã sửa đúng** (dựng từ `full_chronological.parquet`, giữ thứ tự gốc — xem lỗi + cách sửa ở `docs/decisions.md` mục 2026-07-19). Cùng cấu hình tinh chỉnh như lượt 2 (chưa đổi gì thêm). Kết quả `val_f1_macro` tốt nhất:

  | Bộ dữ liệu | Random Forest | XGBoost | GCN | GAT |
  |---|---|---|---|---|
  | nf-cse-cic-ids2018-v2 | 0.7479 | 0.8115 | 0.6540 (epoch 9, early stop epoch 14) | 0.5594 (epoch 5, early stop epoch 10) |
  | nf-unsw-nb15-v2 | 0.6694 | 0.6483 | **0.4738 (epoch 40, KHÔNG early stop — vẫn đang tăng)** | 0.4189 (epoch 21, early stop epoch 26) |

  **2 bộ dữ liệu phản ứng ngược chiều nhau sau khi sửa lỗi:**
  - `nf-unsw-nb15-v2`: **tăng tiếp cả GCN lẫn GAT**, nhất quán qua cả 3 lượt (0.26→0.43→0.47 và 0.26→0.35→0.42). GCN chạy đủ 40/40 epoch không hề early-stop — vẫn còn dư địa cải thiện nếu tăng epoch tối đa. Đúng kỳ vọng: bộ này có tính "cụm theo IP" mạnh nhất (60% ở file gốc) nên sửa lỗi thứ tự ảnh hưởng rõ rệt nhất.
  - `nf-cse-cic-ids2018-v2`: **giảm tiếp** (0.72→0.67→0.65 GCN; 0.41→0.59→0.56 GAT), early stopping kích hoạt rất sớm (epoch 10-14 thay vì 30-35 ở lượt 2).

  **Lý do CSE-CIC giảm — khả năng cao là đánh giá giờ khắt khe/trung thực hơn, không hẳn mô hình học kém đi:** ở lượt 1-2, Graph Builder cắt cửa sổ *riêng biệt trong từng file* `train/val/test.parquet` (vốn đã là mẫu ngẫu nhiên đại diện toàn bộ dữ liệu do bị xáo trộn) → tập val cũ có phân bố gần giống train → đánh giá "dễ" hơn thực tế. Lượt 3: dựng hết đồ thị từ dữ liệu đúng thứ tự thời gian rồi mới chia ngẫu nhiên **các đồ thị** vào train/val/test → tập val giờ là lát cắt thời gian thực, có thể rơi vào giai đoạn khác hẳn đặc điểm với train → khó hơn nhưng trung thực hơn nhiều so với triển khai thực tế.

  **⚠️ Điểm chưa công bằng cần lưu ý cho Giai đoạn 4:** baseline (Random Forest/XGBoost) vẫn dùng `train/val/test.parquet` kiểu cũ (chia ngẫu nhiên theo dòng, không theo thời gian) — nghĩa là **baseline đang được đánh giá "dễ" hơn** so với GNN (đánh giá theo thời gian thực). Cần ghi rõ hạn chế phương pháp luận này trong báo cáo, hoặc cân nhắc đánh giá baseline theo cùng kiểu chia thời gian để so sánh khách quan hơn.

- **2026-07-19 — lượt 4: cải tiến sâu hơn, có nghiên cứu tài liệu.** Phát hiện thêm 1 thiếu sót kiến trúc: `edge_attr` (39 đặc trưng luồng mạng thật) trước đây chỉ được ghép vào **sau khi** tính xong message passing/attention — nghĩa là cơ chế attention của GAT chỉ "nhìn" được cấu trúc đồ thị (bậc node), không biết nội dung luồng mạng khi quyết định "chú ý" ai. Áp dụng các cải tiến:
  1. **`gat.py`: GATConv dùng `edge_dim=39`** — đưa đặc trưng cạnh trực tiếp vào công thức tính attention (đúng tinh thần E-GraphSAGE đã trích dẫn). GCN giữ nguyên đơn giản (đúng vai trò baseline GNN cơ bản nhất).
  2. **`train_gnn.py`: thay `class_weight="balanced"` (sklearn) bằng Class-Balanced Loss** (Cui et al., CVPR 2019 — công thức "số mẫu hiệu quả" `(1-β)/(1-β^n)`, β=0.999) — bớt cực đoan hơn nhiều với lớp siêu hiếm so với tỷ lệ nghịch trực tiếp (kiểm chứng cục bộ: trọng số lớp thường ~1.12 so với lớp đông nhất ~0.0011, thay vì chênh lệch hàng chục nghìn lần).
  3. Thêm `ReduceLROnPlateau` (tự giảm nửa learning rate nếu `val_f1_macro` không cải thiện sau 3 epoch) — GNN vốn nổi tiếng huấn luyện không ổn định, kỹ thuật này giúp hội tụ mượt hơn.
  4. `MAX_EPOCHS`: 40 → 80 (UNSW-NB15 chưa bão hoà ở epoch 40 lượt 3).
  - Đã test cục bộ: cả 2 kiến trúc chạy đúng, LR scheduler tự giảm đúng lúc khi chững lại. Cần chạy lại trên Colab để có kết quả thật.

  **Kết quả thật trên Colab (`val_f1_macro` tốt nhất):**

  | Bộ dữ liệu | Random Forest | XGBoost | GCN | GAT |
  |---|---|---|---|---|
  | nf-cse-cic-ids2018-v2 | 0.7479 | 0.8115 | **0.7262 (epoch 22)** — tốt nhất từ trước tới giờ | 0.5073 (epoch 9) — giảm nhẹ so với lượt 3 |
  | nf-unsw-nb15-v2 | 0.6694 | 0.6483 | **0.1167 (epoch 1)** — sụp đổ | **0.0978** — sụp đổ hoàn toàn |

- **2026-07-19 — lượt 5: sửa nguyên nhân sụp đổ** (`WINDOW_SIZE` 10.000→2.000, `EARLY_STOPPING_PATIENCE` 5→15, `LR_SCHEDULER_PATIENCE` 3→8 — chi tiết + căn cứ nghiên cứu tại [`docs/decisions.md`](../decisions.md) mục "GNN sụp đổ..."). Kết quả `val_f1_macro` tốt nhất:

  | Bộ dữ liệu | Random Forest | XGBoost | GCN | GAT |
  |---|---|---|---|---|
  | nf-cse-cic-ids2018-v2 | 0.7479 | 0.8115 | **0.7390 (epoch 59)** — cao nhất từ trước tới giờ, gần bằng RF | **0.6751 (epoch 53)** — tăng vọt so với lượt 4 (0.51) |
  | nf-unsw-nb15-v2 | 0.6694 | 0.6483 | 0.4190 (epoch 28) — hồi phục từ sụp đổ, nhưng thấp hơn đỉnh lượt 3 (0.4738) | 0.3796 (epoch 27) — hồi phục, thấp hơn đỉnh lượt 3 (0.4189) |

  **Đánh giá:** hết hiện tượng "sụp đổ" (all-collapse) — cả 4 mô hình học được trở lại, không còn đứng yên ở mức đoán bừa. CSE-CIC-IDS2018 đạt kết quả tốt nhất từ trước tới giờ cho cả 2 kiến trúc. Nhưng UNSW-NB15-v2 tuy đã hồi phục vẫn **chưa vượt qua đỉnh của lượt 3** — nghi ngờ `WINDOW_SIZE=2000` hơi nhỏ so với bộ này (ít host hơn CSE-CIC, cửa sổ nhỏ có thể không đủ dài để lộ rõ hành vi bất thường theo IP, đúng cảnh báo "cửa sổ quá nhỏ làm giảm ngữ cảnh hành vi" đã trích dẫn trước đó) — cần nghiên cứu thêm hướng cải thiện riêng cho bộ này.

- **2026-07-19 — nghiên cứu riêng cho UNSW-NB15-v2 + thêm công cụ chẩn đoán.** Tìm được: UNSW-NB15 có vấn đề **"chồng lấn lớp" (class overlap)** đã được ghi nhận trong nhiều nghiên cứu độc lập — không chỉ mất cân bằng số lượng mà đặc trưng của nhiều loại tấn công còn "lấn" vào nhau, đây là hạn chế cố hữu của chính bộ dữ liệu (cần ghi vào phần hạn chế báo cáo). Tuy nhiên tìm được tham chiếu 1 mô hình (SAGEConv-GNN + Transformer) đạt macro-F1 0.9749 trên UNSW-NB15 — chứng tỏ còn nhiều dư địa cải thiện, không phải trần thấp cố định. Đã áp dụng:
  1. **Thêm confusion matrix** (`src/models/train_gnn.py`: `compute_confusion`, `save_confusion_matrix`, `print_confusion_summary`) — tính trên tập val bằng model tốt nhất (không phải epoch cuối), lưu file CSV đầy đủ + in tóm tắt "lớp nào hay bị nhầm thành lớp nào" ra màn hình. Mục đích: xác minh đúng nguyên nhân (chồng lấn lớp hay lỗi khác) trước khi sửa tiếp.
  2. **Tách `WINDOW_SIZE` riêng theo từng bộ dữ liệu** (`src/graph/config.py`: `WINDOW_SIZE_BY_DATASET`) — giữ nguyên 2.000 cho CSE-CIC-IDS2018 (đang tốt, không đụng vào), thử 5.000 riêng cho UNSW-NB15-v2 (nghi ngờ cửa sổ 2.000 quá nhỏ, thiếu ngữ cảnh hành vi cho bộ ít host hơn).
  - Đã test cục bộ: confusion matrix tính đúng kích thước (sửa 1 lỗi nhỏ — sklearn tự bỏ lớp không xuất hiện trong mẫu nhỏ, phải ép `labels=range(num_classes)`), `WINDOW_SIZE` đọc đúng theo từng bộ. Cần chạy lại Graph Builder + train để có kết quả thật.

  **CSE-CIC-IDS2018 (GCN):** cải tiến rất rõ, `val_acc` ổn định ~0.99 suốt quá trình (không còn dao động 0.32-0.62 như lượt 2/3) — xác nhận Class-Balanced Loss + LR scheduler giải quyết đúng vấn đề bất ổn định. Gần bằng Random Forest.

  **UNSW-NB15-v2 (cả 2 mô hình): sụp đổ nghiêm trọng** — GCN tốt nhất ngay epoch 1 rồi càng train càng tệ; GAT đứng yên ở mức "đoán bừa toàn Benign" suốt cả quá trình, không học được gì. Nguyên nhân + cách sửa (giảm `WINDOW_SIZE`, nới patience) — xem chi tiết đầy đủ có trích dẫn nghiên cứu tại [`docs/decisions.md`](../decisions.md) mục 2026-07-19 ("GNN sụp đổ..."). **Cần chạy lại Graph Builder + train lại để kiểm chứng.**

- **2026-07-19 — lượt 6: `WINDOW_SIZE=5000` riêng cho UNSW-NB15 + confusion matrix thật đầu tiên.** Kết quả `val_f1_macro` tốt nhất:

  | Bộ dữ liệu | Random Forest | XGBoost | GCN | GAT |
  |---|---|---|---|---|
  | nf-cse-cic-ids2018-v2 | 0.7479 | 0.8115 | 0.7357 (epoch 16) | 0.5588 (epoch 3) — dao động run-to-run, thấp hơn lượt 5 dù data/config giống hệt (chưa cố định random seed) |
  | nf-unsw-nb15-v2 | 0.6694 | 0.6483 | **0.4266 (epoch 50)** — nhích nhẹ so với lượt 5 (0.4190) | **0.3797 (epoch 79, chạy đủ 80/80, không early-stop)** — gần như không đổi so với lượt 5 (0.3796) |

  **Kết luận quan trọng: tăng `WINDOW_SIZE` 2.000→5.000 KHÔNG cải thiện đáng kể như kỳ vọng** — giả thuyết "cửa sổ quá nhỏ thiếu ngữ cảnh" bị loại, không phải nguyên nhân chính.

  **Confusion matrix (lần đầu có dữ liệu thật) hé lộ đúng nguyên nhân:**
  - **CSE-CIC-IDS2018 (GCN, 0.7357)**: 11/15 lớp đạt 99-100% chính xác — F1-macro thấp hoàn toàn do 4 lớp: `Brute Force -Web`, `Brute Force -XSS`, `SQL Injection` (đều 0%, bị đoán nhầm thành `DDOS attack-HOIC` — các tấn công dựa trên web có cấu trúc đồ thị giống DDoS, GCN không có `edge_dim` nên dễ nhầm dựa trên cấu trúc) và `Infilteration` (18.6%, bị đoán nhầm thành `Benign` — tấn công loại này vốn thiết kế để lẩn vào lưu lượng bình thường, hạn chế đã biết trong y văn NIDS, không phải lỗi pipeline).
  - **UNSW-NB15-v2 (GAT)**: `Analysis`, `Backdoor`, `DoS`, `Shellcode` đều 0%, **bị "dồn" có hệ thống vào đúng 1 lớp — `Reconnaissance`** (750-834 lần mỗi lớp) — không phải nhầm lẫn ngẫu nhiên. Đây khớp chính xác với hiện tượng **"chồng lấn lớp" (class overlap)** đã tìm thấy trong tài liệu nghiên cứu độc lập về UNSW-NB15 — đặc trưng luồng mạng của các loại tấn công này vốn dĩ rất giống nhau trong chính bộ dữ liệu, không phải lỗi đồ thị/mô hình.

  **Đánh giá tổng thể sau 6 lượt**: đã loại hết các nguyên nhân "lỗi pipeline" có thể sửa được (xáo trộn thứ tự, sụp đổ do class weight/scheduler, cửa sổ quá nhỏ). Nguyên nhân còn lại của UNSW-NB15 giờ có bằng chứng cụ thể (confusion matrix) + tài liệu độc lập xác nhận: **hạn chế cố hữu của chính bộ dữ liệu**, không phải điều sửa được bằng tinh chỉnh siêu tham số/cấu trúc cửa sổ.

- **2026-07-19 — lượt 7: thử Transfer Learning cho UNSW-NB15-v2.** Trước khi làm, cân nhắc thêm **Focal Loss** (kỹ thuật tập trung vào mẫu khó/dễ nhầm, tưởng hợp với vấn đề chồng lấn lớp) nhưng **loại bỏ** — nghiên cứu cho kết quả trái chiều cụ thể với GNN (*"class-balanced focal loss did not yield benefits for several... GNN models"*), trong khi cross-entropy có trọng số lớp (cách đang dùng) được xác nhận là lựa chọn ổn định hơn ở 1 nghiên cứu khác. Chọn hướng có cơ sở chắc hơn: **Transfer Learning**.

  **Cách làm:** viết file mới `src/models/train_gnn_transfer.py` (không sửa `train_gnn.py` gốc, giữ nguyên kết quả train-từ-đầu để đối chiếu) — nạp trọng số đã train trên CSE-CIC-IDS2018 (`gcn_best.pt`, `gat_best.pt`, 13.224 đồ thị) làm điểm khởi đầu cho UNSW-NB15 (668 đồ thị), thay vì khởi tạo ngẫu nhiên, rồi fine-tune với learning rate thấp hơn 10 lần (`FINE_TUNE_LEARNING_RATE`).

  **Vấn đề kỹ thuật xử lý:** 2 bộ dữ liệu có **số lớp khác nhau** (15 vs 10) nên không thể nạp nguyên trọng số — viết hàm `load_transferable_weights()` (`train_gnn.py`) chỉ nạp các tham số **cùng kích thước** (các lớp truyền thông điệp — không phụ thuộc số lớp), tự động bỏ qua lớp phân loại cuối cùng (phụ thuộc số lớp, giữ nguyên khởi tạo ngẫu nhiên). Đã test cục bộ: nạp đúng 6/8 tham số, bỏ đúng 2 tham số lớp cuối, không lỗi.

  **⚠️ Lưu ý quan trọng khi viết báo cáo**: đây là **thử nghiệm bổ sung** nhằm cải thiện kết quả within-dataset (RQ1) của UNSW-NB15 — **khác hoàn toàn** với Thí nghiệm 2/RQ2 (train xong 1 bộ, test thẳng sang bộ kia, KHÔNG tinh chỉnh gì) — phải trình bày tách bạch 2 thí nghiệm này, không gộp chung kẻo hội đồng hiểu nhầm phá vỡ tính khách quan của RQ2.

  Kết quả (`*_transfer_best.pt`, `*_transfer_confusion_matrix.csv`, lưu riêng không ghi đè model train-từ-đầu) — **cần chạy trên Colab để có số liệu thật**, cập nhật bảng so sánh sau khi có kết quả.

- **2026-07-24 — rà soát bộ chỉ số đánh giá so với kế hoạch (mục 6.1 `docs/00_research_plan.md`).** Kế hoạch định nghĩa 6 chỉ số: Accuracy (tham khảo), Precision, Recall, F1-macro (chính), AUC-ROC, MCC — cộng thêm kiểm định McNemar (so sánh có ý nghĩa thống kê). Code hiện tại (`train_gnn.py`) mới chỉ tính **Accuracy + F1-macro + Confusion matrix** — thiếu Precision, Recall, AUC-ROC, MCC.

  **Giải thích 4 chỉ số còn thiếu** (đều tính kiểu macro — trung bình cộng không trọng số qua các lớp, cùng tinh thần F1-macro):
  - **Precision (macro):** `Precision_lớp_i = TP_i / (TP_i + FP_i)` — trong số các mẫu bị gắn nhãn "lớp i", bao nhiêu % đúng thật. Precision thấp = báo động giả nhiều (gây "mệt cảnh báo" cho đội vận hành, tốn công điều tra nhầm).
  - **Recall (macro)** (= Detection Rate, DR — cách gọi trong nhiều bài báo NIDS, vd Sarhan et al.): `Recall_lớp_i = TP_i / (TP_i + FN_i)` — trong số mẫu thật sự thuộc lớp i, bắt được bao nhiêu %. Recall thấp = bỏ sót tấn công thật — **nguy hiểm hơn Precision thấp** trong bối cảnh an ninh mạng.
  - **AUC-ROC (macro, one-vs-rest):** diện tích dưới đường cong ROC (True Positive Rate vs False Positive Rate ở mọi ngưỡng quyết định) — với đa lớp, tính kiểu "1-vs-còn-lại" cho từng lớp rồi lấy trung bình. Giá trị 0.5 = đoán ngẫu nhiên, 1.0 = phân loại hoàn hảo. Khác các chỉ số trên: **không phụ thuộc 1 ngưỡng quyết định cụ thể**, cần xác suất dự đoán (softmax) chứ không chỉ nhãn cuối cùng.
  - **MCC (Matthews Correlation Coefficient):** hệ số tương quan giữa dự đoán và nhãn thật, dùng toàn bộ 4 thành phần TP/TN/FP/FN cùng lúc trong 1 công thức (`sklearn.metrics.matthews_corrcoef` hỗ trợ đa lớp sẵn). Giá trị từ -1 (dự đoán ngược hoàn toàn) đến 0 (ngẫu nhiên) đến +1 (hoàn hảo). Nhiều nghiên cứu (Chicco & Jurman, 2020) khuyến nghị đây là chỉ số **đáng tin cậy nhất khi dữ liệu mất cân bằng nghiêm trọng** — vì chỉ đạt điểm cao khi model dự đoán tốt đồng thời cả lớp đa số lẫn thiểu số, không thể "lách" bằng cách chỉ đoán tốt lớp đông (khác Accuracy/Weighted F1 dễ bị lớp đông chi phối — xem ví dụ thực tế đã phân tích ở bảng Sarhan et al. mục trên).

  **Quyết định:** chưa thêm code ngay bây giờ — để dành **thêm vào cùng lúc với việc chạy Thí nghiệm 1 chính thức trên tập test ở Giai đoạn 4** (lúc đó mới là lần tính "điểm thật" duy nhất, tránh phải chạy lại đánh giá nhiều lần). Các lượt train hiện tại (Giai đoạn 3, dùng tập val để tinh chỉnh) vẫn chỉ cần Accuracy + F1-macro + Confusion matrix là đủ cho mục đích chọn mô hình/siêu tham số.

- **2026-07-19 — lượt 8: làm giàu đặc trưng node (4→43 chiều) + tăng `HIDDEN_DIM` (64→128), áp dụng cho cả 2 bộ, cả GCN lẫn GAT.** Chi tiết đầy đủ (vì sao Node2Vec/line-graph bất khả thi, cách thay thế) xem [`docs/decisions.md`](../decisions.md) mục "Làm giàu đặc trưng node". Đã test cục bộ (shape, tốc độ, unit test) — **cần chạy lại Graph Builder + train lại toàn bộ (cả train-từ-đầu lẫn transfer learning) để có kết quả thật.**

  **2026-07-24 — kết quả thật trên Colab (lượt 8, train-từ-đầu, sau khi sửa OOM bằng chia shard — xem `docs/decisions.md` mục 2026-07-24):** `val_f1_macro` tốt nhất:

  | Bộ dữ liệu | Random Forest | XGBoost | GCN | GAT |
  |---|---|---|---|---|
  | nf-cse-cic-ids2018-v2 | 0.7479 | 0.8115 | **0.7437 (epoch 16)** — cao nhất từ trước tới giờ, nhích nhẹ so với đỉnh cũ (0.7390, lượt 5) | 0.6727 (epoch 2) — xấp xỉ đỉnh cũ (0.6751, lượt 5), nhưng **mất ổn định giữa chừng** (val_acc rơi từ 0.994 xuống 0.944-0.948 ở epoch 9-11 rồi hồi phục một phần đến 0.9716 ở epoch 17 lúc dừng sớm) |
  | nf-unsw-nb15-v2 | 0.6694 | 0.6483 | 0.4024 (epoch 17) — **giảm** so với đỉnh cũ (0.4738, lượt 3; 0.4266, lượt 6) | 0.3721 (epoch 10) — **giảm** so với đỉnh cũ (0.4189, lượt 3; 0.3797, lượt 6) |

  **Đánh giá: làm giàu đặc trưng node có lợi cho CSE-CIC-IDS2018 nhưng có hại cho UNSW-NB15-v2 — không đồng đều giữa 2 bộ như kỳ vọng ban đầu.**

  **Giả thuyết nguyên nhân (chưa kiểm chứng thêm, cần lưu ý khi phân tích):** lượt này tăng cùng lúc 2 thứ làm phình to số tham số/độ phức tạp đầu vào — đặc trưng node (4→43, ~11 lần) và `HIDDEN_DIM` (64→128, gấp đôi tham số model). CSE-CIC có 13.224 đồ thị train — đủ dữ liệu để tận dụng model lớn hơn. UNSW-NB15-v2 chỉ có **668 đồ thị train** — tăng độ phức tạp đầu vào + gấp đôi tham số model trên cùng lượng dữ liệu ít ỏi này có khả năng gây khó tối ưu/overfit nhẹ (loss vẫn giảm đều nhưng F1-macro không theo kịp, thấp hơn cấu hình đơn giản hơn trước đó).

  **Confusion matrix UNSW-NB15 (GCN, 0.4024):** vẫn cùng kiểu lỗi đã ghi nhận từ lượt 6 — `Backdoor`, `DoS`, `Shellcode`, `Worms` đều 0% (dồn vào `Fuzzers`/`Reconnaissance`/`Exploits`) — xác nhận lại vấn đề chồng lấn lớp cố hữu của bộ dữ liệu, không phải lỗi mới phát sinh từ đặc trưng node.

  **Việc cần làm tiếp:** chạy `train_gnn_transfer.py` (lượt 7, đã code nhưng log lần này chưa thấy chạy) — kỳ vọng mượn trọng số từ CSE-CIC (nhiều dữ liệu) sẽ bù lại được phần thiệt của UNSW-NB15 khi model lớn hơn nhưng dữ liệu ít. Nếu transfer vẫn không cải thiện, cân nhắc tách riêng `HIDDEN_DIM`/`NODE_FEATURE_DIM` nhỏ hơn cho UNSW-NB15 (tương tự cách đã tách `WINDOW_SIZE_BY_DATASET`).

## ⚠️ 2026-07-26 — ĐỔI SANG BÀI TOÁN NHỊ PHÂN, chỉ còn GraphSAGE — bảng dưới đây KHÔNG so sánh được với các bảng đa lớp phía trên

**Bối cảnh đầy đủ + lý do: xem `docs/decisions.md` mục 2026-07-26.** Tóm tắt: đọc bài báo "Few Edges Are Enough" (arXiv:2501.16964) thấy E-GraphSAGE báo cáo F1=96.02% trên NF-CSE-CIC-IDS2018-v2 — sau khi xác minh, đó là **F1-macro trên bài toán nhị phân** (attack/benign), không phải đa lớp. Quyết định (của người thực hiện đề tài, sau khi đã cảnh báo đánh đổi): đổi hẳn sang nhị phân + chỉ dùng GraphSAGE (bỏ GCN/GAT), ưu tiên điểm số dễ đạt và demo dễ hiểu hơn.

**Baseline train lại trên nhãn nhị phân (`Label`), kết quả thật (tập val):**

| Bộ dữ liệu | Model | Accuracy | F1-macro |
|---|---|---|---|
| nf-cse-cic-ids2018-v2 | Random Forest | 0.9939 | 0.9855 |
| nf-cse-cic-ids2018-v2 | XGBoost | 0.9959 | 0.9900 |
| nf-unsw-nb15-v2 | Random Forest | 0.9977 | 0.9850 |
| nf-unsw-nb15-v2 | XGBoost | 0.9975 | 0.9837 |

**So với baseline đa lớp cũ (bảng đầu file, không xoá — giữ để đối chiếu):** CSE-CIC RF 0.7479→0.9855, XGBoost 0.8115→0.9900; UNSW-NB15 RF 0.6694→0.9850, XGBoost 0.6483→0.9837 — **tăng vọt ~0.2-0.35 điểm F1-macro**. Đúng như đã phân tích trước khi đổi hướng: đây là do bài toán **dễ hơn hẳn** (2 lớp thay vì 15/10 lớp), không phải do baseline "học tốt hơn" — cần diễn đạt đúng bản chất này khi viết báo cáo, tránh gây hiểu nhầm là có đột phá về phương pháp.

**Lỗi kỹ thuật phát sinh + đã sửa:** `models/baselines.py` — `build_xgboost()` dùng cứng `objective="multi:softprob"` (viết cho đa lớp), khi chạy với 2 lớp gây lỗi tương thích đã biết của XGBoost (`.predict()` trả về mảng 2 chiều thay vì nhãn đơn, sklearn báo `"mix of binary and multilabel-indicator targets"`). Sửa: thêm nhánh dùng `objective="binary:logistic"` (không truyền `num_class`) khi `num_classes == 2`.

**Việc cần làm tiếp:** dựng lại Graph Builder (đã xong, cả 2 bộ, nhãn nhị phân) → train GraphSAGE trên Colab → cập nhật bảng so sánh đầy đủ (baseline vs GraphSAGE, nhị phân).

**2026-07-26 — GraphSAGE train xong trên Colab (`val_f1_macro` tốt nhất, nhãn nhị phân):**

| Bộ dữ liệu | Random Forest | XGBoost | GraphSAGE |
|---|---|---|---|
| nf-cse-cic-ids2018-v2 | 0.9855 | **0.9900** | 0.9880 (epoch 16) |
| nf-unsw-nb15-v2 | **0.9850** | 0.9837 | 0.9778 (epoch 32) |

**Đánh giá thẳng thắn: GraphSAGE KHÔNG vượt trội baseline như kỳ vọng khi đổi hướng.** CSE-CIC: GraphSAGE nằm giữa RF và XGBoost, XGBoost vẫn cao nhất. UNSW-NB15: GraphSAGE **thua cả 2 baseline**. Vẫn là câu trả lời hợp lệ cho RQ1 (GNN hiệu quả tương đương ML truyền thống trên bài toán nhị phân), nhưng không có "đột phá" như có thể đã kỳ vọng lúc quyết định đổi hướng (xem mục trên, ngày 2026-07-26).

**Chất lượng huấn luyện:** ổn định, không sụp đổ — confusion matrix cho thấy CSE-CIC: Benign 100%, Attack 95.9% đúng; UNSW-NB15: cả 2 lớp đều >98.7% đúng. Có 1-2 lần loss tăng đột biến giữa chừng (epoch 20 CSE-CIC, epoch 36 UNSW-NB15) rồi tự phục hồi — không ảnh hưởng kết quả cuối vì cơ chế lưu checkpoint tốt nhất theo `val_f1_macro` (không phải epoch cuối).

**Lỗi phát hiện + đã sửa (không ảnh hưởng số liệu F1-macro/accuracy ở trên):** `train_gnn.py` hàm `load_class_names()` đọc nhầm `attack_label_mapping.json` (file ánh xạ tên cho bài toán **đa lớp** cũ) khi hiển thị confusion matrix cho bài toán **nhị phân** — khiến tên lớp hiện sai (vd "Bot" thay vì "Attack" cho CSE-CIC, "Backdoor" thay vì "Attack" cho UNSW-NB15). Đã sửa: khi `num_classes == 2`, trả thẳng `["Benign", "Attack"]`, không đọc file mapping đa lớp nữa. File confusion matrix CSV đã lưu trên Drive từ lượt train này vẫn còn tên sai — nếu cần dùng cho báo cáo, phải chạy lại suy luận (không cần train lại) để tạo file mới với tên đúng.

## 2026-07-26 — Thí nghiệm 1 (TN1) chính thức trên tập TEST — thêm đủ 6 chỉ số + kết quả cuối cùng

**Bổ sung code:** theo kế hoạch đã ghi ở mục trước (2026-07-24) — thêm `src/models/metrics.py` (hàm `compute_full_metrics()` dùng chung, tính Accuracy, Precision-macro, Recall-macro, F1-macro, AUC-ROC, MCC bằng `sklearn.metrics`) và `src/models/evaluate_test.py` (script **chỉ đọc** model đã train (`random_forest.joblib`, `xgboost.joblib`, `graphsage_best.pt`) + tập `test.parquet`/`test_graphs.pt`, chạy suy luận **đúng 1 lần**, không train/tinh chỉnh gì — đúng nguyên tắc `docs/00_research_plan.md` mục 4.1). Kết quả lưu tại `data/processed/test_metrics.csv`.

**Kết quả chính thức trên tập TEST (chưa từng dùng để train/chọn checkpoint):**

**CSE-CIC-IDS2018-v2:**

| Model | Accuracy | Precision | Recall | F1-macro | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.9940 | 0.9879 | 0.9832 | 0.9856 | 0.9862 | 0.9711 |
| XGBoost | 0.9959 | **0.9975** | 0.9829 | **0.9901** | **0.9931** | **0.9804** |
| GraphSAGE | 0.9950 | 0.9969 | 0.9795 | 0.9880 | 0.9888 | 0.9763 |

**UNSW-NB15-v2:**

| Model | Accuracy | Precision | Recall | F1-macro | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.9977 | 0.9835 | 0.9867 | **0.9851** | 0.9995 | **0.9702** |
| XGBoost | 0.9975 | 0.9798 | 0.9878 | 0.9838 | **0.9998** | 0.9676 |
| GraphSAGE | 0.9964 | 0.9626 | **0.9938** | 0.9776 | 0.9994 | 0.9559 |

**Nhận xét 1 — Xếp hạng không đổi so với val (XGBoost/RF vẫn nhỉnh hơn GraphSAGE trên F1-macro/MCC ở cả 2 bộ):**
- CSE-CIC: XGBoost 🥇 (0.9901) > GraphSAGE 🥈 (0.9880) > RF 🥉 (0.9856)
- UNSW-NB15: RF 🥇 (0.9851) > XGBoost 🥈 (0.9838) > GraphSAGE 🥉 (0.9776)

**Nhận xét 2 — Val và test gần như giống hệt nhau, tín hiệu phương pháp luận tốt:** vd GraphSAGE CSE-CIC val F1=0.9880 → test F1=0.9880 (y hệt); GraphSAGE UNSW-NB15 val F1=0.9778 → test F1=0.9776 (chênh 0.0002); các cặp RF/XGBoost val-test cũng chênh lệch tối đa ~0.001. **Chứng minh việc chọn checkpoint theo val không bị "học tủ"/overfit lên tập val** — kết quả tổng quát hoá tốt sang dữ liệu hoàn toàn chưa từng thấy, đáng tin cậy để báo cáo.

**Nhận xét 3 — GraphSAGE đánh đổi Precision lấy Recall ở UNSW-NB15:** GraphSAGE đạt Recall cao nhất (0.9938) trong cả 3 model, dù Precision thấp nhất (0.9626). Theo đúng nguyên tắc đã đặt ra ở `docs/00_research_plan.md` mục 6.1 ("bỏ sót tấn công nguy hiểm hơn báo động nhầm"), đây là điểm mạnh có thể nêu trong báo cáo: GraphSAGE bắt được nhiều tấn công thật hơn, đổi lại báo động nhầm nhiều hơn một chút — một góc nhìn tích cực dù F1-macro tổng thể thấp hơn baseline.

**Kết luận RQ1 (chính thức):** GNN (GraphSAGE) đạt hiệu quả **tương đương, cạnh tranh được** với ML truyền thống (RF/XGBoost) trên bài toán phân loại nhị phân — không vượt trội nhưng cũng không thua kém đáng kể (chênh lệch F1-macro ~0.006-0.02), riêng về Recall ở UNSW-NB15 thì GraphSAGE còn tốt hơn baseline.
