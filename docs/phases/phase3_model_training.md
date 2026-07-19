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
