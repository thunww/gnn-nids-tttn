# Kiến trúc mô hình — E-GraphSAGE

**Nguồn tham khảo:** Lo, W. W., Layeghy, S., Sarhan, M., Gallagher, M., & Portmann, M. (2021). *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT*. arXiv:2103.16329 (công bố chính thức tại NOMS 2022).

## Vì sao chọn E-GraphSAGE thay vì GraphSAGE gốc

`torch_geometric.nn.SAGEConv` (lớp có sẵn của PyTorch Geometric) **không hỗ trợ đặc trưng cạnh** — chỉ dùng được đặc trưng node. Với dữ liệu NetFlow, phần lớn thông tin quan trọng (byte, thời lượng, cờ TCP...) nằm ở **cạnh**, không phải node — nếu dùng GraphSAGE gốc sẽ bỏ phí gần hết thông tin. Do đó phải **tự viết 1 lớp truyền thông điệp (message passing) tuỳ chỉnh**, kế thừa `torch_geometric.nn.MessagePassing`, đưa đặc trưng cạnh vào trực tiếp — đúng công thức E-GraphSAGE gốc.

## Công thức toán học

Với mỗi đỉnh `u`, tại mỗi lớp truyền thông điệp:

**Bước 1 — Tính "thông điệp" từ mỗi hàng xóm `v`** (kết hợp embedding của hàng xóm + đặc trưng cạnh nối `u`-`v`):

```
φ(x_v, e_vu) = W1 · [x_v ; e_vu]
```

**Bước 2 — Tổng hợp (aggregate) thông điệp từ tất cả hàng xóm** (lấy tổng):

```
a = Σ_{v ∈ N(u)} φ_v
```

**Bước 3 — Cập nhật embedding của chính đỉnh `u`:**

```
h_u = σ(W2 · [x_u ; a])
```

**Bước 4 — Sau K lớp, ghép embedding 2 đầu cạnh + đặc trưng cạnh gốc để phân loại:**

```
h_uv = [h_u ; h_v ; e_uv]  →  MLP 2 lớp  →  phân loại nhị phân (Benign/Attack)
```

## Cài đặt (mã nguồn)

| File | Nội dung |
|---|---|
| `src/models/sage_layer.py` | `EGraphSAGEConv` — lớp `MessagePassing` tuỳ chỉnh, cài đúng công thức bước 1-3 |
| `src/models/graphsage.py` | `GraphSAGEEdgeClassifier` — xếp chồng nhiều `EGraphSAGEConv`, ghép embedding 2 đầu cạnh + `edge_attr` gốc, đưa qua lớp phân loại (bước 4) |
| `src/models/metrics.py` | Tính đủ 6 chỉ số đánh giá (xem `03_ket_qua.md`) |
| `src/models/train_gnn.py` | Vòng lặp huấn luyện, early stopping, lưu checkpoint |
| `src/models/evaluate_test.py` | Đánh giá chính thức trên tập test (Thí nghiệm 1) |
| `src/models/evaluate_cross_dataset.py` | Đánh giá chéo bộ dữ liệu (Thí nghiệm 2) |

**Đơn giản hoá so với bài báo gốc:** không lấy mẫu ngẫu nhiên hàng xóm (neighbor sampling) như GraphSAGE gốc — tổng hợp qua **toàn bộ** hàng xóm, vì các cửa sổ đồ thị hiện tại không đủ lớn để sampling tạo khác biệt hiệu năng đáng kể.

## Siêu tham số (giá trị thực tế đã dùng)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Số lớp (`NUM_LAYERS`) | 2 | |
| `HIDDEN_DIM` | **128** cho `nf-cse-cic-ids2018-v2`; **32** cho `nf-unsw-nb15-v2` | Tách riêng theo bộ dữ liệu — UNSW-NB15 có ít đồ thị train hơn nhiều, model nhỏ hơn giúp giảm overfitting |
| `NODE_FEATURE_DIM` | 43 | 4 cấu trúc + 39 tổng hợp từ cạnh |
| `EDGE_FEATURE_DIM` | 39 | |
| Dropout | 0.4 | |
| `LEARNING_RATE` | 0.001 | Adam optimizer |
| `WEIGHT_DECAY` | 5e-4 | L2 regularization |
| `BATCH_SIZE` | 32 | |
| `MAX_EPOCHS` | 80 | |
| Early stopping patience | 15 epoch | Dừng nếu `val_f1_macro` không cải thiện |
| LR Scheduler | `ReduceLROnPlateau`, factor=0.5, patience=8 | Tự giảm nửa learning rate khi chững lại |
| Hàm mất mát | Class-Balanced Cross-Entropy Loss (Cui et al., CVPR 2019, β=0.999) | Trọng số theo "số mẫu hiệu quả", giảm cực đoan so với tỷ lệ nghịch trực tiếp khi mất cân bằng lớp nặng |
| Checkpoint | Lưu mỗi epoch + lưu riêng bản tốt nhất theo `val_f1_macro` | Không dùng epoch cuối cùng để tránh overfitting |

## Baseline đối chứng

| Model | Cấu hình |
|---|---|
| Random Forest | 200 cây, không giới hạn độ sâu, `class_weight="balanced"` |
| XGBoost | 300 cây, độ sâu tối đa 8, learning rate 0.1, `objective="binary:logistic"` |
