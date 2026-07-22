# Kế hoạch triển khai GraphSAGE (E-GraphSAGE)

**Trạng thái:** Chưa triển khai — tài liệu chuẩn bị trước để dùng khi bắt tay vào làm.

**Điều kiện thực hiện** (theo `docs/00_research_plan.md` mục 5.3.4): đây là kiến trúc GNN thứ 3, **tuỳ chọn — chỉ làm nếu còn thời gian sau khi GCN/GAT đã ổn định**. Hiện GCN/GAT vẫn đang trong quá trình tinh chỉnh (đặc biệt bộ UNSW-NB15), nên **chưa bắt đầu việc này**. Không bắt buộc để hoàn thành đề tài — so sánh GCN (đơn giản) với GAT (attention) đã đủ trả lời RQ1 nếu không kịp làm thêm GraphSAGE.

## 1. Dữ liệu sử dụng

**Giống hệt GCN/GAT — cả 2 bộ dữ liệu** (`nf-cse-cic-ids2018-v2`, `nf-unsw-nb15-v2`), dùng lại đúng file đồ thị đã có sẵn (`data/processed/<bộ>/{train,val,test}_graphs.pt`) — không cần chuẩn bị dữ liệu mới, không cần chạy lại Graph Builder. Đúng nguyên tắc đã thống nhất: mỗi kiến trúc train/đánh giá riêng trên từng bộ, để so sánh công bằng ở RQ1.

## 2. Vì sao chọn E-GraphSAGE, không phải GraphSAGE gốc

GraphSAGE nguyên bản chỉ dùng đặc trưng **node**, không có cơ chế đưa đặc trưng **cạnh** vào. Với dữ liệu của đề tài, 39/43 đặc trưng luồng mạng thật (byte, thời lượng, cờ TCP...) đều nằm ở **cạnh** — nếu dùng GraphSAGE gốc sẽ bỏ phí gần hết thông tin quan trọng. **E-GraphSAGE** (Lo và cộng sự, 2021 — đã trích dẫn trong `Nhom_A05/chapters/chap_02_methodology.tex`) là bản mở rộng chính thức của GraphSAGE dành riêng cho bài toán network intrusion detection, đưa đặc trưng cạnh vào ngay bước tổng hợp (aggregation) — đây là kiến trúc cần triển khai, không phải bản gốc.

Thiết kế đồ thị của E-GraphSAGE khớp 100% với Graph Builder đã có: **node = cặp (IP, port)**, **cạnh = luồng mạng (flow)**.

## 3. Công thức toán học (E-GraphSAGE, theo đúng bài báo gốc)

Với mỗi đỉnh `u`, tại mỗi lớp truyền thông điệp:

**Bước 1 — Tính "thông điệp" từ mỗi hàng xóm `v`** (kết hợp embedding của hàng xóm + đặc trưng cạnh nối `u`-`v`):

```
φ(x_u, x_v, e_vu) = W1 · [x_v ; e_vu]
```
(`[a ; b]` = ghép nối 2 vector, `W1` = ma trận trọng số học được)

**Bước 2 — Tổng hợp (aggregate) thông điệp từ tất cả hàng xóm** (lấy trung bình/tổng, GraphSAGE gốc dùng tối đa ~15 hàng xóm lấy mẫu ngẫu nhiên để giữ khả năng mở rộng quy mô):

```
a = Σ_{v ∈ N(u)} φ_v
```

**Bước 3 — Cập nhật embedding của chính đỉnh `u`** (kết hợp embedding cũ của `u` với thông tin vừa tổng hợp):

```
h_u = σ(W2 · [x_u ; a])
```

**Bước 4 — Sau K lớp, ghép embedding 2 đầu cạnh để phân loại** (giống hệt cách GCN/GAT hiện tại đang làm):

```
h_uv = [h_u ; h_v]  →  đưa qua lớp phân loại đa lớp (kèm thêm edge_attr gốc, theo đúng pattern GCN/GAT đã dùng)
```

## 4. Lưu ý kỹ thuật quan trọng — khác GCN/GAT ở chỗ này

Đã kiểm tra: `torch_geometric.nn.SAGEConv` (lớp có sẵn của PyTorch Geometric) **không hỗ trợ `edge_dim`** (khác với `GATConv` đang dùng cho GAT). Nghĩa là **không thể** chỉ gọi `SAGEConv(..., edge_dim=39)` như đã làm với GAT.

**Cách xử lý**: phải tự viết 1 lớp message passing tuỳ chỉnh, kế thừa `torch_geometric.nn.MessagePassing` (lớp nền có sẵn trong PyG, đã xác nhận dùng được), tự cài đặt đúng 3 bước công thức ở mục 3 (`message()`, `aggregate()`, `update()`) — không phức tạp, nhưng là điểm khác biệt so với `gcn.py`/`gat.py` (chỉ cần gọi lớp có sẵn).

## 5. File sẽ viết (theo đúng pattern đã dùng cho GCN/GAT)

| File | Nội dung |
|---|---|
| `src/models/sage_layer.py` | Lớp `EGraphSAGEConv` tuỳ chỉnh, kế thừa `MessagePassing`, cài đúng công thức mục 3 |
| `src/models/graphsage.py` | `GraphSAGEEdgeClassifier` — xếp chồng nhiều `EGraphSAGEConv`, ghép embedding 2 đầu cạnh + `edge_attr` gốc, đưa qua lớp phân loại đa lớp (giống hệt cấu trúc `gcn.py`/`gat.py`) |
| `src/models/gnn_config.py` | Thêm siêu tham số riêng nếu cần (vd `SAGE_NUM_SAMPLES = 15` — số hàng xóm lấy mẫu tối đa, theo đúng bài báo gốc) |
| `src/models/train_gnn.py` | Thêm `"graphsage": GraphSAGEEdgeClassifier(...)` vào dict `models` trong hàm `run()` — tái sử dụng nguyên vẹn vòng lặp train/early-stopping/confusion-matrix đã có, không cần viết lại |

## 6. Việc cần làm khi bắt tay vào (tóm tắt các bước)

1. Đợi GCN/GAT ổn định (đặc biệt UNSW-NB15 đạt kết quả chấp nhận được).
2. Viết `sage_layer.py` + `graphsage.py` theo đúng công thức mục 3.
3. Test cục bộ trên mẫu nhỏ (giống cách đã làm với GCN/GAT) trước khi chạy Colab.
4. Thêm vào `train_gnn.py`, chạy full trên cả 2 bộ dữ liệu.
5. Cập nhật `docs/phases/phase3_model_training.md` với kết quả (thêm cột GraphSAGE vào bảng so sánh).
6. Nếu kết quả tốt hơn cả GCN lẫn GAT — cần xem lại phần lý thuyết Chương 1/2 của báo cáo, bổ sung mô tả kiến trúc này (hiện `chap_02_methodology.tex` mới nhắc GraphSAGE khá sơ lược ở phần khảo sát lý thuyết, chưa mô tả như 1 mô hình chính thức train/so sánh).

## Nguồn tham khảo

- Lo, W. W., Layeghy, S., Sarhan, M., Gallagher, M., & Portmann, M. (2021). *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT*. [arXiv:2103.16329](https://arxiv.org/abs/2103.16329) — công bố chính thức tại NOMS 2022 (IEEE/IFIP Network Operations and Management Symposium).
