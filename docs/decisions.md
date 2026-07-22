# Quyết định kỹ thuật quan trọng

Nhật ký các quyết định ảnh hưởng nhiều giai đoạn/nhiều file — tránh lặp lại giải thích rải rác trong từng `docs/phases/*.md`, các phase liên quan chỉ trỏ link về đây.

## 2026-07-18 — Bài toán phân loại: đa lớp, không phải nhị phân

**Quyết định:** Cả baseline (Random Forest, XGBoost) lẫn GNN (GCN, GAT) đều dự đoán **đa lớp** — cột `Attack_encoded` (Benign hoặc 1 trong các loại tấn công cụ thể: DDoS, SSH-Bruteforce, PortScan...) — không phải nhị phân (`Label`: 0/1, chỉ "bình thường/tấn công").

**Bối cảnh phát sinh:** Lúc viết `src/graph/build_graph.py` ở Giai đoạn 2, đã lỡ gán nhãn cạnh (`y`) là `Label` (nhị phân) — khớp với mô tả trong `Nhom_A05/chapters/chap_02_methodology.tex` ("phân loại nhị phân") và bảng phân công nhiệm vụ tự soạn của nhóm. Trong khi đó `src/models/config.py` (baseline, viết sau) lại dùng `Attack_encoded` (đa lớp) — khớp với `docs/00_research_plan.md` mục 2.1 ("bài toán được tiếp cận dưới dạng phân loại đa lớp có giám sát"). Hai phía không khớp nhau — không thể so sánh công bằng GNN với baseline nếu 2 bên giải 2 bài toán khác nhau.

**Lý do chọn đa lớp** (thay vì sửa ngược lại thành nhị phân):
1. Baseline đã train xong theo đa lớp — chọn đa lớp thì giữ nguyên baseline, chỉ cần sửa lại Graph Builder (ít việc hơn).
2. Mục 6.1 của `docs/00_research_plan.md` mô tả F1-macro là "trung bình... trên cả các lớp tấn công hiếm" — ngầm định nhiều lớp tấn công cụ thể, không phải 1 nhãn "tấn công" gộp chung.
3. Giá trị khoa học/báo cáo cao hơn: biết chính xác loại tấn công hữu ích hơn cho phân tích (đặc biệt khi làm GNNExplainer, mục 5.4) so với chỉ biết "có/không bất thường".

**Đã sửa:**
- `src/etl/config.py`: thêm hằng số `ATTACK_ENCODED_COL = "Attack_encoded"` dùng chung.
- `src/graph/build_graph.py`: đổi nguồn nhãn `y` từ `LABEL_COL` sang `ATTACK_ENCODED_COL`.
- `src/etl/clean.py`, `src/etl/run_etl.py`, `src/graph/run_graph_builder.py`, `src/models/config.py`: dùng hằng số chung thay vì hard-code chuỗi `"Attack_encoded"`.
- Cần chạy lại `python src/graph/run_graph_builder.py data/processed` (~30 phút) để tạo lại `*_graphs.pt` với nhãn đúng — file cũ (nhị phân) đã lỗi thời.

**Còn nợ (chưa làm, cần nhớ khi viết báo cáo chính thức):**
- `Nhom_A05/chapters/chap_00_introduction.tex` (dòng "phân loại kết nối mạng bình thường/bị tấn công") và `Nhom_A05/chapters/chap_02_methodology.tex` (dòng "phân loại nhị phân (bình thường/bị tấn công)") — mô tả đang lệch với thiết kế thật (đa lớp). **Chưa sửa** (theo yêu cầu — không đụng LaTeX lúc này), cần sửa lại khi chính thức viết/hoàn thiện các chương này.

## 2026-07-19 — Sửa lỗi kiến trúc nghiêm trọng: dữ liệu bị xáo trộn thứ tự trước khi dựng đồ thị

**Vấn đề phát hiện:** Sau 2 lượt train GCN/GAT, cả 2 đều thua baseline (Random Forest/XGBoost) khá xa (xem `docs/phases/phase3_model_training.md`). Nghi ngờ có lỗi bản chất chứ không chỉ do siêu tham số, kiểm tra lại toàn bộ pipeline thì phát hiện:

- `src/etl/split.py` (`stratified_split`) dùng `sklearn.train_test_split`, **mặc định xáo trộn thứ tự dòng** trước khi chia — đúng và cần thiết cho baseline (Random Forest/XGBoost coi mỗi dòng độc lập).
- Nhưng `src/graph/run_graph_builder.py` (bản cũ) lại đọc trực tiếp từ `train.parquet`/`val.parquet`/`test.parquet` (đã bị xáo ở bước trên) rồi cắt "cửa sổ" 10.000 dòng **liên tiếp trong dữ liệu đã xáo đó**.

**Bằng chứng cụ thể:** đo tỷ lệ 2 dòng liên tiếp có cùng địa chỉ IP nguồn (network flow vốn có tính "cụm" theo thời gian):
- File gốc (raw, đúng thứ tự ghi nhận): **60.76%**
- `train.parquet` (sau `stratified_split`): chỉ còn **9.32%** — gần như bị xáo ngẫu nhiên hoàn toàn.

**Hệ quả:** mỗi "đồ thị con" đưa vào GCN/GAT trước đây thực chất là 10.000 dòng **rút ngẫu nhiên từ khắp nơi trong toàn bộ tập dữ liệu**, không phải 1 lát cắt thời gian thực của lưu lượng mạng — cấu trúc đồ thị (bậc node, ai nối với ai) gần như vô nghĩa, không phản ánh đúng mẫu hình tấn công thật (vd 1 IP quét cổng dồn dập trong 1 khung giờ thực). Đây nhiều khả năng là **nguyên nhân gốc rễ** khiến GNN thua baseline — baseline không bị ảnh hưởng vì vốn không quan tâm thứ tự dòng.

**Đã sửa:**
- `src/etl/scale.py`: `fit_scale` giờ trả thêm `upper_bound` (ngưỡng clip đã tính từ train); thêm hàm mới `apply_scale(df, feature_cols, scaler, upper_bound)` để áp dụng lại (không fit) clip + scaler cho 1 bảng dữ liệu khác.
- `src/etl/run_etl.py`: sau khi lưu `train/val/test.parquet` như cũ (dùng cho baseline), áp dụng **đúng** scaler/ngưỡng clip đã fit từ train cho bản dữ liệu **giữ nguyên thứ tự gốc** (chưa xáo/chưa chia), lưu thành `data/processed/<bộ>/full_chronological.parquet` — dành riêng cho Graph Builder.
- `src/graph/run_graph_builder.py`: đổi nguồn đọc từ `train/val/test.parquet` sang `full_chronological.parquet` — cắt cửa sổ trên dữ liệu còn nguyên thứ tự thời gian thật. Sau khi có toàn bộ đồ thị con, mới **chia danh sách đồ thị** (không phải dòng) thành 70/15/15 train/val/test.
- Đã test cục bộ: xác nhận `full_chronological.parquet` giữ đúng thứ tự (49.5% dòng liên tiếp cùng IP nguồn trên mẫu test, gần khớp file gốc thay vì gần 0 như bản cũ).

**Cần làm tiếp:** chạy lại `run_etl.py` (nhanh) → `run_graph_builder.py` (~30 phút, tạo lại toàn bộ `*_graphs.pt` với đồ thị có ý nghĩa) → train lại GCN/GAT trên Colab. Kỳ vọng kết quả GNN cải thiện rõ rệt vì đầu vào giờ mới thực sự phản ánh đúng cấu trúc mạng.

**2026-07-19 — cập nhật sau khi chạy `run_etl.py` thật trên toàn bộ dữ liệu:** kiểm tra `full_chronological.parquet` bằng chỉ số "tỷ lệ dòng liên tiếp cùng IP nguồn":
- `nf-unsw-nb15-v2`: 51.78% — khớp đúng kỳ vọng (gần bằng file gốc 60.76%).
- `nf-cse-cic-ids2018-v2`: chỉ 0.23% — lúc đầu tưởng vẫn còn lỗi, nhưng kiểm tra thẳng **file gốc CSE-CIC-IDS2018 (chưa qua ETL)** thì tỷ lệ này vốn dĩ đã chỉ 0.25% — tức bộ dữ liệu gốc này **không sắp các dòng cùng IP liền kề nhau** (khác đặc tính với UNSW-NB15), có thể do cách công cụ trích xuất flow gốc (CICFlowMeter) ghi/xuất file theo thứ tự khác (vd theo thời điểm flow kết thúc thay vì bắt đầu, xen kẽ nhiều host). `full_chronological.parquet` (0.23%) khớp gần khít file gốc (0.25%) → **xác nhận code đúng, không xáo thêm gì** — chỉ là 2 bộ dữ liệu có đặc tính thứ tự gốc khác nhau. Thứ tự dòng vẫn phản ánh đúng trình tự thời gian ghi nhận thật của file gốc, nên mục đích chính (cửa sổ = lát cắt thời gian thực, không phải mẫu ngẫu nhiên từ khắp nơi) vẫn đạt được cho cả 2 bộ.

**⚠️ Giới hạn cần ghi rõ trong phần "Hạn chế" của báo cáo:** cả 2 bộ dữ liệu (schema NetFlow V2, 45 cột) **không có cột timestamp/thời điểm** — nên không thể chứng minh tuyệt đối 100% rằng thứ tự dòng trong file gốc là đúng trình tự thời gian ghi nhận thực tế, chỉ có thể suy luận hợp lý dựa trên đặc tính công cụ xuất NetFlow (nProbe) thường xuất flow gần theo thời điểm hoàn tất. Tỷ lệ xen kẽ nhiều IP ở CSE-CIC-IDS2018 nhiều khả năng do mạng có nhiều host hoạt động đồng thời (không phải dấu hiệu mất thứ tự) nhưng không loại trừ hoàn toàn khả năng khác. **Kết luận chắc chắn được:** cách làm mới (giữ nguyên thứ tự file gốc) chắc chắn không tệ hơn, và về logic tốt hơn hẳn so với xáo ngẫu nhiên hoàn toàn (cách làm cũ, đã chứng minh sai qua `train_test_split`). Khi viết báo cáo, diễn đạt thận trọng là "giữ nguyên thứ tự ghi nhận gốc của dữ liệu" thay vì khẳng định tuyệt đối "đúng thời gian thực".

## 2026-07-19 — GNN "sụp đổ" (all-collapse) trên UNSW-NB15-v2 sau lượt cải tiến thứ 4, giảm WINDOW_SIZE để sửa

**Bối cảnh:** sau khi thêm Class-Balanced Loss + `edge_dim` cho GAT + `ReduceLROnPlateau` (lượt 4, xem `docs/phases/phase3_model_training.md`), GCN/CSE-CIC cải thiện mạnh (0.65→0.73) nhưng **cả GCN lẫn GAT trên UNSW-NB15-v2 sụp đổ hoàn toàn** — chỉ đoán 1 lớp duy nhất (`val_f1_macro` ≈ 0.098, đúng mức "đoán bừa toàn Benign").

**Nghiên cứu tìm được nguyên nhân, có căn cứ:**
1. **"All-collapse"** là hiện tượng đã ghi nhận trong nghiên cứu GNN với dữ liệu mất cân bằng — cơ chế truyền thông điệp (message passing) của GNN làm trầm trọng thêm mất cân bằng: *"information from minority nodes can be overwhelmed by majority nodes"*. Đây là điểm yếu đặc thù của GNN, baseline (coi mỗi mẫu độc lập, không lan truyền/pha trộn) không gặp phải.
2. **Cửa sổ (window) quá lớn làm loãng tín hiệu tấn công**: nghiên cứu chuyên về tham số này trong NIDS-GNN xác nhận *"With a larger snapshot, the edge/event ratio diminishes, making it more difficult to distinguish between attack events and normal events."* `WINDOW_SIZE=10000` khả năng đang pha loãng tín hiệu quá mức, đặc biệt hại cho UNSW-NB15 (ít cạnh tấn công tuyệt đối hơn CSE-CIC nhiều).
3. **`LR_SCHEDULER_PATIENCE=3` quá ngắn cho bộ ít batch/epoch**: UNSW-NB15 chỉ ~10 lượt cập nhật/epoch (333 đồ thị ÷ batch 32) → `val_f1_macro` rất nhiễu từng epoch → patience ngắn dễ kích hoạt giảm learning rate do nhiễu ngẫu nhiên (không phải chững thật), khoá cứng mô hình vào trạng thái đoán bừa ngay khi vừa rơi vào. Tìm được ví dụ thực tế dùng `patience=20` cho trường hợp tương tự — patience=3 rõ ràng quá thấp.

**Đã sửa:**
- `src/graph/config.py`: `WINDOW_SIZE` 10.000 → **2.000** (giữ nguyên `WINDOW_OVERLAP=0.5`) — tạo nhiều đồ thị hơn ~5 lần (UNSW-NB15: 477 → 2389 cửa sổ), giảm độ loãng tín hiệu tấn công mỗi cửa sổ.
- `src/models/gnn_config.py`: `EARLY_STOPPING_PATIENCE` 5→15, `LR_SCHEDULER_PATIENCE` 3→8 — bớt nhạy với nhiễu ngắn hạn.
- **Cần chạy lại toàn bộ Graph Builder** (~30 phút, có thể lâu hơn do nhiều cửa sổ hơn) rồi train lại GNN trên Colab.

## 2026-07-19 — Làm giàu đặc trưng node (thay cho Node2Vec/line-graph bất khả thi)

**Bối cảnh:** sau 6 lượt tinh chỉnh, xác định nguyên nhân UNSW-NB15 thấp là do chồng lấn lớp (xem `docs/phases/phase3_model_training.md` lượt 6). Nghiên cứu thêm để cải thiện chung cả 2 bộ (cả CSE-CIC lẫn UNSW-NB15, cả GCN lẫn GAT), tìm được: mô hình **N2V-EGS-PCA** (Node2Vec + đặc trưng cạnh + PCA) đạt Macro F1 = 93.92% trên dữ liệu tương tự — cao hơn GraphSAGE thường ~45%, lý do chính: *"utilizing a comprehensive feature set that includes both edge and node features"*.

**Vấn đề:** đặc trưng node hiện tại chỉ có 4 số thuần cấu trúc (bậc vào/ra, PageRank, clustering) — không hề biết nội dung luồng mạng thật đi qua node đó. Cách "chuẩn" trong tài liệu (Node2Vec, hoặc biến đổi "đồ thị đường" — line graph, biến mỗi luồng thành 1 node) **không khả thi trong pipeline theo-cửa-sổ hiện tại**:
- **Node2Vec riêng cho từng cửa sổ**: ước tính chỉ 5 giây/cửa sổ × ~18.892 cửa sổ (CSE-CIC) = **~26 tiếng** chỉ riêng bước dựng đồ thị — không thực tế.
- **Line graph**: 1 cửa sổ có tấn công DDoS (1 nạn nhân nhận hàng nghìn kết nối) sẽ tạo ra hàng triệu cạnh mới chỉ từ 1 node (toán tổ hợp k luồng → ~k²/2 cạnh) — rủi ro nổ bộ nhớ đúng ở loại tấn công quan trọng nhất cần phát hiện.

**Đã làm (thay thế thực tế, không cần huấn luyện gì thêm):**
- `src/graph/node_features.py`: thêm hàm `aggregate_edge_features_per_node()` — tính **trung bình cộng** (không cần model, tính trực tiếp bằng numpy, rất nhanh — 0.22s/cửa sổ 5.000 cạnh) của toàn bộ 39 đặc trưng cạnh (cả chiều vào lẫn ra) cho từng node. Ghép với 4 đặc trưng cấu trúc cũ → node giờ có **43 đặc trưng** (tăng từ 4).
- `src/graph/build_graph.py`: truyền thêm `edge_attr` vào `compute_node_features()`.
- `src/models/gnn_config.py`: `NODE_FEATURE_DIM` 4→43; đồng thời tăng `HIDDEN_DIM` 64→128 (thêm dung lượng mô hình cho bài toán 15 lớp + đầu vào node giờ phong phú hơn nhiều).
- Áp dụng cho **cả 2 bộ dữ liệu, cả GCN lẫn GAT** (không phải chỉ riêng UNSW-NB15).
- Đã test cục bộ: shape đúng (N×43), tốc độ không ảnh hưởng đáng kể, unit test cập nhật và pass.

**Cần làm tiếp:** chạy lại `run_graph_builder.py` (dựng lại toàn bộ đồ thị với đặc trưng node mới) → train lại GCN/GAT (cả train-từ-đầu lẫn transfer learning) trên Colab để có kết quả thật.
