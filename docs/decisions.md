# Quyết định kỹ thuật quan trọng

Nhật ký các quyết định ảnh hưởng nhiều giai đoạn/nhiều file — tránh lặp lại giải thích rải rác trong từng `docs/phases/*.md`, các phase liên quan chỉ trỏ link về đây.

## 📌 Tóm tắt trạng thái hiện tại (cập nhật 2026-07-27) — đọc mục này trước khi đọc phần lịch sử bên dưới

**🔗 Muốn viết báo cáo/lấy số liệu chính thức? Dùng thư mục `docs/graphsage/` — tài liệu đã lọc sạch, không lẫn thông tin cũ về GCN/GAT/đa lớp.** File `decisions.md` này và `docs/phases/*.md` chỉ là nhật ký làm việc (giữ lại lịch sử để truy vết), không phải nguồn số liệu để viết báo cáo.

File này ghi theo **thứ tự thời gian, không xoá entry cũ** — nhiều quyết định bên dưới đã bị **đảo ngược sau đó**. Để tránh nhầm lẫn khi đọc lướt, trạng thái **đang dùng thật sự** hiện tại là:

- **Bài toán: NHỊ PHÂN** (`Label`: 0=Benign/1=Attack) — KHÔNG phải đa lớp. (Đổi ngày 2026-07-26, đảo ngược quyết định đa lớp ngày 2026-07-18 bên dưới.)
- **Kiến trúc GNN: chỉ còn GraphSAGE** (đúng ra là **E-GraphSAGE**, có dùng đặc trưng cạnh) — **KHÔNG còn GCN/GAT** trong luồng train chính (`train_gnn.py`). Code `gcn.py`/`gat.py` vẫn còn trong repo (không xoá) nhưng không dùng nữa.
- **Baseline:** Random Forest + XGBoost, train trên cùng nhãn nhị phân.
- **Đã hoàn thành:** Thí nghiệm 1 (TN1, đánh giá trong-cùng-bộ trên tập test) và Thí nghiệm 2 (TN2, đánh giá chéo bộ dữ liệu) — xem `docs/phases/phase3_model_training.md` các mục ngày 2026-07-26 và 2026-07-27.
- **Chưa làm:** GNNExplainer, kiểm định McNemar, Thí nghiệm 6/mô phỏng real-time (VMware + Zeek + Suricata) — xem mục cuối file này.
- **Mọi kết quả liên quan GCN/GAT/đa lớp trong lịch sử bên dưới đều đã LỖI THỜI** — chỉ giữ lại để truy vết quá trình ra quyết định (phục vụ giải trình trước hội đồng nếu cần), không dùng để viết kết quả chính thức vào báo cáo.

---

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

## 2026-07-24 — Colab bị crash (nghi OOM) sau khi làm giàu đặc trưng node; sửa bằng chia shard thay vì giảm đặc trưng

**Vấn đề:** sau khi tăng `NODE_FEATURE_DIM` 4→43 (mục trên), `train_graphs.pt` của CSE-CIC-IDS2018 tăng từ ~5GB lên **10.9GB** (`val_graphs.pt` 2.2GB). Trên Colab free (~12-13GB RAM), cell train chạy đúng ~10 phút rồi dừng đột ngột (hiện `^C` dù người dùng không bấm gì) — dấu hiệu kinh điển của process bị hệ điều hành/Colab kill do hết RAM (OOM), không phải lỗi code.

**Đo đạc cụ thể** (kiểm tra trực tiếp trên file đã build): mỗi đồ thị con trung bình **2.695 node × 43 chiều** (đặc trưng node) **+ 2.000 cạnh × 39 chiều** (đặc trưng cạnh) = 193.885 số thực/đồ thị × 13.224 đồ thị ≈ khớp đúng 10.9GB quan sát được. Đáng chú ý: số **node** trung bình mỗi cửa sổ (2.695) còn nhiều hơn số **cạnh** (2.000 = `WINDOW_SIZE`) — vì mỗi cạnh có 2 đầu mút (IP:port) thường không trùng nhau.

**Cân nhắc 2 hướng khắc phục:**
1. **Giảm số đặc trưng node** (43→~14, chỉ giữ 4 đặc trưng cấu trúc + ~10 đặc trưng cạnh tổng hợp quan trọng nhất): giảm ~40% dung lượng, nhưng (a) phải dựng lại toàn bộ đồ thị từ đầu (~30-45 phút), (b) mất thông tin của 29 đặc trưng cạnh còn lại ở node (dù rủi ro thấp vì GAT/GCN đã "nhìn" toàn bộ 39 đặc trưng cạnh trực tiếp qua message passing, đặc trưng tổng hợp ở node chỉ là tín hiệu gợi ý thêm).
2. **Chia nhỏ file thành nhiều shard, nạp dần vào RAM** (đã chọn): không đụng đến đặc trưng đã làm giàu, không cần dựng lại đồ thị — chỉ cắt file `.pt` đã build sẵn thành nhiều mảnh, sửa vòng lặp train để mỗi lúc chỉ giữ 1 shard trong RAM.

**Lý do chọn hướng 2:** giữ nguyên toàn bộ hướng cải tiến "làm giàu đặc trưng node" đã làm và đã research kỹ (mục trên) — không đánh đổi chất lượng dữ liệu để đổi lấy RAM, đồng thời tránh phải tốn thêm 30-45 phút dựng lại đồ thị.

**Đã làm:**
- `src/models/shard_graphs.py` (mới): cắt `train_graphs.pt` đã build sẵn thành nhiều `train_graphs_shard{i}.pt`, mỗi shard ~2.200 đồ thị (~1.7GB). Chỉ chia tập **train** (đọc lại mỗi epoch) — tập **val** (2.2GB, chỉ đọc 1 lần/epoch) giữ nguyên cả file, không chia. Bộ dữ liệu nhỏ hơn `WINDOW_SIZE` shard (vd UNSW-NB15, 668 đồ thị) tự động bỏ qua, không tạo shard thừa.
- `src/models/train_gnn.py`: thêm `list_train_shards()` (tự tương thích ngược — nếu không có file shard thì trả về `[train_graphs.pt]` gốc, dùng được cho cả UNSW-NB15), `count_and_collect_labels()` (duyệt 1 lượt qua các shard để đếm tổng số đồ thị + gom nhãn tính class weight, không giữ cả list `Data` trong RAM). `train_one_model()` đổi sang nhận `train_shard_paths` + `num_train_graphs` thay vì `train_graphs: list` — mỗi epoch nạp từng shard (thứ tự xáo ngẫu nhiên), train hết batch của shard đó rồi giải phóng (`del` + `gc.collect()`) trước khi nạp shard tiếp theo. `compute_class_weights()` đổi sang nhận thẳng tensor nhãn đã gom sẵn thay vì tính lại từ `train_graphs`.
- `src/models/train_gnn_transfer.py`: cập nhật theo signature mới tương ứng.
- Đã chạy thử với dữ liệu giả (2 shard nhỏ) xác nhận toàn bộ luồng (đếm nhãn → class weights → train 2 epoch → lưu checkpoint) chạy đúng, không lỗi.
- Đã chạy `shard_graphs.py` thật trên `train_graphs.pt` cục bộ: CSE-CIC-IDS2018 → 7 shard (~1.7GB/shard, shard cuối 19MB), UNSW-NB15 giữ nguyên (668 đồ thị, không đủ lớn để cần chia).

**RAM dự kiến khi train:** tại một thời điểm chỉ cần giữ 1 shard train (~1.7GB) + val nguyên (2.2GB) trong RAM cùng lúc ≈ 3.9GB — dư dả nhiều so với ~12-13GB RAM Colab free, thay vì phải nạp hết 10.9GB một lúc như trước.

**Cần làm tiếp:** upload lại các file shard mới (+ xoá/không nén `train_graphs.pt` gốc 11GB đã dư thừa) lên Drive, cập nhật `notebooks/00_colab_bootstrap.ipynb` cho khớp tên file mới, chạy lại train GCN/GAT (cả train-từ-đầu lẫn transfer learning) trên Colab.

## 2026-07-24 — Transfer learning không cải thiện UNSW-NB15; tách riêng cấu hình model/đồ thị cho từng bộ dữ liệu

**Kết quả transfer learning (chạy thật trên Colab):** GCN transfer đạt `val_f1_macro = 0.3984` (epoch 79/80) — **kém hơn cả train-từ-đầu cùng lượt** (0.4024) và kém xa đỉnh cao nhất từng đạt (0.4738, lượt 3). Kết luận: hướng transfer learning không hiệu quả, dừng đầu tư thêm vào hướng này.

**Nguyên nhân nhiều khả năng nhất:** 15 lớp tấn công của CSE-CIC-IDS2018 (DDoS-HOIC, Botnet, Brute Force...) và 10 lớp của UNSW-NB15 (Analysis, Backdoor, Shellcode, Worms...) mô tả 2 miền tấn công gần như không trùng khái niệm — trọng số học từ CSE-CIC không phải "kiến thức nền" hữu ích cho UNSW-NB15. Learning rate fine-tune thấp (0.0001) cũng khiến quá trình học rất chậm, chưa hội tụ sau 80 epoch.

**Rà soát lại: 2 bộ dữ liệu có phù hợp làm cặp nghiên cứu không?** Kiểm tra lại `docs/00_research_plan.md` mục 4.1-4.2 (không sửa file này, chỉ đối chiếu):
- Phát hiện kế hoạch gốc ghi nhầm tên bộ dữ liệu thứ nhất là "NF-CICIDS2017" — dataset này **không tồn tại** trong bộ chuẩn hóa NetFlow của Sarhan et al. (chỉ có 4 bộ chuẩn: `NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2`). Bộ đang dùng thật trong code (`NF-CSE-CIC-IDS2018-v2`) mới là tên đúng thuộc bộ chuẩn — nhiều khả năng lúc viết kế hoạch bị nhớ/gõ nhầm tên, không phải lỗi triển khai. **Cần sửa lại tên trong `docs/00_research_plan.md`** (file này chủ động không tự sửa, để người dùng quyết định/tự sửa).
- Mục 4.2 kế hoạch gốc giải thích rõ: 2 bộ dữ liệu được chọn **CHỦ ĐÍCH vì độc lập** (khác nhóm thu thập, khác môi trường mạng, khác thời điểm) — để Thí nghiệm 2 (RQ2, đánh giá khả năng tổng quát hoá) có ý nghĩa thật sự. Nếu 2 bộ giống nhau, phép kiểm tra "model học ở A có nhận ra tấn công lạ ở B không" sẽ vô nghĩa.
- **Kết luận: việc 2 bộ "không giống nhau" (khác tập lớp tấn công) là đúng thiết kế nghiên cứu, không phải sai lầm cần sửa bằng cách đổi bộ dữ liệu.** Việc transfer learning thất bại là 1 thí nghiệm bổ sung riêng biệt (không phải RQ1/RQ2) — bản thân kết quả "kiến thức không chuyển được giữa 2 miền tấn công" cũng là 1 phát hiện hợp lệ, không phải bằng chứng chọn sai dữ liệu.
- Có tìm thấy bộ NetFlow v3 (`NF-UNSW-NB15-v3`, `NF-CSE-CIC-IDS2018-v3`) mới hơn, thêm ~10 đặc trưng thời gian thực (giải quyết đúng hạn chế "không có timestamp" đã ghi ở mục trên) — nhưng **quyết định không đổi sang v3** vì phải làm lại toàn bộ pipeline từ ETL, chi phí quá lớn so với lợi ích ở giai đoạn hiện tại. Ghi nhận làm hướng phát triển tương lai nếu còn thời gian.

**Quyết định cải tiến UNSW-NB15 (thay cho transfer learning): tách riêng cấu hình cho từng bộ dữ liệu**, có căn cứ nghiên cứu trực tiếp — tìm được mô hình GraphIDS (NeurIPS 2025) tinh chỉnh hyperparameter riêng biệt đúng trên cùng 2 bộ này (UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2), xác nhận đây là thực hành chuẩn trong lĩnh vực, không phải cách làm tuỳ tiện.

**Đã làm:**
- `src/graph/config.py`: `WINDOW_SIZE_BY_DATASET["nf-unsw-nb15-v2"]` giảm mạnh **5.000 → 500** (giữ nguyên CSE-CIC ở 2.000). Lý do khác lần thử trước (2.000→5.000, lượt 6, không cải thiện vì thử theo hướng "cửa sổ lớn hơn = nhiều ngữ cảnh hơn"): lần này đi theo hướng ngược lại — cửa sổ nhỏ hơn sinh **nhiều đồ thị con hơn** từ cùng lượng flow thô (668 đồ thị train hiện tại quá ít cho GNN học) — ước tính tăng lên gần 10 lần.
- `src/models/gnn_config.py`: thêm `HIDDEN_DIM_BY_DATASET = {"nf-unsw-nb15-v2": 32}` (giữ nguyên CSE-CIC ở 128 qua `HIDDEN_DIM` mặc định). Căn cứ: tài liệu GNN xác nhận *"smaller hidden dimension works well for smaller datasets... less likely to overfit to noise"* — khớp đúng hiện tượng quan sát được ở lượt 8 (tăng `HIDDEN_DIM` 64→128 giúp CSE-CIC nhưng làm UNSW-NB15 giảm điểm, nghi ngờ overfit do model quá lớn so với 668 đồ thị).
- `src/models/train_gnn.py`: đọc `HIDDEN_DIM_BY_DATASET.get(folder_name, HIDDEN_DIM)` khi khởi tạo model — áp dụng cho **cả train-từ-đầu** (`train_gnn.py`). **`train_gnn_transfer.py` KHÔNG đổi** — vẫn giữ `HIDDEN_DIM=128` cố định vì cần khớp kích thước với model nguồn CSE-CIC để nạp trọng số (dù transfer đã xác nhận không hiệu quả ở trên, giữ nguyên script để không phá vỡ tính nhất quán nếu sau này muốn thử lại).
- Đã test cục bộ: biên dịch, pytest, smoke test train theo shard đều pass.

**Cần làm tiếp:** `WINDOW_SIZE` của UNSW-NB15-v2 đã đổi → **bắt buộc dựng lại Graph Builder cho riêng bộ này** (CSE-CIC không đổi, không cần dựng lại) trước khi train lại.

## 2026-07-26 — Ưu tiên triển khai GraphSAGE sớm hơn kế hoạch

**Bối cảnh:** trong lúc bàn về việc chỉ chọn 1 kiến trúc GNN duy nhất để tập trung công sức (thay vì cả GCN+GAT+GraphSAGE), nghiên cứu lại y văn xem kiến trúc nào thường được đánh giá tốt nhất cho bài toán NIDS dựa trên NetFlow/phân loại cạnh. Kết quả tìm được: nhiều benchmark cho thấy **GraphSAGE (đặc biệt E-GraphSAGE, biến thể tích hợp đặc trưng cạnh) thường vượt trội GAT, GAT vượt GCN** — ví dụ 1 benchmark cụ thể: GraphSAGE 94.2% > GAT 92.3% > GCN 90.8% (accuracy). Đây **ngược hoàn toàn** với kết quả nội bộ đề tài đang có (GCN liên tục thắng GAT ở mọi lượt train, xem `docs/phases/phase3_model_training.md`).

**Diễn giải:** không mâu thuẫn — literature nói về GraphSAGE/GAT đã được tối ưu kỹ. GAT trong đề tài liên tục mất ổn định/overfitting (đã ghi nhận nhiều lần), tức chưa phát huy hết tiềm năng lý thuyết. E-GraphSAGE cũng là kiến trúc **gốc mà toàn bộ thiết kế đồ thị của đề tài lấy cảm hứng** (Lo et al. 2021 — node=IP:port, cạnh=luồng mạng), nên có căn cứ tốt để tin tưởng sẽ phù hợp với dữ liệu đã chuẩn bị.

**Quyết định:** đổi lịch trình `docs/graphsage_plan.md` (vốn ghi "làm nếu còn thời gian sau GCN/GAT") — triển khai **ngay bây giờ**, thay vì chờ tinh chỉnh GCN/GAT xong hoàn toàn trước. Chi tiết triển khai xem `docs/graphsage_plan.md` mục "Cập nhật sau khi triển khai".

**Về việc "chỉ chọn 1 kiến trúc":** quyết định giữ nguyên theo `docs/00_research_plan.md` (so sánh đủ GCN, GAT — có kiểm định McNemar) — **không xoá GCN/GAT**, chỉ thêm GraphSAGE vào cùng vòng lặp train hiện có (`train_gnn.py`), không tốn thêm công sức vận hành (cùng 1 lệnh chạy cả 3 model). Việc "chỉ tập trung 1 cái" hiểu theo nghĩa: không đầu tư thêm công sức *tinh chỉnh sâu* GAT nữa, dồn sự chú ý vào so sánh kết quả GraphSAGE mới.

## 2026-07-26 — ĐẢO NGƯỢC quyết định 2026-07-18: đổi sang bài toán NHỊ PHÂN, chỉ dùng GraphSAGE (kế hoạch, chưa thực hiện xong)

**Bối cảnh:** đọc bài báo "Few Edges Are Enough" (arXiv:2501.16964), thấy bảng so sánh ghi E-GraphSAGE (fully supervised) đạt F1 = 96.02% trên NF-CSE-CIC-IDS2018-v2 — cao hơn nhiều so với GCN của đề tài (0.7437). Kiểm tra kỹ (đọc bản HTML đầy đủ bài báo) phát hiện: con số đó là **F1-macro trên bài toán NHỊ PHÂN** (attack vs benign, gộp hết 6 loại tấn công CSE-CIC thành 1 nhãn "Attack"), không phải đa lớp (15 lớp) như đề tài đang làm — không so sánh trực tiếp được, tương tự bẫy đã gặp với bảng Sarhan et al. (weighted vs macro F1) trước đó.

**Quyết định của người thực hiện đề tài (sau khi đã được cảnh báo rõ đánh đổi):** vẫn chọn đổi sang nhị phân + chỉ dùng GraphSAGE, ưu tiên điểm số dễ đạt cao hơn và demo dễ hiểu hơn, chấp nhận đánh đổi:
- Đảo ngược hoàn toàn lý do đã chọn đa lớp ở mục 2026-07-18 (giá trị khoa học cao hơn, khớp mô tả F1-macro trong `docs/00_research_plan.md`).
- Baseline (RF/XGBoost) đã train (đa lớp, F1=0.7479/0.6694) **không còn dùng được** cho so sánh RQ1 — phải train lại trên nhãn nhị phân.
- Kiểm định McNemar (kế hoạch gốc: GAT vs RF, GAT vs GCN) không còn áp dụng — chỉ còn 1 kiến trúc GNN (GraphSAGE) so với baseline.
- `docs/00_research_plan.md` (không tự sửa file này) sẽ cần cập nhật lại phần mô tả bài toán nếu chính thức chốt hướng nhị phân.

**Kỹ thuật — mức độ ảnh hưởng thực tế (thấp hơn tưởng tượng ban đầu):** không cần chạy lại ETL — cột `Label` (nhị phân) đã có sẵn trong `full_chronological.parquet` từ đầu (ETL không hề xoá, chỉ không dùng tới). Chỉ cần đổi nguồn nhãn ở 2 nơi + dựng lại Graph Builder + train lại baseline.

**Việc cần làm (theo thứ tự), trạng thái tại thời điểm ghi:**
1. ✅ `src/graph/build_graph.py`: đổi `y` từ `ATTACK_ENCODED_COL` sang `LABEL_COL`.
2. ✅ `src/models/config.py`: đổi `TARGET_COL` sang `LABEL_COL` (cho baseline) — kèm sửa thêm 1 lỗi phát sinh: `NON_FEATURE_COLS` thiếu `ATTACK_ENCODED_COL` (trước đó trùng với `TARGET_COL` nên không cần liệt kê riêng, giờ tách ra phải thêm tay, nếu không `Attack_encoded` sẽ bị lọt vào làm feature — rò rỉ dữ liệu).
3. ✅ `src/models/train_gnn.py`: bỏ `"gcn"`, `"gat"` khỏi dict `models`, chỉ giữ `"graphsage"` (kèm dọn import không dùng nữa).
4. ✅ Dựng lại Graph Builder cho **CẢ 2 bộ dữ liệu** — chạy bởi người dùng (local, không phải tôi chạy — theo yêu cầu "không tự ý train/chạy nền"). Phát sinh: file shard cũ (đa lớp) của CSE-CIC còn sót lại, `list_train_shards()` sẽ ưu tiên đọc nhầm — đã xoá shard cũ, chạy lại `shard_graphs.py` để chia shard mới đúng nhãn nhị phân (CSE-CIC 7 shard, UNSW-NB15 giờ cũng đủ lớn để chia 4 shard — trước đây 668 đồ thị không cần, giờ 6.692 đồ thị do đã giảm `WINDOW_SIZE`).
5. ✅ Train lại baseline RF/XGBoost trên nhãn nhị phân, cả 2 bộ (local, người dùng tự chạy) — kết quả xem `docs/phases/phase3_model_training.md` mục 2026-07-26. Phát sinh lỗi `build_xgboost()` (đã sửa `baselines.py`, xem chi tiết ở đó).
6. ✅ Test cục bộ (pytest + smoke test) trước khi chạy Colab.
7. ✅ Cập nhật `docs/phases/phase3_model_training.md` (bảng so sánh giờ đổi hoàn toàn ý nghĩa, ghi rõ đây là kết quả nhị phân, tách biệt khỏi bảng đa lớp cũ).
8. ✅ Commit + push, chạy lại toàn bộ trên Colab (GraphSAGE) — đã train xong, kết quả val ghi tại `phase3_model_training.md`.

**Tiếp nối:** đã hoàn thành thêm Thí nghiệm 1 (TN1) chính thức trên tập TEST (thêm `metrics.py` + `evaluate_test.py`, đủ 6 chỉ số) — xem `docs/phases/phase3_model_training.md` mục 2026-07-26 "Thí nghiệm 1 (TN1) chính thức trên tập TEST". Bước tiếp theo (chưa làm): Thí nghiệm 2 (TN2, đánh giá chéo bộ dữ liệu), GNNExplainer, kiểm định McNemar — xem `docs/00_research_plan.md` mục 7.4.

**`train_gnn_transfer.py`:** không sửa, không dùng tiếp (đã xác nhận transfer learning không hiệu quả ở mục trước).

## 2026-07-27 — Thí nghiệm 2 (TN2) hoàn thành: kết quả kém, đã điều tra kỹ, kết luận giữ nguyên hướng đi

**Tóm tắt:** TN2 (đánh giá chéo bộ dữ liệu, `src/models/evaluate_cross_dataset.py`) cho kết quả rất kém ở cả 3 model (MCC quanh 0/âm, AUC-ROC có nơi dưới 0.5) — số liệu đầy đủ + phân tích chi tiết xem `docs/phases/phase3_model_training.md` mục 2026-07-27.

**Đã điều tra kỹ trước khi kết luận (không vội cho là lỗi hay vội chấp nhận):**
1. Kiểm tra thứ tự cột đặc trưng giữa `full_chronological.parquet` và `train.parquet` (khớp) và giữa 2 bộ dữ liệu (khớp, 46 cột cùng tên cùng thứ tự) — loại trừ khả năng lỗi ánh xạ cột.
2. Kiểm tra quy ước nhãn `Label` (0=Benign/1=Attack) giữa 2 bộ — khớp nhau, loại trừ khả năng đảo nhãn.
3. Đo trực tiếp: **69,2% số HÀNG** (mẫu) dữ liệu UNSW-NB15 sau khi quy đổi đúng cách sang thang đo CSE-CIC có **ít nhất 1** trong 39 cột đặc trưng rơi vào vùng cực đoan (|z|>5, ngoài phạm vi model từng học) — xác nhận đây là hiện tượng trôi dạt phân phối (distribution drift) thật, không phải lỗi code. *(Đo bổ sung 2026-07-27, khi làm hình minh hoạ: nếu tính theo từng GIÁ TRỊ ĐƠN LẺ, không gộp theo hàng — tức % trên tổng số ô dữ liệu (hàng × 39 cột) — tỷ lệ là **6,2%**. Cả 2 con số đều đúng, chỉ là 2 cách đo khác nhau: 69,2% trả lời "bao nhiêu % mẫu bị ảnh hưởng bởi ít nhất 1 đặc trưng lệch", 6,2% trả lời "trung bình bao nhiêu % thông tin trong mỗi mẫu bị lệch" — dùng 69,2% khi lập luận về rủi ro cấp độ mẫu, dùng 6,2% khi mô tả phân phối z-score tổng thể (xem `report_figures/zscore_distribution.png`).)*
4. Đối chiếu y văn: tìm được nghiên cứu độc lập đo lường đúng hiện tượng này (36/45 đặc trưng NIDS vượt ngưỡng trôi dạt nghiêm trọng PSI≥0.25 giữa các bộ dữ liệu khác) — xác nhận không phải hiện tượng lạ riêng của đề tài.

**Cân nhắc hướng "sửa" (self-supervised learning, kiểu Anomal-E) — quyết định KHÔNG làm ngay:**
- Đã đọc lại kỹ bài Anomal-E (arXiv:2207.06819) — **đính chính**: bài này KHÔNG hề đánh giá cross-dataset (chỉ within-dataset), thông tin trích dẫn ban đầu trong hội thoại là sai (dựa trên tóm tắt tìm kiếm chưa kiểm chứng). Không tìm được bất kỳ nghiên cứu nào công bố số liệu self-supervised GNN cho đúng kiểu TN2 trên đúng cặp dữ liệu này — đây là khoảng trống nghiên cứu, không phải giải pháp đã được chứng minh.
- Tự phân tích kỹ thuật: sở dĩ GraphSAGE không có ưu thế tổng quát hoá là do `NODE_FEATURE_DIM=43` chỉ có 4/43 chiều là cấu trúc thuần (bất biến quy mô mạng), còn 39/43 chiều + `edge_attr` đưa thẳng vào message passing đều là đặc trưng thô nhạy quy mô mạng — pha loãng gần hết ưu thế lý thuyết của kiến trúc đồ thị.
- Chuyển sang self-supervised là đổi TOÀN BỘ phạm trù huấn luyện (không phải sửa nhỏ), không có bằng chứng đảm bảo giải quyết đúng vấn đề — quyết định: **không triển khai**, ghi nhận là hướng nghiên cứu tương lai trong báo cáo.

**Kết luận cuối cùng: giữ nguyên hướng đang làm (supervised, E-GraphSAGE, nhị phân).** Kết quả TN2 kém KHÔNG phải lý do để đổi hướng — đây là câu trả lời khoa học thật, hợp lệ cho RQ2, có bằng chứng định lượng + xác nhận độc lập từ y văn, phù hợp đưa vào phần "Hạn chế và hướng phát triển" của báo cáo thay vì cố "sửa" để ra số đẹp hơn (sẽ làm mất giá trị khoa học thật của thí nghiệm).

## 2026-07-27 — Thí nghiệm 6 / mô phỏng real-time — đã xác nhận SẼ LÀM, kế hoạch (chưa bắt đầu)

**Bối cảnh:** sau khi TN1+TN2 xong, đã hỏi lại khung quyết định ở `docs/00_research_plan.md` mục 4.3 (có cần demo tấn công real-time trước hội đồng hay không). Người thực hiện đề tài xác nhận: **có**, sẽ dựng môi trường lab thật (Phương án A). Đây là công việc **khác hẳn phạm vi đã làm** (không còn là xử lý dữ liệu/train model, mà là dựng hạ tầng mạng thật) — tạm dừng, ưu tiên viết báo cáo với kết quả đã có (TN1+TN2) trước, quay lại làm sau.

**Checklist các việc cần làm khi bắt tay vào** (theo đúng `docs/00_research_plan.md` mục 4.3-4.4, phân rõ việc của người dùng vs việc code):

1. ⬜ Dựng VMware: máy tấn công + máy nạn nhân + máy giám sát, mạng **host-only, cô lập hoàn toàn khỏi Internet** (bắt buộc theo mục 4.4 — kể cả kịch bản DoS cũng chỉ chạy trong mạng ảo cục bộ). *(Người dùng tự làm.)*
2. ⬜ Cài Zeek trên máy giám sát, bật Promiscuous Mode để bắt toàn bộ traffic. *(Người dùng tự làm.)*
3. ⬜ Chạy lần lượt 5 kịch bản tấn công đã hoạch định trong máy ảo, ghi lại chính xác thời điểm/loại tấn công (để gán nhãn đúng sau này). *(Người dùng tự làm.)*
4. ⬜ Viết code chuyển log Zeek → đúng định dạng đặc trưng NetFlow V2 (43-49 cột, khớp schema đã dùng để train) — **bước kỹ thuật khó nhất**, log Zeek không giống định dạng NF-v2 (khác tên cột, khác cách tính 1 số chỉ số), sai bước này model sẽ dự đoán vô nghĩa. *(Cần code hỗ trợ.)*
5. ⬜ Viết script nạp dữ liệu đã chuyển đổi qua model đã train (GraphSAGE + baseline) để phân loại. *(Cần code hỗ trợ, tái sử dụng được `evaluate_test.py`/`evaluate_cross_dataset.py` làm khung tham khảo.)*
6. ⬜ Cài Suricata + ET Open Rules trên cùng traffic. *(Người dùng tự làm.)*
7. ⬜ Viết script so sánh kết quả model vs Suricata (TP/FP/FN từng bên) trên cùng traffic đã gán nhãn thủ công ở bước 3. *(Cần code hỗ trợ.)*

**Rủi ro kỹ thuật đã ghi nhận trước trong kế hoạch gốc:** cấu hình VMware/Promiscuous Mode, độ chính xác gán nhãn thủ công (mục 4.3, bảng khung quyết định).

## 2026-07-27 — Xác nhận demo real-time kiểu TRỰC TIẾP (không phải chạy trước phân tích sau) — thêm Phase F (server/API)

**Quyết định:** sau khi hỏi rõ, người thực hiện đề tài xác nhận muốn **demo trực tiếp trước hội đồng** — model phản ứng gần thời gian thực ngay khi tấn công diễn ra (không phải chạy tấn công trước rồi phân tích/trình bày kết quả sau). Đánh đổi: tốn thêm công sức đáng kể so với phương án đơn giản (chạy trước, phân tích sau) — cần thêm hẳn 1 phase mới.

**Kế hoạch đầy đủ (7 phase, A→G), cập nhật từ checklist cũ ở mục trên:**

- **Phase A — Hạ tầng:** dựng 3 máy ảo VMware (tấn công/nạn nhân/giám sát), mạng host-only cô lập, bật Promiscuous Mode cho máy giám sát, cài Zeek + Suricata + ET Open Rules, chuẩn bị dịch vụ trên máy nạn nhân. *(Người dùng tự làm.)*
- **Phase B — Kịch bản:** chốt rõ nội dung cụ thể của 5 kịch bản tấn công đã hoạch định trong `docs/00_research_plan.md` (chưa thấy liệt kê chi tiết ở đâu trong tài liệu hiện có — cần làm rõ trước khi chạy), chuẩn bị cách ghi lại chính xác mốc thời gian từng kịch bản.
- **Phase C — Thu thập:** chạy Zeek + Suricata song song, thực hiện lần lượt 5 kịch bản xen kẽ traffic bình thường.
- **Phase D — Chuyển đổi dữ liệu (khó nhất):** viết script log Zeek → đúng 43 cột đặc trưng NetFlow V2 khớp schema đã train; gán nhãn thủ công theo mốc thời gian.
- **Phase E — Suy luận + đánh giá (offline, kiểm chứng trước khi làm live):** script nạp model đã train chạy trên dữ liệu đã chuyển đổi (tái sử dụng khung `evaluate_test.py`), script so sánh với Suricata (TP/FP/FN).
- **Phase F — Server/API cho demo trực tiếp (MỚI, do vừa xác nhận cần demo live):** dựng API (`fastapi`/`uvicorn` đã có sẵn trong `requirements.txt`, ghi chú "giai đoạn 5" nhưng **chưa có code nào** — cần viết từ đầu) nhận traffic gần thời gian thực (Zeek ghi log liên tục → script đọc log mới xuất hiện kiểu `tail -f` → chuyển đổi đặc trưng → gọi model → trả kết quả), hiển thị kết quả (console hoặc dashboard đơn giản) ngay khi có tấn công — dùng để trình chiếu trực tiếp trước hội đồng.
- **Phase G — Báo cáo:** tổng hợp kết quả vào `docs/graphsage/`.

**Bước tiếp theo cụ thể:** bắt đầu từ Phase A (hạ tầng, người dùng tự làm) + làm rõ Phase B (nội dung 5 kịch bản tấn công) trước khi viết bất kỳ code nào ở Phase D-F.

## 2026-07-31 — ⚠️ LỖI RÒ RỈ DỮ LIỆU nghiêm trọng trong Graph Builder — ảnh hưởng TOÀN BỘ kết quả TN1 đã có, đã sửa

**Bối cảnh phát hiện:** người thực hiện đề tài nghi ngờ chỉ số GraphSAGE cao bất thường (F1-macro 0.988, cao hơn cả baseline) so với y văn (bài "Few Edges Are Enough", arXiv:2501.16964, cùng bộ CSE-CIC-IDS2018, cùng bài toán nhị phân, chỉ đạt F1=96,02%) — yêu cầu kiểm tra kỹ thay vì chỉ tra cứu thêm tài liệu.

**Nguyên nhân xác nhận (đo lường định lượng, không phải suy đoán):**
- `src/graph/windowing.py`: cửa sổ trượt chồng lấp **50%** (`WINDOW_OVERLAP=0.5`) — cửa sổ `i` và `i+1` dùng chung ~50% số flow.
- `src/graph/run_graph_builder.py` (bản cũ): chia danh sách đồ thị (cửa sổ) vào train/val/test bằng `train_test_split(graphs, ..., random_state=42)` — **xáo trộn ngẫu nhiên** trước khi chia, không quan tâm cửa sổ nào chồng lấp cửa sổ nào.
- **Đo trực tiếp (mô phỏng đúng cùng `random_state=42`, cùng số lượng cửa sổ thật của CSE-CIC-IDS2018):** trong 18.891 cặp cửa sổ liền kề (mỗi cặp chồng lấp 50%), có **8.725 cặp (46,2%) bị tách vào 2 tập khác nhau** (vd cửa sổ `i` vào train, cửa sổ `i+1` vào test) — nghĩa là cùng 1 lượng lớn flow xuất hiện ở CẢ train lẫn test, gây rò rỉ dữ liệu (data leakage) thật.

**Vì sao chỉ ảnh hưởng TN1 (GraphSAGE), không ảnh hưởng baseline hay TN2:**
- **Baseline (RF/XGBoost):** chia theo **dòng** (`stratified_split`, không qua bước cửa sổ/đồ thị) — không có khái niệm "chồng lấp cửa sổ", không bị lỗi này. Số liệu baseline vẫn đáng tin.
- **TN2 (cross-dataset):** train 1 bộ, test bộ KHÁC — 2 bộ dữ liệu không hề chồng lấp nhau, lỗi này không áp dụng được. **Kết quả TN2 (cả 3 model sụp đổ khi đổi môi trường) vẫn đúng, không cần làm lại.**
- **TN1 (GraphSAGE, within-dataset):** bị ảnh hưởng trực tiếp — **toàn bộ số liệu GraphSAGE trong TN1 đã có (F1-macro 0.988/0.978) không đáng tin cậy, cần làm lại từ đầu.**

**Đối chiếu y văn để ước lượng mức ảnh hưởng thực tế:** bài "Few Edges Are Enough" (không có lỗi rò rỉ) báo cáo E-GraphSAGE nhị phân, cùng bộ CSE-CIC-IDS2018, đạt F1=96,02%. Baseline (không lỗi) của chính đề tài đạt ~98,5-99%. → **Kỳ vọng sau khi sửa: F1-macro GraphSAGE rơi vào khoảng ~93-97%** — vẫn là kết quả tốt, chỉ không còn cao vô lý như hiện tại (98,8%).

**Đã sửa (`src/graph/run_graph_builder.py`):** thay xáo trộn ngẫu nhiên bằng **chia theo khối liên tục theo đúng thứ tự thời gian** đã dùng để dựng cửa sổ (giữ đúng nguyên tắc đã áp dụng ở bước ETL với `full_chronological.parquet`). Thêm "purge gap" — bỏ đúng 1 đồ thị ở mỗi ranh giới train/val và val/test — để loại bỏ hoàn toàn phần chồng lấp còn sót lại ở đúng đường ranh giới (chỉ mất 2 đồ thị/~18.892, không đáng kể). Đã test cục bộ (pytest pass, không ảnh hưởng logic chia đồ thị/tính đặc trưng khác).

**Việc cần làm tiếp (toàn bộ, vì train set đã đổi hoàn toàn):**
1. Chạy lại Graph Builder cho **cả 2 bộ dữ liệu** (nhãn/đặc trưng không đổi, chỉ đổi cách chia train/val/test).
2. Chạy lại `shard_graphs.py`, xoá `train_graphs.pt` gốc dư thừa (như các lần trước).
3. **Train lại GraphSAGE từ đầu** (cả 2 bộ, trên Colab) — bắt buộc, vì tập train giờ có nội dung khác hẳn (không còn tình cờ chứa 1 phần dữ liệu test).
4. Chạy lại `evaluate_test.py` (TN1) với model mới — đây mới là số liệu chính thức thật để dùng viết báo cáo.
5. Chạy lại `evaluate_cross_dataset.py` (TN2) với model mới (dù bản chất TN2 không lỗi, nhưng model nguồn đã đổi nên cần cập nhật số liệu cho khớp model mới).
6. Cập nhật toàn bộ bảng số liệu trong `docs/phases/phase3_model_training.md`, `docs/graphsage/03_ket_qua.md` — **giữ nguyên số liệu cũ (không xoá, đánh dấu rõ "đã lỗi thời do lỗi rò rỉ dữ liệu")**, thêm bảng mới bên cạnh.
7. Sinh lại các ảnh trong `report_figures/` (dùng `scripts/generate_report_figures.py`) với số liệu mới.

**2026-08-02 — Train lại xong + chạy lại TN1, xác nhận kết quả (val và test):**

Val (`train_gnn.py`, log thật trên Colab): CSE-CIC `val_f1_macro=0.9879` (epoch 17); UNSW-NB15 `val_f1_macro=0.9787` (epoch 35).

TN1 (`evaluate_test.py`, tập test, chính thức):

| Bộ dữ liệu | Model | Accuracy | Precision | Recall | F1-macro | AUC-ROC | MCC |
|---|---|---|---|---|---|---|---|
| nf-cse-cic-ids2018-v2 | Random Forest | 0.9940 | 0.9879 | 0.9832 | 0.9856 | 0.9862 | 0.9711 |
| nf-cse-cic-ids2018-v2 | XGBoost | 0.9959 | 0.9975 | 0.9829 | 0.9901 | 0.9931 | 0.9804 |
| nf-cse-cic-ids2018-v2 | GraphSAGE | 0.9950 | 0.9969 | 0.9793 | 0.9879 | 0.9888 | 0.9760 |
| nf-unsw-nb15-v2 | Random Forest | 0.9977 | 0.9835 | 0.9867 | 0.9851 | 0.9995 | 0.9702 |
| nf-unsw-nb15-v2 | XGBoost | 0.9975 | 0.9798 | 0.9878 | 0.9838 | 0.9998 | 0.9676 |
| nf-unsw-nb15-v2 | GraphSAGE | 0.9941 | 0.9659 | 0.9928 | 0.9789 | 0.9991 | 0.9583 |

**Phát hiện quan trọng: số liệu GraphSAGE gần như KHÔNG ĐỔI so với trước khi sửa lỗi** (CSE-CIC: F1-macro 0.9880→0.9879, gần như y hệt; UNSW-NB15: 0.9776→0.9789, **nhích lên**). Baseline (RF/XGBoost) y hệt tuyệt đối so với trước (không dùng lại model/data nào bị ảnh hưởng bởi lỗi).

**Giải thích tại sao sửa lỗi rò rỉ mà điểm không giảm:** GraphSAGE quyết định dựa trên `[embedding node u, embedding node v, edge_attr]` — phần embedding node phụ thuộc **toàn bộ ngữ cảnh đồ thị xung quanh** (các luồng lân cận trong cùng cửa sổ), không cố định theo từng luồng đơn lẻ. Dù 1 luồng cụ thể (cùng `edge_attr`) xuất hiện ở cả 2 cửa sổ chồng lấp (1 rơi vào train, 1 rơi vào test), **ngữ cảnh đồ thị bao quanh nó khác nhau ở mỗi cửa sổ** — nên model không thể "tra cứu" thẳng ra đáp án như baseline (chỉ nhìn đúng 1 dòng dữ liệu, không có ngữ cảnh) có thể làm được. Lỗi rò rỉ **có thật và bắt buộc phải sửa** (đúng phương pháp luận, tránh bị hội đồng bắt lỗi khi hỏi sâu), nhưng **kiểm chứng thực tế xác nhận nó không hề thổi phồng kết quả đáng kể** — tăng độ tin cậy cho toàn bộ kết luận nghiên cứu.

**Cần làm tiếp:** chạy lại TN2 (`evaluate_cross_dataset.py`) với model mới, cập nhật `docs/graphsage/03_ket_qua.md` (thay số liệu cũ bằng số liệu mới, vì đây là tài liệu tổng hợp sạch dùng viết báo cáo — không cần giữ song song 2 bảng như ở đây), sinh lại ảnh `report_figures/`.

**2026-08-02 — Đã chạy lại TN2 với model mới, xác nhận kết luận RQ2 không đổi (nếu có thay đổi thì càng chắc chắn hơn):**

| | Model | F1-macro (trước sửa) | F1-macro (sau sửa) | MCC (trước) | MCC (sau) |
|---|---|---|---|---|---|
| CSE-CIC→UNSW | Random Forest | 0.4899 | 0.4899 (y hệt) | 0.0000 | 0.0000 (y hệt) |
| CSE-CIC→UNSW | XGBoost | 0.5065 | 0.5065 (y hệt) | 0.1080 | 0.1080 (y hệt) |
| CSE-CIC→UNSW | GraphSAGE | 0.4698 | 0.4436 (giảm nhẹ) | -0.0577 | -0.1082 (tệ hơn) |
| UNSW→CSE-CIC | Random Forest | 0.4465 | 0.4465 (y hệt) | -0.1056 | -0.1056 (y hệt) |
| UNSW→CSE-CIC | XGBoost | 0.4701 | 0.4701 (y hệt) | -0.0329 | -0.0329 (y hệt) |
| UNSW→CSE-CIC | GraphSAGE | 0.3502 | 0.3398 (giảm nhẹ) | -0.2430 | -0.2753 (tệ hơn) |

Baseline y hệt tuyệt đối (không dùng lại model/data bị ảnh hưởng). GraphSAGE giảm nhẹ, không tăng — model "sạch" hơn (không còn khả năng hưởng lợi dù nhỏ từ rò rỉ) càng lộ rõ việc không tổng quát hoá được sang môi trường khác. **Kết luận RQ2 giữ nguyên, được củng cố thêm chứ không bị lung lay.**
