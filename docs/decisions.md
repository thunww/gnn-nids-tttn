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
