# src

Mã nguồn dự án, chia theo giai đoạn trong `docs/00_research_plan.md`:

- **etl/** — load, làm sạch, mã hoá, chuẩn hoá, chia tập 70/15/15 (stratified) cho NF-CSE-CIC-IDS2018-v2 và NF-UNSW-NB15-v2. Output ghi vào `data/processed/`. (Giai đoạn 1)
- **graph/** — Graph Builder: bảng → đồ thị (IP:port = node, flow = edge có hướng), đặc trưng cấu trúc node (in/out-degree, PageRank, clustering coefficient), sliding window 50% overlap, kèm unit test. (Giai đoạn 2)
- **models/** — kiến trúc GCN, GAT, GraphSAGE (điều kiện), baseline Random Forest/XGBoost. Bài toán edge classification. (Giai đoạn 3)
- **training/** — training loop, Optuna, MLflow tracking (trỏ Google Drive), checkpoint mỗi epoch. (Giai đoạn 3)
- **evaluation/** — metrics (F1-macro là chỉ số chính, Precision/Recall, AUC-ROC, MCC), Thí nghiệm 1 (within-dataset), Thí nghiệm 2 (cross-dataset), kiểm định McNemar. (Giai đoạn 4)
- **explainability/** — GNNExplainer cho mô hình tốt nhất. (Giai đoạn 4)
- **api/** — dịch vụ dự đoán + giải thích phục vụ demo, chỉ serving trên cloud, không tấn công trên cloud. (Giai đoạn 5)
