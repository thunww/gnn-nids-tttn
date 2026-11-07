# Bài báo và tài liệu dùng cho đồ án

Ngày tra cứu: **22/10/2026**. Danh mục gồm **15 công trình nghiên cứu** và **14 tài liệu chính thức**. Mỗi mục có vai trò cụ thể trong kế hoạch; không phải danh sách bắt buộc triển khai toàn bộ thuật toán.

**Mức kiểm tra:** đã tra trang nguồn, abstract/metadata và những phần phương pháp/API truy cập được phục vụ kế hoạch. Không coi đây là một systematic review hay khẳng định đã tái lập mọi bài báo. Khi viết tổng quan chính thức, đọc toàn văn các nguồn cốt lõi P01–P07, P09, P12 và ghi bảng so sánh. Với nguồn bị giới hạn truy cập, không lấy số liệu chi tiết chưa kiểm chứng.

## A. Kiến trúc, dữ liệu và phương pháp nghiên cứu

### P01

**Hamilton, Ying, Leskovec — Inductive Representation Learning on Large Graphs, NeurIPS 2017.** [Bản tác giả trên arXiv](https://arxiv.org/abs/1706.02216).

- Vai trò: cơ sở GraphSAGE và việc tạo embedding từ lân cận cho dữ liệu mới.
- Dùng ở phase 00/02; phân biệt inductive learning với khả năng tự tổng quát hóa tốt giữa mọi mạng. Bài gốc không chứng minh kết quả NIDS của repo.
- Mức đã xem: abstract và mô tả phương pháp trên trang bài.

### P02

**Lo và cộng sự — E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT, NOMS 2022; preprint 2021.** [Trang bài](https://arxiv.org/abs/2103.16329), [toàn văn HTML](https://arxiv.org/html/2103.16329).

- Vai trò: GNN khai thác đặc trưng cạnh cho NIDS; bài có phần thực nghiệm binary và multiclass.
- Dùng ở phase 00/02. Đối chiếu graph construction, aggregation, edge embedding và protocol trước khi nói repo tái lập bài gốc.
- Mức đã xem: abstract, cấu trúc và nội dung HTML liên quan đến mô hình/thực nghiệm.

### P03

**Sarhan, Layeghy, Portmann — Towards a Standard Feature Set for Network Intrusion Detection System Datasets, Mobile Networks and Applications, 2022; preprint 2021.** [arXiv](https://arxiv.org/abs/2101.11315), [DOI](https://doi.org/10.1007/s11036-021-01843-0).

- Vai trò: chuẩn hóa feature set NetFlow để thuận tiện so sánh các nguồn dữ liệu.
- Dùng ở phase 01; schema chung không bảo đảm cùng phân phối hoặc cùng cơ chế tính feature với Zeek.
- Mức đã xem: trang bài, abstract và trang dataset của tác giả.

### P04

**Luay và cộng sự — Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems, preprint 2025, cập nhật 2026.** [Trang bài](https://arxiv.org/abs/2503.04404), [HTML bản v3](https://arxiv.org/html/2503.04404v3).

- Vai trò: NetFlow V3 bổ sung flow start/end timestamps và thống kê inter-packet arrival time.
- Dùng để quyết định V2/V3 ở phase 00/01 và mô tả chính xác cửa sổ thời gian. Không cần thay V2 chỉ vì có phiên bản mới.
- Mức đã xem: abstract và phần temporal features/bảng feature trong HTML; phiên bản ngày 28/08/2026.

### P05

**Ying và cộng sự — GNNExplainer: Generating Explanations for Graph Neural Networks, NeurIPS 2019.** [arXiv](https://arxiv.org/abs/1903.03894), [toàn văn lưu tại PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7138248/).

- Vai trò: tìm subgraph và node feature liên quan tới dự đoán.
- Dùng ở phase 03; adapter edge-classification và direct edge_attr path của repo phải kiểm chứng riêng.
- Mức đã xem: abstract, phần phương pháp và đánh giá được truy cập trên trang toàn văn.

### P06

**Sundararajan, Taly, Yan — Axiomatic Attribution for Deep Networks, ICML 2017.** [Bản tác giả](https://arxiv.org/abs/1703.01365).

- Vai trò: Integrated Gradients để quy điểm ảnh hưởng cho input theo reference/baseline.
- Dùng ở phase 03 cho đặc trưng cạnh bằng adapter; không suy diễn attribution là bằng chứng nhân quả.
- Mức đã xem: abstract và mô tả phương pháp; đối chiếu thêm API Captum T04.

### P07

**Amara và cộng sự — GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks, Learning on Graphs / PMLR 198, 2022.** [Trang hội nghị và bài](https://proceedings.mlr.press/v198/amara22a.html).

- Vai trò: đánh giá lời giải thích bằng fidelity và nhu cầu sufficient/necessary, thay vì chỉ nhìn hình.
- Dùng ở phase 03/06. Nghiên cứu gốc tập trung node classification; protocol edge-level trong kế hoạch là phần điều chỉnh phải ghi rõ.
- Mức đã xem: abstract, phạm vi đánh giá và thông tin xuất bản từ PMLR.

### P08

**Luo và cộng sự — Parameterized Explainer for Graph Neural Network, NeurIPS 2020.** [arXiv](https://arxiv.org/abs/2011.04573), [trang phương pháp của tác giả](https://users.cs.fiu.edu/~dluo/papers/neurips20/Main.html).

- Vai trò: PGExplainer học mạng sinh giải thích cho nhiều instance.
- Dùng để so sánh lý thuyết với GNNExplainer; chỉ triển khai nếu đã xong phương án chính và xác minh hỗ trợ task/model.
- Mức đã xem: abstract và nội dung trang bài tác giả.

### P09

**Cui và cộng sự — Class-Balanced Loss Based on Effective Number of Samples, CVPR 2019.** [arXiv](https://arxiv.org/abs/1901.05555), [bài tại CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf).

- Vai trò: class reweighting theo effective number of samples; phù hợp làm thử nghiệm xử lý mất cân bằng.
- Dùng ở phase 02. Hiệu quả trên bài toán gốc không bảo đảm lớp hiếm NIDS sẽ học tốt; phải so với CE trên validation.
- Mức đã xem: abstract và mô tả công thức/phương pháp trên nguồn bài.

### P10

**Lin và cộng sự — Focal Loss for Dense Object Detection, ICCV 2017.** [Bản tác giả](https://arxiv.org/abs/1708.02002).

- Vai trò: giảm trọng số các mẫu dễ, tập trung lỗi khó; nguồn gốc thuộc object detection.
- Chỉ là thử nghiệm bổ sung ở phase 02, không coi là kết quả đã được chứng minh trên dữ liệu đồ án.
- Mức đã xem: abstract và mô tả mục tiêu của loss.

### P11

**Chen, Guestrin — XGBoost: A Scalable Tree Boosting System, KDD 2016.** [arXiv](https://arxiv.org/abs/1603.02754).

- Vai trò: cơ sở cho baseline tree boosting.
- Dùng ở phase 02/06; chọn tham số bằng validation với cùng split/profile như GNN.
- Mức đã xem: abstract và thông tin xuất bản.

### P12

**Arp và cộng sự — Dos and Don'ts of Machine Learning in Computer Security, USENIX Security 2022.** [Trang xuất bản chính thức](https://www.usenix.org/conference/usenixsecurity22/presentation/arp). Bản trình bày lại: **Pitfalls in Machine Learning for Computer Security**, CACM 2024, [bản tác giả](https://www.mlsec.tu-berlin.de/docs/2024-cacm.pdf).

- Vai trò: các lỗi phương pháp có thể làm kết quả ML bảo mật lạc quan hoặc khó dùng thực tế.
- Dùng xuyên suốt để audit split, leakage, model selection và cách diễn giải. Hai tên trên là hai bản của cùng tuyến công trình, không tính thành hai nghiên cứu độc lập.
- Mức đã xem: trang USENIX và phần abstract/nội dung phương pháp truy cập được của bản CACM.

### P13

**Cantone và cộng sự — Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study, IEEE Access, 2024.** [Bản lưu của cơ sở tác giả](https://iris.unicas.it/handle/11580/110014), [DOI](https://doi.org/10.1109/ACCESS.2024.3472907).

- Vai trò: bối cảnh đánh giá mô hình khi đổi dataset/mạng.
- Dùng ở phase 05/06 để giải thích vì sao phải đo lab độc lập; không lấy một con số từ công trình này làm kết quả của hệ thống hiện tại.
- Mức đã xem: metadata/abstract được công cụ tìm kiếm lập chỉ mục; truy cập toàn văn trực tiếp bị HTTP 403 trong phiên tra cứu. Cần đọc bản toàn văn có quyền truy cập trước khi trích chi tiết thực nghiệm.

### P14

**Caville và cộng sự — Anomal-E: A Self-Supervised Network Intrusion Detection System based on Graph Neural Networks, Knowledge-Based Systems 258, 110030; preprint 2022.** [Trang bài](https://arxiv.org/abs/2207.06819), [DOI](https://doi.org/10.1016/j.knosys.2022.110030).

- Vai trò: học biểu diễn cạnh tự giám sát cho phát hiện bất thường.
- Chỉ dùng phần nghiên cứu liên quan/hướng phát triển. Không coi là giải pháp bảo đảm cho domain shift hay nhiệm vụ phải thêm trong 8 tuần.
- Mức đã xem: abstract và metadata; chưa tái lập thực nghiệm.

### P15

**Ismaeli Taba, Zare, Rakhshani — Graph neural networks for network intrusion detection systems: a comprehensive survey, taxonomy, and empirical evaluation, Artificial Intelligence Review, 2026.** [Nguồn nhà xuất bản](https://link.springer.com/article/10.1007/s10462-026-11683-7).

- Vai trò: cập nhật phân loại phương pháp theo graph construction, feature encoding, learning và evaluation.
- Dùng ở phase 00/06 để hệ thống hóa tổng quan. Trang xuất bản ngày 07/09/2026 ghi đây là bản công bố sớm có thể còn biên tập.
- Mức đã xem: trang xuất bản, abstract và phạm vi khảo sát; không dùng các bảng full-text chưa đối chiếu để xếp hạng model.

## B. Tài liệu chính thức để triển khai

### T01

[University of Queensland — Machine Learning-Based NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/).

Nguồn dataset V2/V3, feature set và bảng nhãn. Dùng phase 00/01 để đối chiếu phiên bản; mapping cuối phải xác nhận lại từ CSV tải thực, không chỉ tên nút tải trên website.

### T02

[PyTorch Geometric — torch_geometric.explain](https://pytorch-geometric.readthedocs.io/en/stable/modules/explain.html).

API Explainer, task level, masks, kwargs và metrics. Dùng phase 03; chọn/pin phiên bản đã thử với model tùy chỉnh, vì API có cảnh báo đang phát triển.

### T03

[PyG — GNNExplainer API](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.explain.algorithm.GNNExplainer.html).

Đối chiếu tham số và giao diện explainer. Dùng cùng P05; không mặc định API cũ `explain_node` phù hợp bản cài mới.

### T04

[Captum — IntegratedGradients](https://captum.ai/api/integrated_gradients.html).

Input, baseline, target, additional arguments và convergence delta. Dùng phase 03 để viết forward adapter cho score của flow; Captum không tự biết flow ID của repo.

### T05

[Zeek — conn.log](https://docs.zeek.org/en/current/reference/logs/conn.html).

Ý nghĩa ts/uid, endpoints, byte/gói và thời lượng. Dùng phase 01/04/05; chú ý payload bytes khác IP bytes và event time khác thời điểm log sẵn sàng.

### T06

[ZeekControl — log rotation and archival](https://github.com/zeek/zeekctl/blob/master/doc/main.rst#log-rotation-and-archival).

Cơ chế current logs, rotation và archive. Dùng phase 04 để thiết kế reader theo file lifecycle, không giữ mãi file handle cũ.

### T07

[Tcpreplay — manual chính thức](https://tcpreplay.appneta.com/reference/man/tcpreplay/).

Các chế độ tốc độ/timing khi replay PCAP. Dùng phase 05; replay packet và đọc log offline là hai phép thử khác nhau.

### T08

[scikit-learn — classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).

Precision, recall, F1 và support theo lớp, macro/weighted. Dùng phase 02/06 để tránh diễn giải macro recall như recall riêng Attack.

### T09

[scikit-learn — confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

Thứ tự labels và normalize. Dùng phase 02/06, xuất cả bảng số đếm và hình normalized theo true class.

### T10

[scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

Điều kiện xác suất, số lớp và averaging cho multiclass AUC. Chỉ số bổ sung; xử lý lớp vắng có lý do thay vì bịa một số hợp lệ.

### T11

[NFStream — API chính thức](https://www.nfstream.org/docs/api).

Nguồn tham khảo extractor từ file/live và các thống kê flow. Chỉ cân nhắc khi cần ở phase 01; chưa có bằng chứng các feature này tương đương nProbe/NetFlow V2 của dữ liệu công khai.

### T12

[PyTorch — Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Seed, nguồn ngẫu nhiên và giới hạn tính tái lập giữa phiên bản/nền tảng. Dùng phase 02; ghi phiên bản cụ thể của runtime đã chạy.

### T13

[Google Colab — FAQ](https://research.google.com/colaboratory/faq.html).

Tài nguyên/giới hạn runtime thay đổi, không có GPU cố định được bảo đảm. Dùng để thiết kế shards/checkpoint/resume; lab mạng chạy ở môi trường riêng của người dùng.

### T14

[FastAPI — WebSockets](https://fastapi.tiangolo.com/advanced/websockets/).

Kết nối, gửi/nhận và xử lý disconnect. Dùng phase 04; reconnect, persistence, dedup và graph/XAI payload vẫn là phần phải triển khai trong ứng dụng.

## C. Thứ tự đọc để không mất thời gian

1. **Nắm hệ thống:** P02 → P03 → T01 → T05. Vẽ lại đường CSV/log → graph → edge prediction bằng lời của mình.
2. **Làm đúng dữ liệu:** P12 → P04 nếu cần timestamp → phase 01. Giải thích được vì sao fit scaler trước split chung là nguy cơ.
3. **Train đa lớp:** P01/P02 phần mô hình, P09 và T08–T10. Chưa cần đọc hết các mô hình mới trong survey.
4. **XAI:** P05 → T02/T03 → P06/T04 → P07. PGExplainer chỉ để hiểu lựa chọn khác.
5. **Hệ thống và lab:** T06/T07/T14 rồi protocol phase 04/05.
6. **Viết tổng quan và bàn luận:** P13/P15; P10/P14 là hướng bổ sung, không mở việc mới khi gần hạn.

Mỗi bài đưa vào báo cáo nên có ghi chú: bài toán, dataset/nhãn, graph construction, model, split, metric, phần kế thừa, khác biệt và giới hạn. Không so trực tiếp điểm số các bài dùng khác dataset/split/taxonomy rồi kết luận kiến trúc nào thắng.
