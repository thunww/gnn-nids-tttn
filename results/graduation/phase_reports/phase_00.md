# Báo cáo Phase 00 — Khảo sát nền GNN-NIDS

Ngày kiểm tra: **24/10/2026**. **Kết luận:** đủ nền dữ liệu và mã nguồn để bắt đầu phase 01; chưa nên train full theo protocol cũ. P0 đã hoàn thành phần kiểm tra local, có notebook để kiểm tra Colab. **Phần máy lab được bỏ qua theo yêu cầu người dùng.**

Chưa đánh dấu nghiệm thu GPU Colab: phiên này chạy trên Windows CPU. Không có số liệu phát hiện đa lớp mới hoặc kết quả lab mới trong báo cáo này.

## 1. Bằng chứng và cách chạy lại

- [JSON audit đầy đủ](../phase_00/20261024T015924_148481Z/audit.json): môi trường, inventory, nhãn, projection split, checkpoint và các probe.
- [Phân bố 25 lớp theo nguồn/split](phase_00_class_counts.csv).
- [Rà soát 39 đặc trưng converter](phase_00_feature_review.csv).
- [Cấu hình quyết định cho phase tiếp theo](../../../docs/do_an/phase_00_quyet_dinh.json).
- [Script kiểm tra](../../../scripts/phase00_audit.py), [notebook Colab](../../../notebooks/01_phase00_audit.ipynb), [cách chạy Colab](../../../docs/do_an/phase_00_huong_dan_colab.md).

Các lệnh đã thực chạy trên local:

```powershell
python -X utf8 scripts/phase00_audit.py --scan-raw
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest src/graph/tests/test_graph_builder.py -q -p no:cacheprovider
```

Script tạo thư mục bằng chứng mới mỗi lần, không sửa CSV/Parquet hoặc model thực tập. Inventory dùng filesystem trực tiếp, gồm file bị Git ignore. Tổng dung lượng file dưới `data/` là **31.56 GB thập phân (~29.39 GiB)**, có tính cả ZIP sao lưu; không phải dung lượng cần nạp vào RAM cùng lúc.

## 2. Dữ liệu thực tế đã kiểm tra

| Nội dung | CSE-CIC-IDS2018-v2 | UNSW-NB15-v2 |
|---|---:|---:|
| CSV raw tồn tại | Có, ~3.22 GB | Có, ~0.44 GB |
| Số dòng raw | 18.893.708 | 2.390.275 |
| Số cột CSV thực tế | 45 | 45 |
| Số đặc trưng model | 39 | 39 |
| Số lớp Attack gốc | 15 | 10 |
| ID Benign trong mapping đa lớp cũ | 0 | **2** |
| Số dòng bảng sau clean đã lưu | 18.893.151 | 2.390.275 |
| Bảng baseline train | 13.225.205 | 1.673.192 |
| Bảng baseline val | 2.833.973 | 358.541 |
| Bảng baseline test | 2.833.973 | 358.542 |

45 cột raw gồm 4 định danh + 39 đặc trưng + `Label` + `Attack`. Parquet có thêm `Attack_encoded` thành 46 cột. Không coi mô tả “46 cột raw” trong tài liệu cũ là schema đã được xác nhận.

Đã đọc **toàn bộ hai cột Label/Attack trong hai CSV**, và đếm nhãn toàn bộ các bảng Parquet. Số dòng sau clean ở đây lấy từ artifact hiện có, **chưa chạy lại clean full độc lập**. CSE-CIC có 557 dòng ít hơn raw, đều thuộc Infilteration theo chênh lệch nhãn; chưa xác định độc lập đó là dòng trùng hay dòng thiếu. Không suy ra raw hoàn toàn sạch chỉ từ chênh lệch này.

Kiểm tra 4.096 dòng đầu mỗi CSV: không thấy ô thiếu, số NaN/Inf hoặc dòng trùng trong mẫu. Đây không phải kiểm tra mọi trường trên toàn bộ dataset. Lượt quét nhãn không phát hiện nhãn thiếu.

CSV không có timestamp tuyệt đối; `FLOW_DURATION_MILLISECONDS` là thời lượng. Giữ V2 và gọi cửa sổ là **cửa sổ số lượng theo thứ tự nguồn**, chưa gọi là thời gian thực lịch sử đã xác minh.

## 3. Phân bố lớp và tính khả thi multiclass

Tất cả lớp gốc hiện có trong các bảng baseline train/val/test. Chiếu cách chia graph hiện tại lên bảng giữ thứ tự nguồn cũng thấy đủ lớp, nhưng đây là phép tính từ code và bảng nguồn, **không phải kiểm tra toàn vẹn các graph archive đã lưu**.

| Ví dụ lớp cần chú ý | Raw | Bảng train | Val | Test |
|---|---:|---:|---:|---:|
| CSE-CIC: SQL Injection | 432 | 303 | 65 | 64 |
| CSE-CIC: Brute Force -XSS | 927 | 649 | 139 | 139 |
| UNSW: Worms | 164 | 115 | 25 | 24 |

Không thiếu điều kiện kỹ thuật để bắt đầu multiclass, nhưng lớp hiếm có ít mẫu test; cần báo support, per-class metrics và giới hạn. Đặc biệt, **Benign của UNSW là ID 2**, nên không thể lấy “mọi class ID khác 0 là Attack” hoặc dùng cứng cột proba số 1.

Quyết định: giữ taxonomy gốc riêng cho từng dataset, binary là task riêng. Không ghép các class ID của hai bộ hoặc hứa mô hình CSE-CIC sẽ nhận đúng lớp Reconnaissance không có trong mapping của nó.

## 4. Mã nguồn: điểm dùng lại và điểm phải sửa

| Ưu tiên | Phát hiện | Bằng chứng | Hành động |
|---|---|---|---|
| Cao | Split baseline và GNN không cùng tập flow | `src/etl/split.py` chia ngẫu nhiên; `src/graph/run_graph_builder.py` chia block cửa sổ | Phase 01 tạo một manifest dùng chung |
| Cao | Scaler fit trên train của nhánh bảng rồi áp cả nhánh graph | `src/etl/run_etl.py:41`, `:57`; scaler lưu số samples bằng baseline train | Chia tập chung trước, fit từ flow train duy nhất |
| Cao | Chưa có flow ID gốc trong graph để audit/khử lặp | `src/graph/build_graph.py` chỉ lưu x/edge_index/edge_attr/y | Lưu ID và metadata cạnh xuyên pipeline |
| Cao | Binary hard-code nhiều nơi | `src/models/config.py:7`, `src/graph/build_graph.py:22`, `src/api/infer.py:36`, `:60` | Task/mapping/C-class proba trong config và model bundle |
| Cao | Converter không đo đủ 39 feature | `src/api/zeek_convert.py`; bảng feature review kèm theo | Không gọi là parity đã đạt; train đúng profile live sau khi xác minh |
| Trung bình | Chưa seed đầy đủ và chưa resume trạng thái optimizer/scheduler/RNG | `src/models/train_gnn.py:216`, `:240`, `:246` | Phase 02 tách best/last và lưu đủ trạng thái |
| Trung bình | Validation đếm mọi edge, gồm flow overlap | `src/models/train_gnn.py:122` | Metric theo flow duy nhất, chính sách thống nhất |
| Trung bình | Đảo scaler không khôi phục raw đã clip | `src/models/evaluate_cross_dataset.py:52`, `:85` | Nếu làm cross-dataset mới, giữ raw test trước clip |
| Trung bình | Build graph giữ toàn bộ danh sách trong RAM rồi mới chia | `src/graph/run_graph_builder.py:46` | Ghi shard tăng dần ở phase 01 |
| Trung bình | Reader giữ file handle, chưa reopen khi rotation | `src/api/realtime_server.py:61` | Phase 04 bổ sung lifecycle/queue/persistence |
| Trung bình | Dashboard cộng số prediction của cửa sổ overlap | `src/api/realtime_server.py:390` | Tách unique flow, prediction event và unique alert |

Chưa đo tỷ lệ giao flow gốc giữa scaler-train và graph-test trên toàn dữ liệu vì artifact chưa có ID/manifest nguồn. Nhận định về nguy cơ leakage dựa trên luồng code và hai cách chia khác nhau; không gán một tỷ lệ đo giả cho dataset thật.

Phép chiếu graph hiện tại cho CSE-CIC ra **13.224/2.833/2.833 graph**, UNSW **6.692/1.434/1.432 graph**. Các khoảng flow của graph split không giao nhau với cấu hình cửa sổ/overlap hiện tại; điều này chưa giải quyết việc scaler được fit trên split của nhánh khác.

## 5. Kiểm tra thực thi local

| Kiểm tra | Kết quả và phạm vi |
|---|---|
| Môi trường | Python 3.12.6; Torch 2.13.0+cpu; PyG 2.8.0; pandas 2.2.3; sklearn 1.6.1; PyArrow 20.0.0 |
| GPU | Không có CUDA ở runtime local |
| RAM máy | ~15.73 GiB; available thay đổi, không đủ để nạp tùy ý graph/RF nhiều GB |
| Scaler/clip | Hai bộ nạp được; 39 feature đúng thứ tự converter, mean/scale hữu hạn, clip index khớp |
| Checkpoint thực tập | Hai model strict-load thành công; forward graph dựng từ 128 dòng bảng train cho output hữu hạn `[128,2]` |
| Graph unit tests sẵn có | **3 passed** |
| Smoke train synthetic | 24 graph, 3 output class, 2 epoch; save/reload sai khác tối đa 0 |
| Pilot trên dữ liệu thật | 1.536 flow đầu bảng CSE-CIC; 24 graph × 64 flow, 15 output class, 2 epoch; save/reload sai khác tối đa 0 |
| GNNExplainer | Chạy được `task_level=edge`, logits/multiclass trên fixture tổng hợp; edge mask `[8]`, node mask `[5,43]` |
| Converter/buffer/reader | Fixture TSV parse/convert được; 39 feature; streaming 4-flow/50% overlap khớp offline; reader bỏ dòng cũ và nhận dòng mới |
| Chưa cài / chưa kiểm tra | Captum, FastAPI trên môi trường local; không có lệnh Zeek/zeekctl/nvidia-smi trong PATH |

Pilot dữ liệu thật dùng legacy scaler và target thay riêng trong harness sang `Attack_encoded`, **không sửa graph builder production**. Prefix có 11/15 lớp, không có tập test; không coi là đã train xong multiclass. Loss giảm trong pilot chỉ chứng minh bước tối ưu chạy được.

Ở lượt audit cuối, hai epoch pilot mất khoảng **0.100 s và 0.112 s**, sampled peak RSS của cả process khoảng **566.6 MiB**, checkpoint thử ~513 KB. Đây là graph 64-flow nhỏ, CPU đã warm-up; **không ngoại suy thời gian/RAM train full với cửa sổ 2.000 flow**, không dùng làm latency hệ thống.

Không nạp RF CSE-CIC ~3.10 GB hoặc graph archives nhiều GB; inventory chỉ xác nhận chúng tồn tại và kích thước. Chưa kiểm tra đủ khả năng nạp/tương thích của các RF/XGBoost và từng graph shard.

## 6. Kết quả spike XAI và giới hạn

Đã chạy GNNExplainer 20 bước trên một output edge của mô hình synthetic 3 lớp. Khi đổi message mask từ toàn 1 sang toàn 0, logit target thay đổi tối đa khoảng **0.0627**; gradient theo edge_attr hữu hạn và khác 0. Điều này xác nhận các đường kỹ thuật có tác động lên score trong fixture.

Fixture kiểm soát riêng đặt model chỉ phụ thuộc một chiều edge_attr đi trực tiếp vào classifier: đổi feature target từ 2 về 0 làm score đổi từ 2 về 0. Do đó **chỉ mask message passing chưa bao phủ toàn bộ đường quyết định**. Phase 03 cần xử lý direct edge_attr và phần node feature tổng hợp phụ thuộc edge_attr.

Chưa chạy IG/Captum, fidelity/necessity/sufficiency đầy đủ hoặc đánh giá lời giải thích của checkpoint thật. Spike này giảm rủi ro tích hợp PyG nhưng không chứng minh explanation đáng tin cho tấn công.

API đã đối chiếu với [tài liệu PyG 2.8.0](https://pytorch-geometric.readthedocs.io/en/2.8.0/modules/explain.html). Ý nghĩa trường log dùng làm tham khảo từ [Zeek conn.log](https://docs.zeek.org/en/current/reference/logs/conn.html); chưa kiểm tra parity exporter bằng PCAP trong P0.

## 7. Quyết định phạm vi sau khảo sát

1. **Giữ V2**, vì raw/artifact đã có. CSE-CIC là bộ chính, UNSW là bộ kiểm chứng thứ hai theo ngân sách Colab; không mở thêm kiến trúc GNN trong đường bắt buộc.
2. **Phase 01 sửa protocol trước train:** split chung, scaler train-only, flow IDs, mapping/task và shard tăng dần. Giữ raw test cho thí nghiệm mới.
3. **Giữ E-GraphSAGE hiện có**; kiến trúc đã cho phép C output. RF/XGBoost dùng lại được nhưng training/evaluation phải đọc task/schema đúng.
4. **Profile nghiên cứu:** 39 feature/43 node dimensions đã xác nhận schema. Profile live ban đầu chỉ có 6 feature ứng viên; chưa được kích hoạt hoặc tuyên bố tương đương NetFlow.
5. **XAI:** ưu tiên GNNExplainer cho cấu trúc; IG cho feature là phần cần triển khai/kiểm tra riêng ở phase 03.
6. **Lab P0:** bỏ qua theo yêu cầu; bốn scenario vẫn là mục tiêu phase 05, không coi kiểm tra fixture là bằng chứng lab.
7. **Không train full P0**, không ghi đè checkpoint cũ và không hứa F1. Notebook Colab để kiểm tra GPU thực, ngân sách full chỉ chốt sau pilot đúng window/batch ở phase 02.

## 8. Trạng thái nghiệm thu

- [x] Có dataset thật, mapping và phân bố lớp toàn bộ CSV/Parquet đã kiểm tra.
- [x] Xác định flow là edge, output giữ số lượng cạnh và các giả định binary cần sửa.
- [x] Chốt phần bắt buộc, nguồn dữ liệu, phạm vi V2 và hướng split chung cho phase 01.
- [x] Kiểm tra local train → save → reload → predict trên mẫu nhỏ thật; checkpoint legacy đọc được.
- [x] Spike kỹ thuật XAI và converter/window/reader trên fixture.
- [ ] Train/reload/pilot trên **GPU Colab thực**: đã chuẩn bị notebook, chưa thực chạy trong phiên này.
- [ ] Thu PCAP/kiểm tra Zeek và parity trên lab: **bỏ qua ở P0 theo yêu cầu người dùng**, không chờ để làm phase 01.

**Bước tiếp theo:** phase 01 theo cấu hình đã lưu; chưa cần đưa toàn bộ dataset vào Colab để train dài ngay.
