# Phase 02 — Binary, multiclass và huấn luyện trên Colab

**Trạng thái:** mã nguồn và test hoàn thành (07/11/2026), chưa train đầy đủ — xem [báo cáo](../../results/graduation/phase_reports/phase_02.md). **Thời gian:** tuần 2–3.  
**Phụ thuộc:** dữ liệu/split audit đạt phase 01; đọc [quy ước artifact](quy_uoc_du_lieu.md).

## Mục tiêu và cách hiểu

Một mô hình multiclass xuất `C` điểm số cho mỗi flow và chọn một lớp, gồm Benign và các loại tấn công. Đây không phải multilabel. Chỉ tăng output lên C chưa đủ nếu graph vẫn dùng `Label=0/1`, metric vẫn đọc cột xác suất 1 hoặc loader vẫn dựng mô hình hai lớp.

## Ma trận thí nghiệm tối thiểu

| Dữ liệu / profile | Task | Model |
|---|---|---|
| Dataset chính, profile nghiên cứu | Binary | RF, XGBoost, E-GraphSAGE |
| Cùng dataset/profile/split | Multiclass | RF, XGBoost, E-GraphSAGE |
| Lab, profile Zeek đã xác minh | Multiclass triển khai | E-GraphSAGE; baseline nếu còn ngân sách |
| Dataset thứ hai | Binary và multiclass | Chỉ thêm khi các hàng bắt buộc đã chạy ổn |

Hàng lab dùng dữ liệu chuẩn bị sớm và hoàn thiện ở phase 05. Có thể tích hợp dashboard bằng model nhỏ/provisional trước; mô hình dùng cho đánh giá cuối phải có train/val/test độc lập và artifact khóa.

## Các bước triển khai

1. Đưa task, label mapping, `num_classes`, input dimensions và `benign_class_id` vào config/model bundle. Lấy số lớp từ mapping đã khóa, không từ `max(label)` của một batch; kiểm tra lớp vắng train và class weights hữu hạn. Giữ binary chạy được bằng task riêng.
2. Train lại RF/XGBoost trên cùng flow train/val/test với GNN. XGBoost hỗ trợ mục tiêu multiclass trong mã hiện có, nhưng phải kiểm tra `predict`/`predict_proba` và thứ tự classes sau reload.
3. Giữ E-GraphSAGE hai lớp truyền thông điệp làm model chính. Đối chiếu điểm khác với bài báo: biểu diễn IP:port, feature node, phép cộng message, classifier ghép edge_attr; mô tả là biến thể ứng dụng.
4. Xử lý mất cân bằng: CE không trọng số làm mốc, so với Class-Balanced CE đang có; weighted CE là phương án đơn giản. Chỉ tính class weights trên train và tránh đếm flow overlap nhiều lần như các mẫu độc lập.
5. Không SMOTE đồ thị trong đường bắt buộc. Focal Loss là thử nghiệm giới hạn nếu lớp hiếm học quá yếu; chỉ chọn cấu hình theo val.
6. Cố định seed Python/NumPy/Torch/DataLoader và lưu RNG/config. Cùng seed không bảo đảm giống bit trên mọi GPU/phiên bản; ghi môi trường theo tài liệu PyTorch.
7. Tính validation theo flow duy nhất với chính sách đã chốt. Chọn checkpoint bằng F1-macro; lưu best và last riêng, best gồm đủ metadata để dựng đúng model.
8. Mở rộng metrics: Accuracy, Precision/Recall/F1 từng lớp, macro và weighted, support; confusion matrix với thứ tự lớp cố định. AUC-ROC OvR là bổ sung, trả N/A có lý do khi không đủ lớp hoặc xác suất không hợp lệ.
9. Lưu dự đoán test theo flow ID. Chỉ dùng test sau khi chốt cấu hình; chạy test nhiều seed không được dùng để chọn seed đẹp nhất.
10. Ablation nhỏ: so model đầy đủ với classifier chỉ dùng edge features/MLP hoặc GNN bỏ một nhóm feature. Mục đích kiểm tra đóng góp ngữ cảnh; không cần train nhiều kiến trúc mới.

## Cách chạy Colab trong ngân sách hai tháng

- Notebook điều phối, logic chính nằm trong `src/`; mở rộng `notebooks/00_colab_bootstrap.ipynb` hoặc notebook train mới khi cần.
- Kiểm tra Torch/CUDA hiện có trước khi cài phụ thuộc; không cài lại Torch một cách máy móc làm lệch môi trường.
- Stage shard cần dùng lên ổ tạm của runtime, thay vì đọc ngẫu nhiên liên tục từ Drive. Dữ liệu gốc và checkpoint bền vững vẫn nằm trên Drive.
- Smoke test vài chục graph → pilot cỡ vừa giữ các lớp phù hợp → full hoặc subset công bố rõ. Chọn subset theo session/cửa sổ, không xáo flow rời rạc để giả lập ngữ cảnh.
- Thử batch 8/16/32 theo VRAM thực tế. Nếu OOM, giảm batch/giải phóng shard, không giảm lớp hay bỏ test mà không ghi nhận.
- Pilot 3–5 epoch để đo; sau đó xác định giới hạn epoch/time budget. Giữ early stopping và resume gồm optimizer, scheduler, epoch, best score và RNG.
- Cấu hình chính cố gắng chạy 3 seed (ví dụ 42, 43, 44). Nếu chỉ đủ một lần full, công bố single-run; chạy nhiều seed trên subset có tên riêng, không trộn thành độ lệch chuẩn full dataset.

## File cần sửa

| File hiện có | Nội dung |
|---|---|
| `src/models/train_gnn.py`, `gnn_config.py` | Config task/profile, seed, resume, weights và validation theo flow |
| `src/models/graphsage.py`, `sage_layer.py` | Tái sử dụng kiến trúc; kiểm tra output C và khả năng giải thích |
| `src/models/train_baseline.py`, `baselines.py` | Multiclass, sample/class weights và lưu class order |
| `src/models/metrics.py`, `evaluate_test.py` | Per-class metrics, flow IDs, C-class proba, confusion matrix |
| `src/models/evaluate_cross_dataset.py` | Không mở rộng multiclass máy móc; giữ raw target và mapping nếu có thí nghiệm bổ sung |
| `src/api/infer.py` | Chuẩn bị loader đọc model bundle; tích hợp live đầy đủ ở phase 04 |

## Đầu ra và nghiệm thu

- [ ] Cả 3 model chạy được binary và multiclass trên dataset chính, artifact không ghi đè nhau.
- [ ] Reload checkpoint cho output khớp model trước save trên cùng batch trong tolerance đã ghi.
- [ ] Báo cáo có per-class P/R/F1/support và confusion matrix raw + normalized.
- [ ] Có audit cùng tập test và khử lặp flow; có phân tích lớp hiếm thay vì chỉ Accuracy.
- [ ] Có log tài nguyên, seed, môi trường, config và lệnh/notebook đã chạy.
- [ ] Resume thực sự tiếp tục trạng thái train, không chỉ nạp weights rồi bắt đầu lại.
- [ ] Có kết quả mốc binary mới theo protocol đồ án; số liệu PDF chỉ dùng làm tham chiếu lịch sử.

Nghiệm thu kỹ thuật không yêu cầu F1-macro vượt 0,95. Nếu model sụp về một lớp, phải kiểm tra mapping, weights, batch support, graph split và input distribution trước khi tiếp tục phase đánh giá cuối.

## Giao việc cho AI

> Thực hiện phase 02 bằng split/profile đã nghiệm thu. Bổ sung cấu hình binary/multiclass xuyên suốt training, checkpoint, loader và metrics; chuẩn bị notebook Colab chạy lại được và resume được. Train trên tài nguyên thực có sẵn, xuất báo cáo từng lớp và các trường hợp model yếu. Không tự lấy số liệu bài báo/PDF làm kết quả chạy mới và không tune trên test.

## Nguồn đọc

[P01 GraphSAGE](tai_lieu_tham_khao.md#p01), [P02 E-GraphSAGE](tai_lieu_tham_khao.md#p02), [P09 Class-Balanced Loss](tai_lieu_tham_khao.md#p09), [P10 Focal Loss](tai_lieu_tham_khao.md#p10), [P11 XGBoost](tai_lieu_tham_khao.md#p11), [T08–T10 metrics](tai_lieu_tham_khao.md#t08), [T12 PyTorch](tai_lieu_tham_khao.md#t12), [T13 Colab](tai_lieu_tham_khao.md#t13).
