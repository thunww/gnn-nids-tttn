# Phase 03 — Giải thích quyết định của GNN

**Trạng thái:** chưa triển khai. **Thời gian:** tuần 4; dùng spike đã thử ở phase 00.  
**Phụ thuộc:** model bundle từ phase 02, ID và graph contract từ phase 01.

## Mục tiêu và cách hiểu

Giải thích **vì sao mô hình dự đoán một flow cụ thể**, bằng phần ngữ cảnh mạng và đặc trưng liên quan. Đây là giải thích hành vi mô hình, không phải chứng minh nguyên nhân thật của cuộc tấn công.

| Loại giải thích | Câu hỏi | Phương án chính |
|---|---|---|
| Cấu trúc | Những kết nối/ngữ cảnh nào ảnh hưởng tới flow được chọn? | GNNExplainer trên output edge |
| Đặc trưng | Byte, gói, thời lượng hoặc feature nào ảnh hưởng tới score? | Integrated Gradients bằng Captum, có adapter edge_attr |
| Bản trình bày | Người dùng xem kết quả thế nào? | Subgraph, feature importance, giá trị đo và thông tin prediction |

Không cần triển khai nhiều explainer nặng. PGExplainer chỉ đọc để so sánh; không mặc định thư viện hỗ trợ trực tiếp cấu hình edge classifier tùy chỉnh.

## Điểm kỹ thuật cần giải quyết

`GraphSAGEEdgeClassifier` lấy cả embedding hai endpoint **và edge_attr trực tiếp** vào classifier. Vì vậy:

- Mask cạnh trong message passing chưa chắc làm mất tín hiệu edge_attr ở classifier.
- Mask node feature không tự động giải thích từng chiều edge_attr.
- Node feature hiện còn chứa trung bình feature cạnh; thay edge_attr nhưng giữ nguyên x chỉ giải thích một phần đường phụ thuộc.

Các nhận xét trên suy ra từ mã nguồn hiện tại. Cần kiểm tra bằng perturbation trước khi kết luận explainer hoạt động đúng.

## Các bước triển khai

1. Đóng băng model, `eval()` nhưng giữ gradient cho explainer. Chọn `target_flow_id`, ánh xạ chính xác sang output edge index.
2. Thử cấu hình PyG `Explainer`: `mode='multiclass_classification'`, `task_level='edge'`, `return_type='raw'` cho model trả logits; `explanation_type='model'` để giải thích dự đoán của mô hình. Chốt phiên bản PyG đã smoke test, không chép API cũ một cách máy móc.
3. Bắt đầu bằng toàn bộ graph nhỏ của một cửa sổ. Khi giới hạn subgraph hai hop quanh cả hai endpoint, kiểm tra score trước/sau crop; nếu khác đáng kể phải báo giải thích cho graph đã crop, không coi là y hệt dự đoán ban đầu.
4. Giữ edge target để classifier vẫn có output. Khi bỏ cạnh ngữ cảnh, giữ/remap index và feature cùng flow ID. Kiểm thử nhiều flow song song cùng endpoint.
5. Tạo IG adapter cho score của lớp được chọn. Có thể bắt đầu với attribution theo `edge_attr` khi x cố định, nhưng phải ghi **conditional attribution**. Muốn giải thích toàn bộ đường feature, recompute phần node aggregation từ edge_attr bằng phép toán khả vi hoặc giải thích đồng thời các input và ghi rõ sự phụ thuộc.
6. Chọn reference theo dữ liệu train, ví dụ mean sau chuẩn hóa; thử thêm một reference hợp lý để đo nhạy. Không coi các feature tương quan hoặc dữ liệu nội suy là một mạng thực có thể tồn tại.
7. Xuất mask liên tục và top-k (ví dụ k=5/10/20), giữ ID thật. Đồ thị tô đậm các cạnh/nút liên quan; biểu đồ feature hiển thị cả tên, giá trị và loại attribution.
8. Lưu explanation theo prediction snapshot để phase 04 gọi lại. Chạy on-demand/asynchronous, không giải thích mọi flow trong luồng suy luận chính.

## Đánh giá chất lượng giải thích

Chọn trước khoảng 20–50 flow theo ngân sách: Benign, các lớp tấn công có support, dự đoán đúng và sai. Nếu không có mẫu thuộc nhóm nào, báo rõ; không chỉ chọn cảnh báo tự tin nhất.

| Kiểm tra | Cách làm |
|---|---|
| Necessity | Bỏ top-k **cạnh ngữ cảnh**, giữ target, đo độ giảm score lớp đang giải thích |
| Sufficiency | Chỉ giữ target và ngữ cảnh top-k, đo khả năng giữ score/prediction |
| Đối chứng | So với bỏ/giữ k cạnh ngẫu nhiên, lặp với seed cố định |
| Sparsity | Số cạnh/nút giữ lại chia cho kích thước graph giải thích |
| Stability | Lặp seed/reference, so overlap top-k hoặc tương quan scores |
| Chi phí | Thời gian p50/p95 và RAM trên máy chạy giải thích |

Ghi rõ perturbation là mask trong forward với x cố định hay dựng lại graph/feature từ flow còn lại; hai phép đo không tương đương. Nếu dựng lại graph, recompute feature node và giữ ánh xạ target. Không báo “explanation accuracy” khi chưa có ground-truth explanation; nhãn Attack không phải nhãn cho cạnh nào quan trọng.

Nếu mask ngữ cảnh ít tác động nhưng IG feature lớn, có thể model dựa nhiều vào edge features. Đó là phát hiện cần báo, không phải lý do tô một subgraph bất kỳ cho đẹp. GraphFramEx cung cấp cơ sở đánh giá fidelity nhưng nghiên cứu gốc tập trung node classification; cách áp dụng cho edge classifier ở đây là thiết kế cần kiểm chứng.

## File và đầu ra dự kiến

- Xem `src/models/graphsage.py`, `sage_layer.py`, `src/graph/node_features.py`.
- Tạo module dự kiến `src/explain/` chứa adapter, giải thích theo flow và đánh giá perturbation; chỉ sửa model khi cần giao diện mask, giữ nguyên dự đoán khi không bật mask.
- `results/graduation/<run_id>/explanations/`: JSON mask/attribution, hình subgraph, feature importance và bảng fidelity/runtime.
- Báo cáo phase giải thích rõ lớp nào đang được giải thích, baseline/reference, các đường input đã bao phủ và giới hạn.

## Nghiệm thu

- [ ] Giải thích đúng output flow ID trước/sau crop hoặc masking.
- [ ] Có kiểm tra gradient/mask tác động tới score trên fixture chủ động phụ thuộc feature hoặc ngữ cảnh.
- [ ] Có giải thích cấu trúc và đặc trưng phù hợp với phương pháp thực tế, không gọi edge_mask là importance 39 feature.
- [ ] Có đối chứng random và đo sparsity/runtime; ghi rõ feature frozen/recomputed.
- [ ] Có case dự đoán đúng/sai và Benign; không tuyên bố quan hệ nhân quả từ XAI.
- [ ] Mô hình giữ output cũ khi không bật giải thích; không retrain mô hình bằng test để làm lời giải thích đẹp hơn.

## Giao việc cho AI

> Thực hiện phase 03 cho edge classifier hiện có. Xác minh adapter PyG/Captum bằng một flow cụ thể và kiểm tra direct edge_attr path cùng node aggregates. Xuất giải thích có ID, phương pháp, reference và đối chứng perturbation. Nếu GNNExplainer không hỗ trợ ổn định, báo lỗi thực, thử phương pháp cấu trúc thay thế có kiểm chứng trong cùng ngân sách; không đánh dấu hoàn thành chỉ vì sinh được hình.

## Nguồn đọc

[P05 GNNExplainer](tai_lieu_tham_khao.md#p05), [P06 Integrated Gradients](tai_lieu_tham_khao.md#p06), [P07 GraphFramEx](tai_lieu_tham_khao.md#p07), [P08 PGExplainer](tai_lieu_tham_khao.md#p08), [T02 PyG](tai_lieu_tham_khao.md#t02), [T03 GNNExplainer API](tai_lieu_tham_khao.md#t03), [T04 Captum](tai_lieu_tham_khao.md#t04).
