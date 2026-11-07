# Phase 06 — Tổng hợp thực nghiệm và báo cáo tốt nghiệp

**Trạng thái:** chưa triển khai. **Thời gian:** tuần 8; viết phần thiết kế và nhật ký bằng chứng từ các tuần trước.  
**Phụ thuộc:** phase 02–05 có artifact và kết quả thật; đọc [quy ước](quy_uoc_du_lieu.md).

## Mục tiêu và cách hiểu

Trả lời mô hình phân biệt được những lớp nào, hiệu quả so baseline ra sao, giải thích có đáng tin và hệ thống vận hành thế nào. Không chỉ tập hợp ảnh chụp màn hình.

## Các câu hỏi nghiên cứu vừa đủ phạm vi

| Câu hỏi | Thí nghiệm và bằng chứng |
|---|---|
| RQ1: E-GraphSAGE binary/multiclass so với baseline thế nào? | Cùng dataset/profile/split, metric từng lớp và confusion matrix |
| RQ2: Ngữ cảnh đồ thị/nhóm feature đóng góp thế nào? | Ablation nhỏ ở phase 02, so chất lượng và chi phí |
| RQ3: Lời giải thích phản ánh hành vi model tới mức nào? | Perturbation, random control, sparsity, stability và case đúng/sai |
| RQ4: Pipeline đáp ứng bốn kịch bản lab và tải mục tiêu ra sao? | Ground truth, unique-flow metrics, false alarm, latency/throughput và lỗi vận hành |

RQ2 trên đây là câu hỏi của **đồ án mới**, không cùng tên gọi với RQ2 đánh giá chéo trong PDF thực tập. Nếu viết nối báo cáo cũ, đổi mã hoặc tên rõ để không nhầm. Cross-dataset là bổ sung, không phải đường bắt buộc phải giải quyết domain shift.

## Danh mục bảng/hình phải có

1. Dataset version, nguồn, feature profile, taxonomy, support và số flow/graph theo split.
2. Binary: RF/XGB/E-GraphSAGE; Accuracy, macro P/R/F1 và confusion matrix.
3. Multiclass: cùng ba model; macro/weighted P/R/F1, từng lớp/support, raw và normalized confusion matrix.
4. Ablation: đóng góp graph/feature so với cấu hình đầy đủ; cùng protocol và giới hạn ngân sách.
5. XAI: fidelity/perturbation, random control, sparsity/runtime; case đúng/sai có flow ID.
6. Lab: ít nhất 4 scenario và B0; detection, sai loại, false positive, nhãn coverage và số phiên.
7. Vận hành: latency từng giai đoạn p50/p95, throughput, CPU/RAM/queue/drop và soak test.
8. So sánh cấu hình nghiên cứu đầy đủ và model/profile triển khai nếu khác; không trộn chung thành một điểm số.

Mỗi bảng phải truy được tới `run_id`, cấu hình, checkpoint và dữ liệu prediction. Số liệu từ PDF ghi là kết quả thực tập, không đổi tiêu đề thành kết quả multiclass mới.

## Cách diễn giải chỉ số

- Accuracy có thể cao do lớp Benign lớn; phải đọc cùng macro F1 và support lớp hiếm.
- Precision-macro/Recall-macro là trung bình theo lớp, không phải trực tiếp tỷ lệ cảnh báo Attack đúng hoặc tỷ lệ bắt hết Attack. Muốn nói hai tỷ lệ đó, tính riêng binary Attack-positive.
- F1-macro là trung bình F1 từng lớp; không lấy macro Precision/Recall rồi tính một F1 thay thế.
- Weighted F1 giúp phản ánh quy mô lớp nhưng có thể che lớp hiếm. Lớp support bằng 0 phải được nêu; giữ tập label cố định để confusion matrix không đổi shape theo seed.
- ROC-AUC multiclass nếu có phải ghi OvR/OvO và average. Khi điều kiện không thỏa, dùng N/A với lý do, không điền 0 hoặc 1.
- Độ tin cậy của XAI và độ đúng của classifier là hai vấn đề; explanation rõ không làm một prediction sai trở thành đúng.

## Độ tin cậy và giới hạn kết luận

Với nhiều seed, báo từng seed và mean/std, không chỉ báo lần tốt nhất. Với nhiều phiên lab, báo phân bố theo phiên; không coi hàng nghìn flow từ một cuộc tấn công là hàng nghìn phép thử độc lập.

Không bắt buộc McNemar. Nếu bổ sung, dùng dự đoán trên cùng flow test và giải thích giới hạn độc lập do flow cùng phiên; không dùng một p-value để thay toàn bộ phân tích thực nghiệm.

Nếu benchmark tốt nhưng lab yếu, báo rõ feature shift, môi trường và protocol. Công trình cross-dataset hỗ trợ lý do cần đánh giá này, không chứng minh trước rằng hệ thống hiện tại sẽ tổng quát hóa tốt.

## Bố cục báo cáo đề xuất

| Phần | Nội dung |
|---|---|
| Mở đầu | Bài toán, mục tiêu, yêu cầu, phạm vi và phần kế thừa thực tập |
| Chương 1 | IDS/NIDS, flow/graph, GNN, XAI, streaming, mất cân bằng; nghiên cứu liên quan có đối chiếu |
| Chương 2 | Dataset/split, feature contract, nhãn binary/multiclass, E-GraphSAGE và explainer |
| Chương 3 | Kiến trúc hệ thống, Colab/artifact, reader/window/queue, API, dashboard, graph và lab |
| Chương 4 | Protocol, số liệu, ablation, XAI, bốn scenario, vận hành và threats to validity |
| Kết luận | Mục tiêu đạt/chưa đạt, giới hạn và hướng phát triển có căn cứ |
| Phụ lục | Cấu hình, lệnh, mapping, đường dẫn/checksum và hướng dẫn tái lập |

Tận dụng nội dung LaTeX tại `Nhom_A05/` sau khi kiểm tra mẫu báo cáo tốt nghiệp cần dùng. Không tự xóa báo cáo PDF hoặc ghi đè bản thực tập; dùng version/bản sao phù hợp khi bắt đầu viết báo cáo mới.

## Đầu ra cuối cùng

- Run index liệt kê các thí nghiệm chính và artifact; bảng/hình sinh từ file kết quả, không hard-code tay điểm số mới.
- Báo cáo tốt nghiệp, danh mục tham khảo và bảng đối chiếu yêu cầu.
- Hướng dẫn cài/chạy/replay/demo, model bundle deploy và dữ liệu mẫu được phép chia sẻ.
- Video/ảnh demo dự phòng và một trace replay đã thử, cùng kịch bản thuyết trình.
- Nhật ký hạn chế: lớp ít mẫu, dataset/source order, parity chưa hoàn hảo, domain shift, tài nguyên và phần XAI chưa bao phủ.

## Nghiệm thu

- [ ] Toàn bộ yêu cầu bắt buộc trong kế hoạch tổng có bằng chứng hoặc được ghi rõ chưa đạt, không giấu trong “hướng phát triển”.
- [ ] Người khác reload bundle và chạy mẫu offline/replay theo hướng dẫn được.
- [ ] Không có bảng multiclass dùng lại chỉ số binary; không có latency end-to-end chỉ đo forward.
- [ ] Có ít nhất 4 hồ sơ scenario, nhãn coverage, bảng P/R/F1 và confusion matrix phù hợp.
- [ ] Citation hỗ trợ đúng nhận định; phân biệt thuật toán gốc và phần điều chỉnh trong đồ án.
- [ ] Có một lần diễn tập đầy đủ từ khởi động dịch vụ đến xem cảnh báo/graph/XAI; có phương án replay dự phòng.

## Giao việc cho AI

> Thực hiện phase 06 từ artifact thật: kiểm tra run index, sinh bảng/hình bằng prediction và log, viết nội dung báo cáo theo RQ và yêu cầu. Kiểm tra cách tính metric, số flow duy nhất, split và nhãn. Không bịa kết quả bị thiếu; nếu phần bắt buộc chưa đạt, nêu việc cần bổ sung cụ thể. Giữ kết quả thực tập và đồ án mới phân biệt rõ.

## Nguồn đọc

[P07 đánh giá giải thích](tai_lieu_tham_khao.md#p07), [P12 đánh giá ML bảo mật](tai_lieu_tham_khao.md#p12), [P13 cross-dataset](tai_lieu_tham_khao.md#p13), [P15 survey cập nhật](tai_lieu_tham_khao.md#p15), [T08–T10 metrics](tai_lieu_tham_khao.md#t08).
