# Giai đoạn 2 — Xây dựng biểu diễn đồ thị và quyết định về môi trường lab

**Thời gian dự kiến:** Tuần 2
**Trạng thái:** Đang thực hiện

## Mục tiêu
- Xây dựng module Graph Builder (bảng → đồ thị), dùng chung cho cả hai bộ dữ liệu, kèm unit test.
- Chốt quyết định có thiết lập môi trường lab VMware hay không (xem khung quyết định tại `docs/00_research_plan.md` mục 4.3).
- Nếu có lab: thiết lập mạng ảo, Zeek/Suricata, thực hiện các kịch bản tấn công.

## Quyết định môi trường lab
- **Phương án chọn:** **A — có lab** (2026-07-12)
- **Lý do:** Nhóm xác nhận có nhu cầu trình diễn tấn công real-time trước hội đồng. Theo khung quyết định ở `docs/00_research_plan.md` mục 4.3, đây là điều kiện bắt buộc phải thiết lập lab (2 bộ dữ liệu công khai chỉ ở dạng tĩnh, không demo trực tiếp được).
- **Hệ quả:** phạm vi thí nghiệm đầy đủ TN1, TN2, TN6; cần thêm: dựng VMware, cấu hình mạng host-only cô lập, cài Zeek + Suricata (ET Open Rules), thực hiện 5 kịch bản tấn công, thu thập + gán nhãn dữ liệu lab. Tuân thủ ranh giới bắt buộc tại mục 4.4 (mọi tấn công chỉ trong mạng ảo cục bộ, không được thực hiện trên hạ tầng cloud).
- **Thứ tự triển khai:** dời việc thiết lập VMware/lab lại **sau Giai đoạn 3** (huấn luyện xong, kiểm thử mô hình ổn định) — ưu tiên làm xong pipeline khoa học cốt lõi (Graph Builder → train GCN/GAT → đánh giá TN1/TN2) trước, tránh vừa lo hạ tầng lab vừa lo mô hình cùng lúc. Giai đoạn 2 hiện tại chỉ tập trung nhánh Graph Builder.

## Đầu ra kiểm chứng được
- [ ] Graph Builder chạy đúng trên cả hai bộ dữ liệu, unit test pass.
- [ ] (Nếu Phương án A) Bộ dữ liệu lab thu thập đầy đủ nhãn, đã kiểm tra độ chính xác gán nhãn.

## Nhật ký cập nhật

## Vấn đề phát sinh / quyết định
