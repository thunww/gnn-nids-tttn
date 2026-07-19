# GNN-NIDS — Nghiên cứu và ứng dụng mạng nơ-ron đồ thị phát hiện xâm nhập mạng

Thực tập tốt nghiệp — D22CQAT01-N, PTIT. Sinh viên: Trần Gia Thân (N22DCAT050), Trần Xuân Đông (N22DCAT018).
Tài liệu định hướng đầy đủ: [`docs/00_research_plan.md`](docs/00_research_plan.md).

## Cấu trúc thư mục

```
docs/            Tài liệu định hướng + nhật ký tiến độ từng giai đoạn (docs/phases/)
data/raw/        Dữ liệu gốc, không chỉnh sửa tay
data/processed/  Dữ liệu sau ETL / đồ thị đã build (Parquet)
src/             Mã nguồn: etl, graph, models, training, evaluation, explainability, api
notebooks/       Bản sao lưu notebook Colab
configs/         Cấu hình experiment (yaml/json)
results/         Bảng/biểu đồ kết quả đã tổng hợp, dùng cho báo cáo
Nhom_A05/        Báo cáo LaTeX (giữ nguyên cấu trúc gốc, không di chuyển/đổi tên)
```

## Quy trình cập nhật tiến độ

Mỗi giai đoạn (1-6) có một file trong `docs/phases/`. Khi hoàn thành hoặc có tiến triển ở một giai đoạn, cập nhật trực tiếp vào file tương ứng (nhật ký + đầu ra kiểm chứng được), không dồn vào cuối.

## Lưu ý dữ liệu

Xem cảnh báo về schema bộ UNSW-NB15 tại [`data/raw/README.md`](data/raw/README.md) — cần bộ NF-UNSW-NB15-v2 (NetFlow V2) chứ không phải bản gốc trước khi triển khai Graph Builder.
