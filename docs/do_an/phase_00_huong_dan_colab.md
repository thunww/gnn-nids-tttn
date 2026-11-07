# P0 — Chạy phần kiểm tra còn lại trên Colab

Phần local đã có [báo cáo khảo sát](../../results/graduation/phase_reports/phase_00.md). Người dùng đã yêu cầu **bỏ qua kiểm tra máy lab ở P0**; tài liệu này chỉ hướng dẫn Colab.

## File dùng để chạy

- Mở [notebook P0](../../notebooks/01_phase00_audit.ipynb) bằng Colab.
- Upload file `results/graduation/phase_00/colab_phase00_kit.zip` khi notebook yêu cầu. Gói này chứa script khảo sát và các file Python trong `src/` tại thời điểm chuẩn bị, không chứa CSV/checkpoint hoặc thông tin đăng nhập.
- Mount Drive bằng hộp thoại Colab. Notebook dùng đường dẫn cũ `/content/drive/MyDrive/GNN-NIDS_TTTN/data`; sửa `DATA_ROOT` nếu thư mục thực tế khác.

Không cần push các thay đổi local lên GitHub để chạy gói kiểm tra này. Đây là bộ kiểm tra tách biệt, không ghi đè kết quả thực tập.

## Kiểm tra nào sẽ chạy?

1. In thông tin GPU/Torch/PyG của runtime thực tế.
2. Kiểm tra kỹ thuật train → save → reload và GNNExplainer bằng dữ liệu tổng hợp.
3. Nếu Drive có `processed/nf-cse-cic-ids2018-v2/full_chronological.parquet` cùng mapping, chạy pilot 24 graph × 64 flow, 2 epoch với nhãn đa lớp thật trên phần đầu bảng.
4. Lưu JSON và checkpoint thử nghiệm vào một thư mục có ID mới dưới `MyDrive/GNN-NIDS_TTTN/graduation/phase_00/`.

Pilot không có test độc lập, chưa dùng split/scaler đã sửa, và prefix chỉ có một phần lớp. Không lấy loss hoặc thời gian của pilot này làm hiệu quả/chi phí train toàn bộ dataset. Muốn chốt ngân sách full training, sau phase 01 phải pilot với cửa sổ 2.000/500 flow và batch thực dùng.

## Cách đọc kết quả

- `device=cuda` và `cuda_peak_allocated_bytes` có giá trị: đã thực chạy probe bằng GPU.
- `device=cpu`: vẫn kiểm tra code được nhưng chưa có số đo GPU Colab.
- `real_multiclass_pilot.status=not_run`: chưa thấy bảng/mapping ở Drive; kiểm tra lại `DATA_ROOT`, không coi đây là kiểm tra dữ liệu thật thành công.
- `sections.*.status=failed`: đọc trường error/traceback, giữ nguyên JSON để chẩn đoán.

Có thể bật `SCAN_FULL_LABELS=True` để quét lại toàn bộ nhãn và kiểm tra hai checkpoint thực tập trên Drive, nhưng không cần lặp lại nếu dữ liệu giống bản local đã quét. Notebook mặc định dùng chế độ kiểm tra giới hạn để tránh đọc lại nhiều GB qua Drive.

Khi có JSON Colab, đưa file kết quả vào `results/graduation/phase_00/` hoặc cung cấp nội dung để cập nhật báo cáo. Chưa đánh dấu mục nghiệm thu Colab hoàn thành trước khi có bằng chứng thực chạy.
