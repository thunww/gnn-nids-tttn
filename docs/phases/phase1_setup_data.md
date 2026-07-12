# Giai đoạn 1 — Thiết lập nền tảng và chuẩn bị dữ liệu

**Thời gian dự kiến:** Tuần 1
**Trạng thái:** Hoàn thành (ETL) — phần lý thuyết/báo cáo xem mục Nhật ký

## Mục tiêu
- Thiết lập môi trường phát triển (VS Code + Google Colab + Drive mount).
- Khảo sát lý thuyết nền tảng (GCN, GAT, E-GraphSAGE).
- ETL cho NF-CSE-CIC-IDS2018-v2 và NF-UNSW-NB15-v2, lưu Parquet trên Drive.
- Dựng khung báo cáo LaTeX.

## Đầu ra kiểm chứng được
- [x] Môi trường chạy không lỗi phụ thuộc (local: Python 3.12 + pandas/pyarrow/sklearn; Colab: kết nối GitHub + Drive qua `notebooks/00_colab_bootstrap.ipynb`).
- [x] Hai bộ dữ liệu đã làm sạch, chuẩn hoá, chia 70/15/15, lưu Parquet — chạy tại local (`data/processed/`), chưa đồng bộ lên Drive.
- [ ] Tài liệu tóm tắt lý thuyết các bài báo nền tảng.
- [ ] Bản nháp Mở đầu + Chương 1.

## Nhật ký cập nhật
- 2026-07-12: Phát hiện bản NF-UNSW-NB15-v2 tải từ Kaggle thiếu cột `IPV4_SRC_ADDR`/`IPV4_DST_ADDR` (bắt buộc cho Graph Builder) — tải lại đúng bản gốc từ UQ eSpace (https://espace.library.uq.edu.au/view/UQ:ffbb0c1), xác nhận đủ 45 cột.
- 2026-07-12: Viết pipeline ETL (`src/etl/`: `load.py`, `clean.py`, `split.py`, `scale.py`, `run_etl.py`). Gồm: downcast kiểu dữ liệu để giảm RAM (~49%), loại trùng lặp/thiếu, mã hoá `Attack` bằng `LabelEncoder`, chia stratified 70/15/15, clip outlier (percentile 99% theo train) rồi `StandardScaler`.
- 2026-07-12: Chạy ETL đầy đủ tại local (Colab bản free bị crash hết RAM khi chạy bộ CSE-CIC-IDS2018 ~19 triệu dòng). Kết quả: `nf-cse-cic-ids2018-v2` còn 18.893.151 dòng sau làm sạch (mất 557 dòng trùng/thiếu), `nf-unsw-nb15-v2` còn đúng 2.390.275 dòng (khớp số liệu gốc, không mất dòng). Cả 2 đều chia đúng tỷ lệ 70.0/15.0/15.0%.

## Vấn đề phát sinh / quyết định
- Cột `Attack_encoded` được `LabelEncoder` đánh số **độc lập theo từng bộ dữ liệu** (vd "Benign" = 0 ở CSE-CIC-IDS2018 nhưng = 2 ở UNSW-NB15-v2) — khi làm Thí nghiệm 2 (cross-dataset) ở Giai đoạn 4, **không so sánh trực tiếp số `Attack_encoded` giữa 2 bộ**, chỉ cột `Label` (nhị phân) là so sánh chéo được.
- Dữ liệu đã xử lý (`data/processed/`, ~780MB) hiện chỉ có ở local, **chưa upload lên Drive** — cần làm trước khi bắt đầu Giai đoạn 3 (huấn luyện GNN trên Colab).
