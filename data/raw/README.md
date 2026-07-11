# data/raw

Dữ liệu gốc, không chỉnh sửa tay. Hai nguồn hiện có:

- `nf-cse-cic-ids2018-v2/` — NF-CSE-CIC-IDS2018-v2 (schema NetFlow V2, UQ/Sarhan et al.), khớp với bộ dữ liệu mô tả trong `Nhom_A05/chapters/chap_02_methodology.tex`.
- `unsw-nb15/` — **CẢNH BÁO:** đây là bộ UNSW-NB15 **bản gốc** (trích xuất bằng Argus/Bro, ~45 cột: `dur, proto, service, spkts,...`), **không phải** NF-UNSW-NB15-v2 (schema NetFlow V2) mà kế hoạch nghiên cứu yêu cầu. Cần tải đúng bản NF-UNSW-NB15-v2 từ UQ trước khi chạy ETL/Graph Builder, nếu không hai bộ dữ liệu sẽ không cùng schema đặc trưng và không thể dùng chung pipeline như mô tả trong `docs/00_research_plan.md` mục 4.1.
