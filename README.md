# GNN-NIDS — Phát hiện xâm nhập mạng bằng E-GraphSAGE

Thực tập tốt nghiệp — D22CQAT01-N, PTIT. Sinh viên: Trần Gia Thân (N22DCAT050), Trần Xuân Đông (N22DCAT018).

Phân loại luồng mạng **Benign/Attack** bằng E-GraphSAGE, so sánh với Random Forest và XGBoost trên NF-CSE-CIC-IDS2018-v2 và NF-UNSW-NB15-v2; demo phát hiện brute-force FTP/SSH từ log Zeek.

Xem [mục tiêu, flow đầy đủ, kết quả và thứ tự chạy](docs/README.md). Tài liệu được rút gọn theo phần đã thực hiện trong [báo cáo PDF](NguyenHoangThanh_A05_BCCK.pdf).

Phần phát triển đồ án tốt nghiệp: [kế hoạch 8 tuần, các phase và tài liệu tham khảo](docs/do_an/README.md).

| Thư mục | Nội dung |
|---|---|
| `src/etl/` | Làm sạch, chia bảng và chuẩn hóa dữ liệu |
| `src/graph/` | Dựng đồ thị IP:port theo cửa sổ trượt |
| `src/models/` | Huấn luyện E-GraphSAGE, baseline và đánh giá |
| `src/api/` | Đọc log Zeek, suy luận và dashboard thời gian thực |
| `data/` | Dữ liệu thô, dữ liệu xử lý và mô hình đã lưu |
| `scripts/` | Script hỗ trợ lab và tạo hình báo cáo |
| `notebooks/` | Notebook chạy trên Colab |
| `report_figures/` | Hình minh họa báo cáo |
| `Nhom_A05/` | Nguồn báo cáo LaTeX |
