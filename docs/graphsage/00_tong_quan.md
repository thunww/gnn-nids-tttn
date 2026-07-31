# Tổng quan — GNN-NIDS bằng E-GraphSAGE

**Đây là tài liệu tham khảo DUY NHẤT nên dùng để viết báo cáo.** Các file `docs/decisions.md`, `docs/phases/*.md` là **nhật ký làm việc** (ghi theo thời gian, giữ lại cả các hướng đã thử rồi bỏ như GCN/GAT/đa lớp — không xoá để truy vết) — **không dùng trực tiếp** để lấy số liệu viết báo cáo, dễ nhầm giữa thông tin cũ/mới. Toàn bộ nội dung trong thư mục `docs/graphsage/` này đã được lọc sạch, chỉ phản ánh đúng **trạng thái cuối cùng đang dùng thật**.

## Bài toán

**Phân loại nhị phân** luồng mạng: **Benign** (bình thường) hay **Attack** (bị tấn công) — dựa trên cột `Label` gốc có sẵn trong 2 bộ dữ liệu (không phải đa lớp theo loại tấn công cụ thể).

## Bộ dữ liệu

Hai bộ dữ liệu công khai, chuẩn hoá theo định dạng **NetFlow V2** (Sarhan và cộng sự) — cùng schema (46 cột), cho phép so sánh trực tiếp không cần ánh xạ đặc trưng thủ công:

| Tên | Số dòng | Ghi chú |
|---|---|---|
| `nf-cse-cic-ids2018-v2` | ~18,9 triệu | Mạng doanh nghiệp mô phỏng (CSE-CIC-IDS2018) |
| `nf-unsw-nb15-v2` | ~2,39 triệu | Testbed UNSW-NB15 |

## Kiến trúc mô hình: E-GraphSAGE

**Chỉ dùng 1 kiến trúc GNN duy nhất: E-GraphSAGE** (Lo, Layeghy, Sarhan, Gallagher & Portmann, 2021 — *"E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT"*, arXiv:2103.16329). Đây là biến thể của GraphSAGE gốc, có thêm khả năng dùng **đặc trưng cạnh** (byte, thời lượng, cờ TCP...) trong bước lan truyền thông điệp — phù hợp trực tiếp với bài toán này vì phần lớn thông tin luồng mạng thật nằm ở cạnh (luồng), không phải ở node (địa chỉ IP:port).

**So sánh với 2 mô hình đối chứng (baseline):** Random Forest, XGBoost.

## Câu hỏi nghiên cứu

- **RQ1 (Thí nghiệm 1 — within-dataset):** trong cùng 1 bộ dữ liệu, E-GraphSAGE có hiệu quả phân loại tốt hơn, tương đương, hay kém hơn Random Forest/XGBoost?
- **RQ2 (Thí nghiệm 2 — cross-dataset):** model huấn luyện trên 1 bộ dữ liệu có tổng quát hoá tốt sang bộ dữ liệu khác (môi trường mạng khác hoàn toàn) không?

## Mục lục tài liệu trong thư mục này

1. `00_tong_quan.md` — file này
2. `01_du_lieu_pipeline.md` — quy trình xử lý dữ liệu, từ CSV thô đến đồ thị
3. `02_kien_truc_mo_hinh.md` — công thức toán học + siêu tham số E-GraphSAGE
4. `03_ket_qua.md` — bảng kết quả đầy đủ TN1 + TN2 (số liệu chính thức, dùng viết Chương 4)
5. `04_cong_viec_con_lai.md` — việc chưa làm (GNNExplainer, kiểm định thống kê, mô phỏng real-time)
6. `05_demo_realtime_setup.md` — tài liệu tự chứa, đầy đủ kỹ thuật để dựng demo real-time (dùng để giao cho agent/người khác thực hiện trên server, không cần đọc thêm file nào khác)
