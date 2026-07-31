# Quy trình xử lý dữ liệu (Pipeline)

## Bước 1 — ETL (`src/etl/`)

1. **Đọc CSV thô** (`data/raw/<bộ>/`).
2. **Làm sạch** (`clean.py`) — loại giá trị lỗi.
3. **Chia tập** (`split.py`) — `stratified_split`, tỷ lệ **70% train / 15% val / 15% test**, xáo trộn ngẫu nhiên theo hàng (dùng cho baseline — coi mỗi dòng độc lập).
4. **Chuẩn hoá** (`scale.py`):
   - Cắt outlier theo ngưỡng phân vị 99% tính từ **train** (`upper_bound`).
   - `StandardScaler` fit trên train, áp dụng lại (không fit lại) cho val/test.
   - Lưu `scaler.joblib` + `upper_bound.joblib` (mỗi bộ dữ liệu 1 cặp riêng).
5. **Xuất thêm bản `full_chronological.parquet`** — giữ nguyên thứ tự dòng gốc (không xáo trộn), áp đúng scaler/upper_bound đã fit ở bước 4 — dành riêng cho Graph Builder (để cửa sổ trượt cắt ra là lát cắt thời gian thực, không phải mẫu ngẫu nhiên rút từ khắp nơi).

**Cột nhãn:** `Label` (0 = Benign, 1 = Attack) — cột gốc có sẵn trong dữ liệu, không cần mã hoá thêm.

## Bước 2 — Graph Builder (`src/graph/`)

**Biểu diễn đồ thị:** node = cặp (địa chỉ IP, cổng); cạnh = 1 luồng mạng (flow), có hướng.

**Cửa sổ trượt (sliding window):** cắt `full_chronological.parquet` thành nhiều đồ thị con, mỗi đồ thị gồm N luồng liên tiếp theo thời gian, độ chồng lấp 50%:

| Bộ dữ liệu | Số luồng/cửa sổ (`WINDOW_SIZE`) |
|---|---|
| `nf-cse-cic-ids2018-v2` | 2.000 |
| `nf-unsw-nb15-v2` | 500 |

**Đặc trưng cạnh (`edge_attr`, 39 chiều):** toàn bộ đặc trưng luồng mạng đã chuẩn hoá (byte, số gói tin, thời lượng, cờ TCP, throughput...).

**Đặc trưng node (`x`, 43 chiều):**
- 4 chiều **cấu trúc** (tính từ vị trí node trong đồ thị con): bậc vào, bậc ra, PageRank, hệ số phân cụm.
- 39 chiều **tổng hợp** = trung bình cộng của toàn bộ 39 đặc trưng cạnh (cả chiều vào lẫn ra) của các cạnh nối tới node đó — giúp node "biết" nội dung luồng mạng thật đi qua nó, không chỉ biết vị trí cấu trúc thuần túy.

**Nhãn cạnh (`y`):** lấy trực tiếp từ cột `Label` (nhị phân) của luồng tương ứng.

**Chia tập:** chia **danh sách đồ thị con** (không phải dòng) theo 70/15/15 — `train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`.

## Bước 3 — Chia shard (chỉ với tập train lớn)

Để tránh tràn RAM khi train trên Google Colab (free tier ~12-13GB), tập train được cắt thành nhiều file nhỏ (`train_graphs_shard{i}.pt`, mỗi shard ≤ 2.200 đồ thị) — mỗi epoch chỉ nạp 1 shard vào RAM tại một thời điểm rồi giải phóng, thay vì nạp toàn bộ 1 lần. Bộ dữ liệu nhỏ hơn 2.200 đồ thị không cần chia (tự động dùng nguyên file).

## Tóm tắt số liệu thực tế

| Bộ dữ liệu | Số đồ thị train | Số đồ thị val | Số đồ thị test |
|---|---|---|---|
| `nf-cse-cic-ids2018-v2` | 13.224 | 2.834 | 2.834 |
| `nf-unsw-nb15-v2` | 6.692 | 1.434 | 1.434 |
