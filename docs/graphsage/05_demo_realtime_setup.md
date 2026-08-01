# Dựng demo real-time — tài liệu tự chứa cho agent/người thực hiện trên server

**Đọc file này là đủ để bắt tay vào làm, không cần xem lại các file `docs/decisions.md`/`docs/phases/*.md` (đó là nhật ký làm việc, không phải hướng dẫn thực thi).**

## 1. Bối cảnh dự án (tóm tắt)

Đồ án: hệ thống phát hiện xâm nhập mạng (NIDS) dùng mạng nơ-ron đồ thị (GNN). Bài toán: **phân loại nhị phân** 1 luồng mạng (flow) là **Benign** (bình thường) hay **Attack** (bị tấn công). Kiến trúc dùng: **E-GraphSAGE** (chi tiết công thức xem `docs/graphsage/02_kien_truc_mo_hinh.md`). Đã huấn luyện xong và có kết quả đầy đủ trên 2 bộ dữ liệu công khai (xem `docs/graphsage/03_ket_qua.md`) — **KHÔNG cần train lại gì**, việc còn lại là dựng demo phát hiện tấn công **real-time** để trình chiếu trước hội đồng bảo vệ đồ án.

## 2. Đã có sẵn trong `src/` — không cần viết lại

```
src/etl/          -- doc CSV tho (NetFlow V2) -> lam sach -> chia train/val/test -> chuan hoa (StandardScaler)
  config.py       -- ten cot: IDENTIFIER_COLS, LABEL_COL="Label", ATTACK_COL, DATASETS
  load.py         -- doc CSV
  clean.py        -- lam sach + ma hoa Attack_encoded (khong dung cho model nhi phan, chi de tham khao)
  split.py        -- chia 70/15/15
  scale.py        -- fit_scale() / apply_scale() -- StandardScaler + clip outlier 99th percentile
  run_etl.py       -- chay toan bo, luu scaler.joblib + upper_bound.joblib (can cho demo, xem muc 4)

src/graph/        -- bien du lieu bang -> do thi PyTorch Geometric
  nodes.py        -- gan ID cho tung dinh (cap IP:port)
  edges.py        -- xay dung canh (edge_index) + dac trung canh (edge_attr, 39 chieu)
  node_features.py -- tinh dac trung node (43 chieu = 4 cau truc + 39 tong hop tu canh)
  build_graph.py  -- ghep thanh 1 object Data hoan chinh, nhan y = cot Label
  windowing.py    -- cat du lieu thanh cua so truot (WINDOW_SIZE flow/cua so)
  run_graph_builder.py -- chay toan bo Graph Builder

src/models/
  gnn_config.py   -- SIEU THAM SO THAT (xem muc 5 duoi day de tai dung dung)
  sage_layer.py   -- EGraphSAGEConv (MessagePassing tuy chinh, PyG SAGEConv khong ho tro edge_dim)
  graphsage.py    -- GraphSAGEEdgeClassifier (kien truc chinh, DANG DUNG)
  gcn.py, gat.py  -- KHONG DUNG NUA (lich su cu, da bo -- xem docs/decisions.md)
  baselines.py    -- Random Forest + XGBoost
  metrics.py      -- compute_full_metrics() -- Accuracy/Precision/Recall/F1-macro/AUC-ROC/MCC
  train_gnn.py    -- vong lap train GraphSAGE (KHONG can chay lai, model da train xong)
  train_baseline.py -- train RF/XGBoost (KHONG can chay lai)
  evaluate_test.py -- **DUNG LAM KHUNG THAM KHAO cho demo** -- nap model da train, chay suy luan,
                       tinh metrics. Xem ham evaluate_baseline(), evaluate_graphsage() de biet cach
                       nap dung model + gia usthiet ve shape dau vao.
  evaluate_cross_dataset.py -- danh gia cheo bo du lieu (khong lien quan demo real-time)
```

## 3. Model đã train xong — vị trí file cần copy sang server

```
data/processed/nf-cse-cic-ids2018-v2/models/random_forest.joblib
data/processed/nf-cse-cic-ids2018-v2/models/xgboost.joblib
data/processed/nf-cse-cic-ids2018-v2/models/graphsage_best.pt
data/processed/nf-cse-cic-ids2018-v2/scaler.joblib
data/processed/nf-cse-cic-ids2018-v2/upper_bound.joblib

data/processed/nf-unsw-nb15-v2/models/random_forest.joblib
data/processed/nf-unsw-nb15-v2/models/xgboost.joblib
data/processed/nf-unsw-nb15-v2/models/graphsage_best.pt
data/processed/nf-unsw-nb15-v2/scaler.joblib
data/processed/nf-unsw-nb15-v2/upper_bound.joblib
```

**Demo real-time nên dùng model của bộ `nf-cse-cic-ids2018-v2`** (kết quả tốt hơn, F1-macro 0.988 so với 0.978 — xem `03_ket_qua.md`), trừ khi có lý do khác.

**Quyết định (2026-07-31):** demo real-time trên server giám sát (`192.168.207.200`, 3.3GB RAM) **chỉ dùng `graphsage_best.pt`** — không dùng `random_forest.joblib`/`xgboost.joblib` làm baseline trực tiếp trong demo. Lý do: `random_forest.joblib` nặng ~3GB trên đĩa, nạp vào RAM bị kernel kill (OOM) trên máy chỉ có 3.3GB RAM. `graphsage_best.pt` rất nhẹ (505KB, 125,186 tham số) và đã kiểm chứng nạp/chạy OK trên server này.

## 4. Kế hoạch tổng thể — 7 phase (A→G)

| Phase | Việc | Trạng thái |
|---|---|---|
| A | Dựng 3 máy ảo VMware (tấn công/nạn nhân/giám sát), mạng host-only cô lập khỏi Internet, bật Promiscuous Mode máy giám sát, cài Zeek (Suricata **tạm bỏ** — xem ghi chú dưới) | **XONG kỹ thuật** (2026-07-31) — còn nợ: đổi NAT→Host-only trước Phase C |
| B | Chốt nội dung cụ thể 5 kịch bản tấn công + công cụ dùng | **XONG kế hoạch** (2026-07-31) — xem mục 4.3, còn nợ: chạy `scripts/victim_setup.sh` trên victim |
| C | Chạy Zeek + Suricata song song, thực hiện 5 kịch bản, ghi lại chính xác mốc thời gian từng kịch bản | **Chưa làm** — Người dùng |
| D | Viết code chuyển log Zeek → đúng 39 cột đặc trưng NetFlow V2 (khớp schema đã train) | **XONG** — `src/api/zeek_convert.py`, chi tiết mục 6 |
| E | Script suy luận offline (kiểm chứng đúng trước khi làm live) + script so sánh với Suricata | **XONG phần GraphSAGE** — `src/api/window_buffer.py` + `infer.py`, chi tiết mục 7. So sánh Suricata: tạm hoãn |
| F | API real-time (FastAPI) — tail log Zeek liên tục, gom cửa sổ, gọi model, hiển thị kết quả | **XONG** — `src/api/realtime_server.py` + `scripts/background_traffic_generator.py`, chi tiết mục 8 |
| G | Cập nhật docs với kết quả TN6 | **Chưa làm** — cần chạy demo thật (Phase B/C) trước mới có số liệu |

**Thứ tự bắt buộc:** D → E (kiểm chứng offline đúng trước) → F (mới làm live). Không nhảy thẳng vào F vì nếu logic chuyển đổi đặc trưng (D) sai, demo live sẽ cho kết quả vô nghĩa mà không biết ngay.

### 4.1. Sơ đồ mạng thực tế (Phase A — đã dựng)

| Vai trò | Địa chỉ IP |
|---|---|
| Máy nạn nhân (victim) | `192.168.207.199` |
| Máy giám sát (monitor) — chạy Zeek + pipeline real-time (Phase D/E/F), Suricata tạm bỏ | `192.168.207.200` |
| Máy tấn công (attacker, Kali Linux) | `192.168.207.194` (DHCP — có thể đổi, xem lưu ý dưới) |

Card mạng đang giám sát trên máy 200: **`ens33`**.

**Lưu ý IP attacker:** `192.168.207.194` hiện là IP do DHCP cấp (`dynamic`), **có thể đổi khi Kali khởi động lại hoặc lease hết hạn** — trước hôm demo thật nên đặt IP tĩnh cho Kali (hoặc ít nhất kiểm tra lại `ip a` mỗi lần vào máy) để không bị lệch với các script/ghi chú đã chuẩn bị sẵn.

**⚠️ Cảnh báo chưa đúng yêu cầu isolation (2026-07-31):** cả 3 máy VMware hiện đang để **NAT** (dải `192.168.207.0/24` là do NAT/VMnet8 của VMware tự cấp trên máy host này), **KHÔNG phải Host-only** như checklist Phase A yêu cầu ("mạng host-only, cô lập hoàn toàn khỏi Internet"). NAT vẫn cho phép các máy ảo ra Internet qua host. Điều này không chặn việc dựng/test hạ tầng (promiscuous mode vẫn bật được, Zeek vẫn hoạt động), nhưng **PHẢI đổi sang Host-only trước khi chạy 5 kịch bản tấn công thật** (Phase C) để tránh traffic tấn công/DoS vô tình lọt ra Internet thật — đổi trong VMware: VM Settings → Network Adapter → chọn "Host-only" thay vì "NAT" cho cả 3 máy, rồi cấu hình lại IP tĩnh nếu cần.

**Promiscuous mode: ĐÃ XÁC NHẬN HOẠT ĐỘNG ĐÚNG (2026-07-31), không cần sửa `.vmx`.** Kiểm chứng bằng `sudo grep 192.168.207.194 /opt/zeek/logs/current/conn.log` (toàn file, không giới hạn `tail -N`) — thấy được đúng dòng ICMP giữa attacker (194) ↔ victim (199), 5 gói mỗi chiều khớp với `ping -c 5` đã chạy, hoàn toàn không liên quan tới máy monitor (200). Vậy chỉ với `sudo ip link set ens33 promisc on` trong guest (đã làm ở mục 4.2) là đủ trên cấu hình NAT/VMnet8 hiện tại — không cần chỉnh `ethernet0.noPromisc` trong file `.vmx`.

*(Ghi chú: lần kiểm tra đầu tiên "tưởng" không thấy traffic là do dùng `tail -20 | grep` — traffic nền (DNS/DHCP) đẩy dòng cần tìm ra khỏi cửa sổ 20 dòng cuối, không phải do capture thật sự thất bại. Khi cần kiểm tra 1 IP cụ thể, luôn `grep` toàn file, không giới hạn `tail -N` trước.)*

### 4.2. Vận hành Zeek trên máy giám sát (192.168.207.200)

**Đã cài xong** (Zeek 8.2.1, cài qua repo OBS `security:zeek` vì Ubuntu 26.04 không có sẵn trong repo mặc định). Zeek cài ở `/opt/zeek`, log ghi vào `/opt/zeek/logs/current/` (thuộc `root:zeek`, cần `sudo` hoặc user phải nằm trong group `zeek` mới đọc được).

**Mỗi lần vào lại server (sau khi tắt máy/khởi động lại VM), làm theo thứ tự sau:**

```bash
# 1. Bật lại promiscuous mode cho card mạng (KHÔNG tự động bật lại sau reboot)
sudo ip link set ens33 promisc on
ip -d link show ens33 | grep -i promisc     # xác nhận thấy chữ PROMISC

# 2. Kiểm tra Zeek đã chạy chưa
sudo /opt/zeek/bin/zeekctl status

# 3. Nếu status KHÔNG phải "running" -> khởi động lại
sudo /opt/zeek/bin/zeekctl deploy
# (deploy = check config + install + restart, dùng lệnh này an toàn nhất mỗi lần có nghi ngờ)

# 4. Xem log đang sinh ra
sudo tail -f /opt/zeek/logs/current/conn.log
```

**Các lệnh vận hành khác khi cần:**

```bash
sudo /opt/zeek/bin/zeekctl stop        # dừng hẳn Zeek
sudo /opt/zeek/bin/zeekctl start       # chỉ khởi động (không re-check config như deploy)
sudo /opt/zeek/bin/zeekctl restart     # dừng rồi chạy lại
sudo /opt/zeek/bin/zeekctl diag        # xem chẩn đoán khi có lỗi
```

**Lưu ý quan trọng:**
- `zeekctl` nằm ở `/opt/zeek/bin/`, **không có sẵn trong PATH của `sudo`** dù đã thêm vào `~/.bashrc` — luôn gọi bằng đường dẫn đầy đủ `sudo /opt/zeek/bin/zeekctl ...` (hoặc dùng `sudo su -` để vào shell root rồi gọi `zeekctl` không cần đường dẫn đầy đủ).
- Bước 1 (bật promiscuous mode) **mất hiệu lực sau khi reboot VM** — phải chạy lại mỗi lần khởi động máy. Nếu muốn tự động, có thể thêm vào crontab `@reboot` hoặc netplan, nhưng chưa cấu hình việc này (làm thủ công cho tới giờ).
- Ở tầng VMware (ngoài OS), Network Adapter của VM máy 200 cũng cần bật **"Allow promiscuous mode: Accept"** trong cấu hình vSwitch/adapter — nếu không traffic của máy khác (attacker/victim) sẽ không thấy được dù `ip link` đã bật promiscuous trong guest.
- Để user `than` đọc trực tiếp log không cần `sudo` mỗi lần: `sudo usermod -aG zeek than` rồi đăng nhập lại (mở terminal mới/`newgrp zeek`).

**Auto-start sau reboot (đã cài đặt xong, xác nhận `active`, 2026-07-31):** vì mỗi lần khởi động lại máy 200, promiscuous mode mất hiệu lực và Zeek không tự chạy lại — đã cài `systemd` service tự làm 2 việc này mỗi khi boot. File nguồn lưu trong repo tại `scripts/zeek_nids_autostart.sh` (script) và `scripts/zeek-nids-autostart.service` (unit file). Cài đặt trên server:
```bash
sudo cp scripts/zeek_nids_autostart.sh /usr/local/bin/zeek-nids-autostart.sh
sudo chmod +x /usr/local/bin/zeek-nids-autostart.sh
sudo cp scripts/zeek-nids-autostart.service /etc/systemd/system/zeek-nids-autostart.service
sudo systemctl daemon-reload
sudo systemctl enable --now zeek-nids-autostart.service
```
Kiểm tra: `sudo systemctl status zeek-nids-autostart.service` (phải thấy `active (exited)`).

### 4.3. Kịch bản tấn công (Phase B — đã chốt, 2026-07-31)

Victim (199) hiện chưa cài dịch vụ gì — đã viết sẵn `scripts/victim_setup.sh` (cài `openssh-server`, `apache2`, `vsftpd`, tạo user `demo_target` mật khẩu yếu cố ý dùng cho kịch bản brute-force). **Chạy trên chính máy victim:**
```bash
git pull
sudo bash scripts/victim_setup.sh
```

**5 kịch bản, chạy từ Kali (194) nhắm vào victim (199)** — mỗi kịch bản chạy RIÊNG LẺ, cách nhau vài chục giây traffic nền yên tĩnh, và **ghi lại chính xác giờ:phút:giây bắt đầu/kết thúc** (dùng `date` trước/sau mỗi lệnh) — bắt buộc theo Phase C để đối chiếu nhãn thật với dự đoán model sau này.

**⚠️ QUAN TRỌNG (phát hiện 2026-07-31) — máy monitor (200) chỉ có 2 CPU / 3.3GB RAM:** tốc độ tấn công quá nhanh (`-T5`, `--min-rate 3000`, `hping3 --flood`) làm **Zeek quá tải, bỏ sót 87-99.8% gói tin thật** (đo được qua `capture_loss.log`, xem `docs/decisions.md` để biết chi tiết chẩn đoán) — flow ghi lại bị rỗng (thiếu gói phản hồi), khiến model không có đủ dữ liệu để phát hiện đúng, DÙ code hoàn toàn không lỗi. Tốc độ trong các lệnh dưới đây **đã giảm phù hợp với phần cứng thật của máy 200** — không tự ý tăng lại `--min-rate`/dùng `--flood` nếu muốn kết quả phát hiện đáng tin cậy. Trước khi chạy bất kỳ kịch bản nào, nên:
1. Tăng bộ đệm capture của Zeek 1 lần duy nhất: `sudo bash -c 'echo "redef Pcap::bufsize = 256;" >> /opt/zeek/share/zeek/site/local.zeek' && sudo /opt/zeek/bin/zeekctl deploy`
2. **Dừng `background_traffic_generator.py`** trước khi chạy tấn công thật (chạy nó trước để làm gần đầy cửa sổ, rồi tắt đi) — chạy đồng thời cả 2 sẽ cộng dồn tải, dễ làm Zeek quá tải trở lại.

**1. Port scan (trinh sát — Reconnaissance):**
```bash
date; nmap -T3 -p- --min-rate 50 --max-rate 200 192.168.207.199; date
```
(script sẵn: `bash scripts/run_scenario_portscan.sh 192.168.207.199`)

**2. Brute-force SSH:**
```bash
date; hydra -l demo_target -P /usr/share/wordlists/rockyou.txt -t 4 ssh://192.168.207.199 ; date
```
(Nếu muốn nhanh, tạo wordlist ngắn chứa sẵn mật khẩu đúng ở đầu: `printf "123456\nPassw0rd123\npassword\n" > /tmp/wl.txt` rồi dùng `-P /tmp/wl.txt`. Đây là kịch bản có khả năng bị phát hiện đúng CAO NHẤT trong 5 kịch bản — mỗi lần thử là 1 kết nối TCP đầy đủ, có trao đổi dữ liệu thật, tốc độ tự nhiên đã chậm do SSH giới hạn xác thực nên không gây quá tải Zeek.)

**3. Brute-force FTP:**
```bash
date; hydra -l demo_target -P /tmp/wl.txt ftp://192.168.207.199 ; date
```

**4. DoS lớp mạng — SYN Flood (cần `hping3`, có sẵn trên Kali) — KHÔNG dùng `--flood`:**
```bash
date; sudo timeout 30 hping3 -S -p 80 -i u10000 192.168.207.199 ; date
```
(`-i u10000` = cách nhau 10ms/gói thay vì tối đa tốc độ — `--flood` đã xác nhận làm Zeek mất >99% gói tin trên máy này.)

**5. DoS lớp ứng dụng — Slowloris (cần `slowloris`, có sẵn trên Kali):**
```bash
date; timeout 60 slowloris 192.168.207.199 -p 80 ; date
```

**Lưu ý an toàn:** cả 5 lệnh trên CHỈ chạy sau khi đã đổi mạng sang Host-only (mục "Cảnh báo chưa đúng yêu cầu isolation" phía trên) — SYN Flood/Slowloris là traffic tấn công thật, tuyệt đối không được có đường ra Internet thật.

## 5. Siêu tham số/hằng số cần dùng đúng khi viết code (Phase D-F)

Từ `src/models/gnn_config.py` (KHÔNG được đoán/tự đổi, phải khớp đúng lúc train):

```python
NODE_FEATURE_DIM = 43
EDGE_FEATURE_DIM = 39
NUM_LAYERS = 2
HIDDEN_DIM = 128            # dung cho nf-cse-cic-ids2018-v2
HIDDEN_DIM_BY_DATASET = {"nf-unsw-nb15-v2": 32}   # UNSW-NB15 dung 32, CSE-CIC dung HIDDEN_DIM=128
```

Cách nạp đúng model GraphSAGE (xem `evaluate_test.py` hàm `evaluate_graphsage` để chép chính xác pattern):

```python
from models.graphsage import GraphSAGEEdgeClassifier
model = GraphSAGEEdgeClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim, 2, NUM_LAYERS)
model.load_state_dict(torch.load(".../graphsage_best.pt", map_location=device, weights_only=True))
model.eval()
```

## 6. Phase D — Chuyển đổi log Zeek → đặc trưng NetFlow V2 (chi tiết kỹ thuật)

### 6.1. Schema đích (39 cột đặc trưng cạnh, đúng thứ tự đã dùng để fit scaler)

```
PROTOCOL, L7_PROTO, IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS, TCP_FLAGS, CLIENT_TCP_FLAGS,
SERVER_TCP_FLAGS, FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT, MIN_TTL, MAX_TTL,
LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, MAX_IP_PKT_LEN, SRC_TO_DST_SECOND_BYTES,
DST_TO_SRC_SECOND_BYTES, RETRANSMITTED_IN_BYTES, RETRANSMITTED_IN_PKTS, RETRANSMITTED_OUT_BYTES,
RETRANSMITTED_OUT_PKTS, SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT,
NUM_PKTS_UP_TO_128_BYTES, NUM_PKTS_128_TO_256_BYTES, NUM_PKTS_256_TO_512_BYTES,
NUM_PKTS_512_TO_1024_BYTES, NUM_PKTS_1024_TO_1514_BYTES, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT,
ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE
```

Cộng thêm 4 cột định danh: `IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT` (không đưa vào model, chỉ dùng để dựng đồ thị — xem `src/graph/nodes.py`).

### 6.2. Nguồn dữ liệu Zeek — `conn.log` là chính

**Schema thật đã xác nhận trên server (Zeek 8.2.1, lấy từ dòng `#fields` thật của `conn.log`, 2026-07-31):**
```
ts  uid  id.orig_h  id.orig_p  id.resp_h  id.resp_p  proto  service  duration  orig_bytes  resp_bytes
conn_state  local_orig  local_resp  missed_bytes  history  orig_pkts  orig_ip_bytes  resp_pkts
resp_ip_bytes  tunnel_parents  ip_proto
```
**Phát hiện quan trọng:** có sẵn cột `ip_proto` (số hiệu IANA protocol dạng số, vd tcp=6, udp=17, icmp=1) ở cuối — dùng **trực tiếp** cho cột `PROTOCOL` của NetFlow V2, không cần tự viết bảng map chuỗi "tcp"/"udp"/"icmp" như dự tính ban đầu bên dưới (mục 6.3 vẫn giữ để tham khảo nhưng cột `PROTOCOL` nên lấy thẳng từ `ip_proto`).

### 6.3. Đã code xong — `src/api/zeek_convert.py`

**Không còn là bảng đề xuất — đã viết code thật, kiểm chứng bằng dữ liệu `conn.log` thật thu được trên server (2026-07-31).** Hàm chính: `convert_row(row, scaler_mean) -> dict` (1 dòng Zeek → đủ 4 cột định danh + 39 cột đặc trưng), `convert_rows_to_dataframe(rows, scaler_mean) -> DataFrame`, `read_conn_log(path)` (đọc file, tự lấy đúng cột từ dòng `#fields`), `get_scaler_mean(scaler)`.

Cách map từng cột (đọc code để biết chi tiết, đây chỉ tóm tắt logic):
| Cột | Cách lấy |
|---|---|
| `IPV4_SRC_ADDR/PORT`, `IPV4_DST_ADDR/PORT` | Trực tiếp từ `id.orig_h/p`, `id.resp_h/p` |
| `PROTOCOL` | Trực tiếp từ `ip_proto` (Zeek đã cho số IANA, không cần tự map — xem mục 6.2) |
| `L7_PROTO` | **Luôn = 0** — không có bảng tra nDPI protocol ID đáng tin cậy, ghi rõ hạn chế này trong báo cáo |
| `IN/OUT_BYTES`, `IN/OUT_PKTS` | Trực tiếp từ `orig_bytes/resp_bytes/orig_pkts/resp_pkts` (parse an toàn, `-` → 0) |
| `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS` | Parse từ `history` — quy ước xác nhận trên dữ liệu thật: **CHỮ HOA = originator, chữ thường = responder** (vd `"Sr"` ở flow REJ = originator SYN, responder RST), map các chữ `F/S/R/D/A/H` sang bit cờ TCP tương ứng (xấp xỉ, không phải byte cờ thật vì Zeek không xuất trực tiếp) |
| `FLOW_DURATION_MILLISECONDS` | `duration × 1000` |
| `DURATION_IN`, `DURATION_OUT` | Xấp xỉ = `FLOW_DURATION_MILLISECONDS` (Zeek không tách riêng theo chiều) |
| `LONGEST/SHORTEST_FLOW_PKT`, `MIN/MAX_IP_PKT_LEN` | Xấp xỉ = độ dài gói **trung bình** (tổng `orig_ip_bytes+resp_ip_bytes` / tổng gói) — không có độ dài từng gói riêng lẻ nên không tính được min/max thật |
| `SRC/DST_TO_..._SECOND_BYTES`, `..._AVG_THROUGHPUT` | Tính từ bytes/duration (bytes/giây và bit/giây) |
| `NUM_PKTS_*_BYTES` (histogram 5 khoảng) | Xấp xỉ: dồn hết số gói vào đúng 1 khoảng chứa độ dài gói trung bình |
| `ICMP_TYPE`, `ICMP_IPV4_TYPE` | **Chính xác** (không phải xấp xỉ) khi `proto=icmp` — Zeek dùng lại `id.orig_p`/`id.resp_p` để chứa type/code, đã xác nhận trên dữ liệu ping thật (`ping` → type=8, code=0) |
| `MIN_TTL`, `MAX_TTL`, `TCP_WIN_MAX_IN/OUT` | Không có trong `conn.log` → dùng **trung bình tập train** lấy trực tiếp từ `scaler.mean_` (tránh giá trị 0 phi lý sau khi chuẩn hoá) |
| `RETRANSMITTED_*`, `DNS_*`, `FTP_COMMAND_RET_CODE` | = 0 (hợp lý vì hầu hết flow thật sự không retransmit / không phải DNS-FTP) — DNS/FTP có thể cải thiện sau bằng join `dns.log`/`ftp.log` theo `uid`, chưa làm |

### 6.4. Áp dụng đúng pipeline chuẩn hoá đã dùng lúc train

```python
from etl.scale import apply_scale  # co san, KHONG viet lai
df_scaled = apply_scale(df_raw, FEATURE_COLS, scaler, upper_bound)
```
(`FEATURE_COLS` từ `src/api/zeek_convert.py`, đúng thứ tự `scaler.feature_names_in_`.)

### 6.5. ⚠️ QUAN TRỌNG — model KHÔNG nhận 1 flow đơn lẻ, phải gom đủ 1 CỬA SỔ (window) rồi dựng đồ thị

**Phát hiện (2026-07-31, sửa lại thiết kế Phase F bên dưới cho đúng):** đọc lại `src/graph/node_features.py` + `build_graph.py` + `edges.py`/`nodes.py` thì thấy — GraphSAGE được train trên **đồ thị của cả 1 cửa sổ flow**, không phải từng flow riêng lẻ:
- Cạnh (edge) = 1 flow, đặc trưng cạnh = 39 cột đã chuẩn hoá (mục 6.1).
- Node (đỉnh) = 1 cặp `IP:port`, đặc trưng node (43 chiều) = 4 đặc trưng cấu trúc (in/out-degree, PageRank, clustering — tính từ **toàn bộ đồ thị của cửa sổ**) + trung bình 39 đặc trưng cạnh kề với node đó **trong cùng cửa sổ**.

→ Không thể "1 dòng `conn.log` mới xuất hiện → suy luận ngay" như bản thiết kế cũ ở mục 8 (Phase F) — phải **gom đủ N flow thành 1 cửa sổ, dựng đồ thị (`build_graph()`), rồi mới suy luận cho cả cửa sổ đó cùng lúc** (kết quả trả về là nhãn cho từng flow/cạnh trong cửa sổ). Nghĩa là demo có độ trễ tự nhiên theo cỡ cửa sổ, không phải tức thời "từng gói tin một".

**Cỡ cửa sổ lúc train:** `WINDOW_SIZE = 2_000` flow/cửa sổ (bộ `nf-cse-cic-ids2018-v2`, overlap 50% → cửa sổ sau trượt thêm 1.000 flow) — xem `src/graph/config.py`. Mạng ảo demo (chỉ 2-3 máy) sinh flow rất chậm bằng traffic thủ công, nên cần cân nhắc:

| Hướng | Ưu điểm | Nhược điểm |
|---|---|---|
| **Giữ nguyên 2.000 flow/cửa sổ** (đúng phân phối lúc train) | Độ chính xác cao nhất, đúng thống kê model đã học | Cần **script sinh traffic nền tự động** chạy song song trong lúc demo để đủ 2.000 flow trong thời gian ngắn, nếu không sẽ không có kết quả để trình chiếu |
| **Cửa sổ nhỏ hơn cho live** (vd 50-100 flow) | Ra kết quả nhanh, hợp trình chiếu trực tiếp | Đặc trưng cấu trúc node (degree/PageRank/clustering) lệch phân phối so với lúc train (đồ thị nhỏ hơn nhiều) — có thể giảm độ chính xác thật, cần ghi rõ hạn chế này trong báo cáo |

**Đề xuất mặc định (sẽ code theo hướng này trừ khi bạn chọn khác):** giữ `WINDOW_SIZE = 2000` đúng như lúc train, đồng thời viết thêm **1 script sinh traffic nền tự động** (vd script Python/bash gửi request lặp giữa các máy trong mạng ảo — ping, curl, DNS query...) chạy nền trong lúc demo để làm đầy cửa sổ nhanh hơn, kịch bản tấn công thật sẽ là một phần nhỏ trong dòng traffic nền đó (giống thực tế: traffic tấn công luôn trộn lẫn trong traffic bình thường). Cách này giữ đúng độ chính xác đã kiểm chứng, đổi lại cần chuẩn bị thêm 1 bước (script traffic nền) trước hôm demo.

**Quyết định (2026-07-31): đồng ý hướng đề xuất** — giữ nguyên `WINDOW_SIZE=2000` đúng phân phối lúc train, viết thêm script sinh traffic nền tự động (mục 8, Phase F) để làm đầy cửa sổ nhanh hơn trong lúc demo.

## 7. Phase E — Suy luận offline (+ so sánh Suricata — TẠM HOÃN)

**Đã code xong — `src/api/window_buffer.py` + `src/api/infer.py`:**
- `SlidingWindowBuffer(window_size, overlap)`: `add(row)` — nhận từng flow thô (dict từ `read_conn_log`), trả về `None` cho tới khi đủ 1 cửa sổ (`WINDOW_SIZE=2000`, trượt mỗi `1000` flow mới theo overlap 50%), lúc đó trả về `list[dict]` của cả cửa sổ. `flows_until_next_window()` — dùng cho UI hiện tiến độ chờ.
- `RealtimeDetector(processed_dir, folder_name)`: nạp **model + scaler 1 lần duy nhất** lúc khởi tạo (đúng pattern `evaluate_graphsage()` ở `evaluate_test.py`). `predict_window(rows)` — chạy trọn bộ pipeline 1 cửa sổ: `convert_rows_to_dataframe` → `apply_scale` → `build_graph` → gọi model → trả về `DataFrame` gồm định danh flow + `pred_label` (0/1) + `pred_proba` (xác suất Attack), **đúng thứ tự với `rows` đầu vào** (vì `build_edges()` giữ nguyên thứ tự dòng).
- **Đã test end-to-end** (2026-07-31) bằng dữ liệu `conn.log` thật thu trên server: chạy trọn pipeline Zeek → convert → scale → đồ thị → GraphSAGE, ra dự đoán hợp lệ, không lỗi (dùng cửa sổ nhỏ 20 flow chỉ để kiểm tra luồng code chạy được, KHÔNG phản ánh độ chính xác thật — độ chính xác thật chỉ đáng tin với cửa sổ đủ 2000 flow như lúc train, xem mục 6.5).
- **Quyết định (2026-07-31): tạm bỏ Suricata**, chỉ tập trung làm cho được demo real-time bằng GraphSAGE trước. Chưa cài Suricata trên server 200.
- So sánh Suricata (làm sau nếu cần, không bắt buộc cho demo trực tiếp): đọc `eve.json` (Suricata xuất JSON có trường `alert` khi phát hiện) theo cùng khung thời gian với dữ liệu Zeek đã gán nhãn, tính TP/FP/FN cho cả 2 hệ thống trên cùng traffic — dùng để chứng minh trong báo cáo rằng GraphSAGE phát hiện được nhiều kiểu tấn công hơn rule-based (ví dụ tấn công mới/biến thể không có sẵn luật) hoặc ít false positive hơn.

## 8. Phase F — API real-time

**Đã code xong — `src/api/realtime_server.py` (FastAPI + WebSocket) + `scripts/background_traffic_generator.py`.**

Cách chạy server demo trên máy 200:
```bash
cd /home/than/projects/gnn-nids-tttn
PYTHONPATH=src venv/bin/uvicorn api.realtime_server:app --app-dir src --host 0.0.0.0 --port 8000
```
Mở trình duyệt tới `http://192.168.207.200:8000/` — trang hiển thị: thanh tiến độ chờ đủ cửa sổ (tránh hội đồng hiểu nhầm hệ thống đứng), tổng số flow Attack/Benign, và bảng flow Attack gần nhất (IP nguồn/đích, xác suất). Cập nhật qua WebSocket, không cần refresh.

**Luồng xử lý bên trong** (đã đúng theo phát hiện mục 6.5, KHÔNG suy luận từng dòng đơn lẻ):
1. Thread nền `tail -f` file `conn.log` (bỏ qua nội dung có sẵn trước khi server khởi động, chỉ lấy flow mới).
2. Mỗi flow mới đưa vào `SlidingWindowBuffer`. Đủ cửa sổ (2000 flow) → `RealtimeDetector.predict_window()` → gửi kết quả qua WebSocket tới mọi client đang xem.
3. Giao diện tại `/` (HTML+JS nhúng thẳng trong `realtime_server.py`, không cần file tĩnh riêng).

**Để làm đầy cửa sổ nhanh hơn lúc demo**, chạy `scripts/background_traffic_generator.py` trên máy attacker hoặc victim (không phải máy monitor), nhắm vào (các) máy còn lại — sinh liên tục traffic TCP/UDP/ICMP đa dạng:
```bash
python3 scripts/background_traffic_generator.py 192.168.207.199 --rate 10
```
Với `--rate 10` (10 flow/giây), cửa sổ đầu tiên (2000 flow) mất khoảng ~3-4 phút, các cửa sổ sau (trượt 1000 flow) mất khoảng ~1.5-2 phút — nên bật script này TRƯỚC khi bắt đầu trình chiếu vài phút để có sẵn kết quả, rồi chạy kịch bản tấn công thật xen giữa dòng traffic nền đó.

**Giới hạn đã biết:** nếu Zeek xoay vòng (archive) `conn.log` giữa lúc demo (mặc định `zeekctl` xoay theo giờ), tiến trình tail đang mở sẽ không thấy file mới — với 1 buổi demo ngắn (~15-30 phút) khả năng thấp, nhưng nếu gặp thì restart lại `realtime_server.py` là đủ (không cần restart Zeek).

## 9. Phase G — Cập nhật docs

Sau khi có kết quả thật, thêm 1 file mới `docs/graphsage/06_ket_qua_realtime.md` (không sửa các file 00-05 hiện có) — ghi: số liệu TP/FP/FN model vs Suricata, ảnh chụp màn hình demo (nếu có), hạn chế thực tế gặp phải ở bước chuyển đổi đặc trưng (mục 6.3).
