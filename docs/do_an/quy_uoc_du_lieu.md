# Quy ước dữ liệu và giao diện giữa các phase

**Trạng thái:** thiết kế đề xuất để triển khai. Đọc cùng [kế hoạch tổng](README.md). Các field và thư mục mới trong file này chưa có sẵn trong repo.

## 1. Những điều phải thống nhất trước khi train

| Khái niệm | Quy ước |
|---|---|
| `task` | `binary` hoặc `multiclass`; không suy ra task từ số nhãn trong một batch |
| `dataset_id` | Tên nguồn và phiên bản, ví dụ `nf-cse-cic-ids2018-v2` |
| `label_schema_id` | Phiên bản định nghĩa nhãn; một ID chỉ có một ý nghĩa trong schema đó |
| `feature_profile` | `netflow_full_v2` cho nghiên cứu; `zeek_shared_v1` cho tập đặc trưng live đã xác minh |
| `split_id` | Phiên bản danh sách flow/session của train/val/test; mọi mô hình so sánh phải dùng chung |
| `run_id` | Tên duy nhất chứa task, dataset, profile, seed và cấu hình |
| `flow_id` | ID ổn định trong nguồn dữ liệu; không chỉ dùng IP:port vì nhiều flow có thể cùng endpoints |
| `window_id` | ID cửa sổ; giữ liên hệ về các flow ID gốc |

Các tên profile trên là tên đề xuất. Chỉ bật profile khi đã có mô tả từng đặc trưng và kiểm tra schema tương ứng.

## 2. Bản ghi flow chuẩn

Mỗi bản ghi cần: `flow_id`, nguồn thu, IP nguồn/đích, cổng nguồn/đích, giao thức, đặc trưng và metadata thời gian nếu nguồn có. `Label`, `Attack`, ID phiên/kịch bản và mọi dữ liệu tạo ground truth **không được vào feature tensor**.

Với CSV V2 chưa có timestamp, tạo ID từ `source_file + original_row_id` trước khi lọc/chia; không gọi thứ tự dòng là thời gian thật nếu chưa xác minh. Với live/replay, dùng `capture_id + zeek_uid` hoặc ID tương đương; `uid` không được mặc định duy nhất qua mọi lần chạy lại.

Giữ cả `event_start_ts`, `event_end_ts` khi có và `ingest_ts`. Chúng khác nhau: thời điểm kết nối bắt đầu không phải thời điểm hệ thống nhận được log để suy luận.

## 3. Nhãn

- Binary giữ rõ `0=Benign`, `1=Attack`.
- Multiclass lưu `class_id → class_name`, `benign_class_id`, danh sách lớp và phiên bản mapping trong artifact. Không giả định Benign luôn là ID 0 của `LabelEncoder`.
- Fit/khóa mapping từ tập train hoặc taxonomy công bố trước; không tự fit encoder riêng trên val/test. Lớp chỉ xuất hiện ở val/test phải được báo là không được hỗ trợ trong thiết lập closed-set.
- Chuẩn hóa tên có quy tắc; nếu gom FTP/SSH vào BruteForce phải có bảng ánh xạ và giữ nhãn gốc để truy vết. Không gọi gộp mọi tấn công vào một nhãn là multiclass.
- Hai dataset không mặc định có cùng lớp. Cross-dataset multiclass chỉ được chạy khi có bảng ánh xạ ngữ nghĩa rõ và quy tắc cho lớp không chung.

## 4. Chia tập và tiền xử lý

1. Giữ raw data và ID nguồn; xác định group/session hoặc các block độc lập trước.
2. Chia train/val/test dùng chung cho baseline và GNN, ưu tiên theo capture/session/ngày khi có metadata tin cậy.
3. Cửa sổ không vượt ranh giới split. Loại trùng xuyên tập và áp dụng khoảng cách ranh giới theo **độ phủ thực tế** của cửa sổ; một cửa sổ gap chỉ đủ với cấu hình overlap nhất định.
4. Fit ngưỡng clip, scaler, xử lý mất cân bằng và mọi bước học tham số **chỉ từ train**. Khi dùng cửa sổ overlap, fit scaler từ flow train duy nhất.
5. Test chỉ transform bằng artifact train. Giữ raw test để đánh giá chéo; `inverse_transform` không phục hồi được thông tin đã bị clip.
6. Không xáo trộn toàn bộ flow rồi dựng đồ thị. Không SMOTE từng cạnh rồi nối bừa vào một đồ thị mới.

Nếu chia theo thời gian làm một lớp vắng khỏi train, phải báo và thiết kế lại split/group hoặc phân biệt bài toán open-set; không lén chuyển mẫu test sang train sau khi xem kết quả.

## 5. Graph contract

`Data.x: [N, F_node]`, `edge_index: [2, E]`, `edge_attr: [E, F_edge]`, `y: [E]`. Mỗi hàng output `[E, C]` phải khớp đúng flow ID của cạnh tương ứng. Lưu node ID → endpoint để dashboard hiển thị.

Kích thước đầu vào lấy từ schema; không hard-code 43/39 khi profile thay đổi. Giữ graph builder dùng được cho suy luận không nhãn: không buộc giả nhãn 0 rồi diễn giải nó là ground truth.

Đặc trưng node tính từ flow của **cửa sổ đã có sẵn**, không lấy cạnh ngoài cửa sổ hoặc tương lai. Đồ thị ngữ cảnh phục vụ batch-window detection, không tuyên bố phát hiện ngay từ gói đầu tiên.

## 6. Artifact mô hình

Cấu trúc đích dự kiến:

```text
data/processed/<dataset_id>/<task>/<feature_profile>/<split_id>/
  schema.json, label_mapping.json, split_manifest.parquet
  raw_test.parquet, scaler.joblib, upper_bound.joblib
  graph_shards/...
results/graduation/<run_id>/
  config.json, environment.txt, metrics.json
  per_class_metrics.csv, confusion_matrix.csv, predictions.parquet
  model_bundle/...
```

`model_bundle` chứa kiến trúc/cấu hình, state dict, ordered feature names, mapping nhãn, scaler/clip, loại và kích thước cửa sổ, metadata phiên bản và checksum. Loader phải kiểm tra tương thích trước khi dự đoán. Tách checkpoint `best` để đánh giá và `last` đủ optimizer/scheduler/RNG để resume.

Không ghi đè model binary thực tập khi train multiclass. Đường dẫn mới phải được truyền qua CLI/config, không suy đoán từ tên thư mục hiện tại.

## 7. Prediction và alert contract

Một prediction lưu `prediction_id`, `flow_id`, `window_id`, `model_id/run_id`, `task`, `feature_profile`, `label_schema_id`, `pred_class_id`, `pred_class_name`, `class_scores`, `is_attack` và các timestamp xử lý.

Với multiclass, `is_attack = pred_class_id != benign_class_id`. Nếu cần điểm Attack tổng hợp, dùng tổng xác suất các lớp không Benign; không dùng cứng cột xác suất số 1.

Đánh giá live mặc định dùng **dự đoán đầu tiên** cho mỗi flow; dashboard có thể cập nhật dự đoán mới nhưng phải lưu phiên bản và khử lặp. Báo riêng `unique_flows`, `prediction_events`, `unique_alerts`; không trộn các mẫu số này.

Giữ bản chụp đồ thị/đặc trưng đã dùng cho cảnh báo để giải thích sau; không dựng giải thích từ cửa sổ hiện tại khác với cửa sổ đã sinh dự đoán.

## 8. Ground truth và explanations

Ground truth lưu riêng `capture_id, scenario_id, run_id, flow_id, true_label, label_source, label_confidence`. Nhãn chưa đủ bằng chứng để `unknown/unlabeled`, không mặc định Benign. Không dùng output mô hình hoặc luật IDS khác làm nhãn thật duy nhất.

Explanation lưu `prediction_id`, `target_flow_id`, lớp được giải thích, algorithm/version, feature profile, ID cạnh/nút, importance scores, reference/baseline, random seed, thời gian và quy tắc perturbation. Phân biệt **edge mask cấu trúc** với **importance từng chiều edge_attr**.

## 9. Lưu vết công việc cho AI

Mỗi phase tạo báo cáo Markdown dự kiến tại `results/graduation/phase_reports/phase_XX.md`. Ghi trạng thái, file sửa, lệnh thật đã chạy, dữ liệu dùng, artifact và mục chưa đạt. Chỉ tick checklist nghiệm thu bằng bằng chứng, không dùng câu “đã xong” thay số liệu.

Các schema ở đây là điểm xuất phát để triển khai. Nếu cần điều chỉnh, cập nhật đồng thời producer, consumer và tài liệu; không yêu cầu người dùng duyệt lại lựa chọn kỹ thuật thường lệ đã nằm trong phạm vi.
