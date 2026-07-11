# Phân tích đề tài và định hướng triển khai

**Tài liệu định hướng nghiên cứu — Thực tập tốt nghiệp**

**Đề tài:** Nghiên cứu và ứng dụng mạng nơ-ron đồ thị phát hiện xâm nhập mạng (GNN-based NIDS)

- Sinh viên thực hiện: Trần Gia Thân — N22DCAT050, Trần Xuân Đông — N22DCAT018
- Lớp: D22CQAT01-N — Chuyên ngành: An toàn thông tin
- Đơn vị thực tập: Nền tảng MyAloha — Mentor: Khanh
- Thành phố Hồ Chí Minh, 2026

---

## 1. Giới thiệu chung về đề tài

Đề tài “Nghiên cứu và ứng dụng mạng nơ-ron đồ thị phát hiện xâm nhập mạng” (Graph Neural Network-based Network Intrusion Detection System — GNN-IDS) hướng đến việc xây dựng một hệ thống phát hiện xâm nhập mạng dựa trên mạng nơ-ron đồ thị, thay vì tiếp cận theo hướng phân tích từng luồng mạng (network flow) một cách độc lập như các phương pháp học máy truyền thống. Điểm khác biệt cốt lõi của hướng tiếp cận này nằm ở việc mô hình hoá toàn bộ lưu lượng mạng thành một cấu trúc đồ thị, trong đó mỗi địa chỉ IP được biểu diễn như một nút (node) và mỗi kết nối mạng được biểu diễn như một cạnh (edge). Nhờ cấu trúc này, mạng nơ-ron đồ thị có khả năng học được các mối quan hệ giữa nhiều kết nối khác nhau — điều mà các mô hình học máy truyền thống như Random Forest hay XGBoost không thể khai thác được do bản chất xử lý độc lập từng bản ghi.

Tài liệu này được biên soạn nhằm trình bày một cách có hệ thống toàn bộ quá trình phân tích đề tài, từ việc xác định loại hình nghiên cứu, phạm vi và câu hỏi nghiên cứu, cho đến việc lựa chọn phương pháp luận, thiết kế thực nghiệm và định hướng triển khai theo từng giai đoạn cụ thể. Mục tiêu là tạo ra một tài liệu tham chiếu đầy đủ, giúp nhóm thực hiện — cũng như giảng viên hướng dẫn — có cái nhìn nhất quán và rõ ràng về hướng đi của đề tài trước khi bắt tay vào triển khai kỹ thuật.

Toàn bộ nội dung trong tài liệu này là kết quả của quá trình rà soát, phản biện và điều chỉnh nhiều lần đối với kế hoạch ban đầu, nhằm đảm bảo tính khả thi trong khung thời gian thực tập tốt nghiệp (6 tuần) mà không đánh đổi giá trị khoa học của đề tài.

## 2. Loại hình và phương pháp nghiên cứu

### 2.1. Xác định loại hình nghiên cứu

Đề tài thuộc loại hình nghiên cứu ứng dụng (applied research) kết hợp với nghiên cứu thực nghiệm so sánh (comparative experimental research). Đây không phải là một nghiên cứu lý thuyết thuần tuý nhằm đề xuất một kiến trúc mạng nơ-ron đồ thị hoàn toàn mới, mà là nghiên cứu ứng dụng các kiến trúc GNN đã được công bố và kiểm chứng trong cộng đồng khoa học (GCN, GAT, GraphSAGE) vào một bài toán cụ thể là phát hiện xâm nhập mạng, sau đó đánh giá một cách có hệ thống hiệu quả của các kiến trúc này so với các phương pháp học máy truyền thống.

Về bản chất, bài toán được tiếp cận dưới dạng phân loại đa lớp có giám sát (multi-class supervised classification). Mỗi luồng mạng trong tập dữ liệu huấn luyện đã được gán nhãn sẵn (bình thường hoặc một trong các loại tấn công cụ thể), và mô hình học ánh xạ từ vector đặc trưng của luồng mạng — hoặc từ biểu diễn đồ thị bao gồm luồng mạng đó — sang nhãn tương ứng. Cách tiếp cận này khác với hướng phát hiện bất thường không giám sát (unsupervised anomaly detection), vốn không yêu cầu nhãn nhưng cũng không cho biết chính xác loại tấn công nào đang xảy ra.

### 2.2. Phương pháp luận tổng quát

Phương pháp luận của đề tài được xây dựng theo quy trình nghiên cứu thực nghiệm chuẩn trong lĩnh vực học máy ứng dụng, bao gồm các bước tuần tự: xác định câu hỏi nghiên cứu, thu thập và chuẩn bị dữ liệu, xây dựng và huấn luyện mô hình, thiết kế thực nghiệm để trả lời từng câu hỏi nghiên cứu, phân tích kết quả và rút ra kết luận. Điểm quan trọng cần nhấn mạnh là mọi quyết định kỹ thuật trong đề tài — từ việc chọn kiến trúc mô hình đến việc thiết kế từng thí nghiệm — đều xuất phát từ một câu hỏi nghiên cứu cụ thể, không triển khai tính năng một cách rời rạc thiếu định hướng.

Toàn bộ quy trình tuân thủ nguyên tắc tách biệt dữ liệu huấn luyện và dữ liệu đánh giá (train-test separation) một cách nghiêm ngặt: mô hình không bao giờ được tiếp xúc với dữ liệu dùng để đánh giá cuối cùng trong quá trình học, đảm bảo kết quả đánh giá phản ánh đúng khả năng của mô hình chứ không phải hiện tượng học thuộc (memorization).

## 3. Phân tích đề tài

### 3.1. Bài toán nghiên cứu

Bài toán trung tâm của đề tài là: cho một luồng mạng (network flow) hoặc một tập hợp các luồng mạng có quan hệ với nhau, xác định luồng đó thuộc loại bình thường hay một trong các loại tấn công đã được định nghĩa (dò quét cổng, dò mật khẩu, từ chối dịch vụ, tấn công web...). Điểm đặc thù của đề tài so với các bài toán phân loại luồng mạng thông thường là việc đưa thêm chiều thông tin về cấu trúc quan hệ giữa các luồng — cụ thể là quan hệ giữa các địa chỉ IP tham gia kết nối — vào quá trình ra quyết định của mô hình.

### 3.2. Phạm vi nghiên cứu: NIDS, không mở rộng sang HIDS

Đề tài xác định rõ phạm vi ở mức phát hiện xâm nhập dựa trên mạng (Network Intrusion Detection System — NIDS), không mở rộng sang phát hiện xâm nhập dựa trên máy chủ (Host-based Intrusion Detection System — HIDS). Sự phân biệt này có ý nghĩa quan trọng và cần được quán triệt xuyên suốt quá trình triển khai:

- **NIDS (phạm vi của đề tài):** phân tích tại mức mạng — các thuộc tính được trích xuất là thuộc tính của luồng kết nối: địa chỉ IP nguồn/đích, cổng, giao thức, thời lượng kết nối, số byte, số gói tin truyền, các cờ TCP... Công cụ thu thập tương ứng là Zeek và CICFlowMeter, tạo ra log dạng conn.log và các đặc trưng flow-level.
- **HIDS (nằm ngoài phạm vi):** phân tích tại mức máy chủ — nhật ký tiến trình, thay đổi tập tin hệ thống, hành vi của tiến trình đang chạy trên máy nạn nhân. Đề tài không thu thập hay xử lý loại dữ liệu này.

Việc xác định rõ ranh giới này ngay từ đầu giúp tránh tình trạng đề tài bị lệch phạm vi trong quá trình triển khai, đồng thời giúp nhóm trả lời chính xác nếu hội đồng đặt câu hỏi về giới hạn của hệ thống. Một hệ quả trực tiếp của việc giới hạn ở mức NIDS là hệ thống chỉ có khả năng phát hiện các dấu hiệu bất thường ở mức luồng mạng (ví dụ: một địa chỉ IP đang quét nhiều cổng trong thời gian ngắn), nhưng không có khả năng phân tích nội dung gói tin hay xác định chính xác lệnh nào đang được thực thi trên máy nạn nhân. Đây là một hạn chế cố hữu của NIDS so với HIDS, cần được trình bày rõ trong phần hạn chế của báo cáo.

### 3.3. Đối tượng nghiên cứu

Đối tượng nghiên cứu trực tiếp là luồng mạng (network flow) được biểu diễn dưới hai dạng song song trong đề tài:

1. **Dạng bảng đặc trưng (tabular representation):** mỗi luồng mạng là một vector đặc trưng độc lập, được sử dụng làm đầu vào cho các mô hình học máy truyền thống (Random Forest, XGBoost).
2. **Dạng đồ thị (graph representation):** các luồng mạng được tổ chức thành đồ thị có hướng, trong đó nút là địa chỉ IP và cạnh là luồng kết nối mang theo vector đặc trưng riêng. Đây là đầu vào cho các mô hình mạng nơ-ron đồ thị (GCN, GAT, GraphSAGE).

Việc duy trì song song hai dạng biểu diễn cho phép đề tài thực hiện so sánh công bằng giữa hai trường phái tiếp cận: tiếp cận truyền thống coi mỗi luồng là độc lập, và tiếp cận đồ thị coi các luồng có quan hệ cấu trúc với nhau.

### 3.4. Câu hỏi nghiên cứu

Toàn bộ thiết kế thực nghiệm của đề tài được xây dựng để trả lời ba câu hỏi nghiên cứu cụ thể sau đây. Mỗi câu hỏi tương ứng trực tiếp với một hoặc nhiều thí nghiệm sẽ được trình bày chi tiết ở phần 6.

#### *Câu hỏi nghiên cứu 1 (RQ1) — Hiệu quả trong phạm vi một tập dữ liệu*

Trong cùng một tập dữ liệu (huấn luyện và kiểm thử đều lấy từ cùng một nguồn thu thập), các mô hình mạng nơ-ron đồ thị (GCN, GAT) có đạt hiệu quả phân loại tốt hơn, tương đương, hay kém hơn so với các mô hình học máy truyền thống (Random Forest, XGBoost)? Câu hỏi này được trả lời thông qua Thí nghiệm 1 (TN1 — đánh giá within-dataset).

#### *Câu hỏi nghiên cứu 2 (RQ2) — Khả năng tổng quát hoá*

Khi mô hình được huấn luyện trên một tập dữ liệu và kiểm thử trên một tập dữ liệu hoàn toàn khác — được thu thập độc lập, ở mạng khác, thời điểm khác, bởi nhóm nghiên cứu khác — thì mô hình mạng nơ-ron đồ thị có duy trì được hiệu quả tốt hơn so với mô hình học máy truyền thống hay không? Đây là câu hỏi quan trọng nhất về mặt khoa học, vì nó kiểm chứng liệu mô hình có thực sự học được bản chất của hành vi tấn công hay chỉ đang ghi nhớ các đặc điểm riêng của một lần thu thập dữ liệu cụ thể (một dạng học thuộc dữ liệu ở cấp độ tập dữ liệu). Câu hỏi này được trả lời thông qua Thí nghiệm 2 (TN2 — đánh giá cross-dataset), sử dụng hai bộ dữ liệu công khai NF-CICIDS2017 và NF-UNSW-NB15-v2, vốn được hai nhóm nghiên cứu độc lập (Đại học New Brunswick và Đại học New South Wales) thu thập trong môi trường mạng và thời điểm khác nhau.

#### *Câu hỏi nghiên cứu 3 (RQ3, tuỳ chọn) — So sánh với hệ thống phát hiện dựa trên luật trong môi trường thực nghiệm tự thiết lập*

Trong một môi trường mạng thực nghiệm tự thiết lập (lab), nơi nhóm thực hiện có toàn quyền kiểm soát và có thể tấn công trực tiếp theo thời gian thực (real-time), mô hình GNN đã huấn luyện có phát hiện được các cuộc tấn công hay không, và kết quả phát hiện so với hệ thống phát hiện xâm nhập dựa trên luật (rule-based IDS, cụ thể là Suricata với bộ luật ET Open Rules) như thế nào? Đây là câu hỏi mang tính ứng dụng thực tiễn, được trả lời thông qua Thí nghiệm 6 (TN6), và chỉ được thực hiện nếu nhóm quyết định thiết lập môi trường lab để phục vụ nhu cầu trình diễn tấn công theo thời gian thực (xem phân tích chi tiết tại mục 4.3).

## 4. Phương án dữ liệu và cơ sở lựa chọn

### 4.1. Dữ liệu huấn luyện: hai bộ dữ liệu công khai chuẩn hoá

Mô hình được huấn luyện duy nhất trên hai bộ dữ liệu công khai đã được chuẩn hoá theo định dạng NetFlow: NF-CICIDS2017 và NF-UNSW-NB15-v2 (Sarhan và cộng sự). Quyết định sử dụng phiên bản NetFlow chuẩn hoá thay vì bộ dữ liệu gốc CICIDS2017/UNSW-NB15 xuất phát từ một cân nhắc kỹ thuật quan trọng: hai bộ dữ liệu gốc có tập đặc trưng hoàn toàn khác nhau về số lượng và ý nghĩa (CICIDS2017 trích xuất qua CICFlowMeter với khoảng 80 đặc trưng, UNSW-NB15 trích xuất qua công cụ Argus/Bro với bộ đặc trưng riêng). Việc tự xây dựng một tập đặc trưng chung (unified schema) giữa hai bộ dữ liệu có schema khác biệt là một bài toán con phức tạp, dễ dẫn đến sai lệch về mặt ngữ nghĩa nếu thực hiện không cẩn trọng. Bộ dữ liệu NetFlow chuẩn hoá đã được nhóm nghiên cứu của Sarhan xử lý sẵn theo một schema thống nhất, cho phép hai bộ dữ liệu tương thích trực tiếp với nhau mà không cần thêm bước ánh xạ đặc trưng thủ công — đây là quyết định giúp tiết kiệm đáng kể thời gian mà không đánh đổi tính đúng đắn khoa học.

Mỗi bộ dữ liệu được chia theo tỷ lệ 70% huấn luyện (train), 15% kiểm định (validation) và 15% kiểm thử (test). Tập huấn luyện dùng để cập nhật trọng số mô hình; tập kiểm định dùng để điều chỉnh siêu tham số và thực hiện dừng sớm (early stopping); tập kiểm thử được giữ lại hoàn toàn tách biệt và chỉ được sử dụng một lần duy nhất, vào giai đoạn đánh giá cuối cùng (Thí nghiệm 1).

### 4.2. Vai trò của tập dữ liệu bổ sung: kiểm tra khả năng tổng quát hoá

Một nguyên tắc cần được quán triệt rõ ràng trong toàn bộ đề tài: bất kỳ tập dữ liệu bổ sung nào được đưa vào để kiểm tra khả năng tổng quát hoá của mô hình — dù là tập dữ liệu công khai thứ hai (UNSW-NB15 khi mô hình được huấn luyện trên CICIDS2017, và ngược lại) hay tập dữ liệu tự thu thập tại môi trường lab — đều chỉ đóng vai trò kiểm thử (test-only). Mô hình không bao giờ được huấn luyện hay cập nhật trọng số dựa trên các tập dữ liệu này. Đây là điều kiện tiên quyết để kết quả đánh giá tổng quát hoá có giá trị khoa học.

Cần làm rõ một điểm dễ gây nhầm lẫn: bản thân việc đánh giá chéo giữa hai bộ dữ liệu công khai (Thí nghiệm 2) đã đóng vai trò kiểm tra khả năng tổng quát hoá một cách có ý nghĩa, bởi lẽ NF-CICIDS2017 và NF-UNSW-NB15-v2 được hai nhóm nghiên cứu hoàn toàn độc lập thu thập, tại hai môi trường mạng khác nhau, vào các thời điểm khác nhau. Nói cách khác, ngay cả khi đề tài không thiết lập thêm môi trường lab để thu dữ liệu riêng, Thí nghiệm 2 vẫn đảm bảo đề tài có một phép kiểm tra tổng quát hoá đáng tin cậy — đây là cơ sở quan trọng cho quyết định về việc có cần thiết lập môi trường lab hay không, được phân tích chi tiết ở mục tiếp theo.

### 4.3. Quyết định về việc thiết lập môi trường lab: khung ra quyết định

Việc có nên thiết lập một môi trường mạng thực nghiệm (lab) riêng bằng VMware để tự thu thập dữ liệu tấn công hay không là một quyết định có ảnh hưởng đáng kể đến khối lượng công việc và rủi ro triển khai của đề tài. Sau quá trình phân tích, nhóm xác định quyết định này phụ thuộc hoàn toàn vào một câu hỏi duy nhất, mang tính chi phối: nhóm có nhu cầu trình diễn tấn công theo thời gian thực (real-time) khi báo cáo trước hội đồng hay không.

Lý do câu hỏi này mang tính chi phối: nếu không có nhu cầu trình diễn tấn công thời gian thực, hai bộ dữ liệu công khai kết hợp với Thí nghiệm 2 (đánh giá chéo dataset) đã đủ để trả lời trọn vẹn các câu hỏi nghiên cứu cốt lõi về hiệu quả và khả năng tổng quát hoá của mô hình (RQ1 và RQ2). Việc thiết lập thêm môi trường lab trong trường hợp này sẽ tạo thêm khối lượng công việc (dựng máy ảo, cấu hình mạng, cài đặt và cấu hình Zeek/Suricata, thực hiện lần lượt năm kịch bản tấn công, xử lý và gán nhãn dữ liệu thu được) mà không mang lại thêm giá trị khoa học tương xứng, do câu hỏi tổng quát hoá đã được Thí nghiệm 2 giải quyết.

Ngược lại, nếu nhóm có nhu cầu trình diễn tấn công theo thời gian thực trước hội đồng — một yêu cầu mang tính trình diễn thực tế, giúp tăng tính thuyết phục và trực quan của buổi báo cáo — thì việc thiết lập môi trường lab là điều kiện bắt buộc, không có phương án thay thế nào khác, bởi hai bộ dữ liệu công khai chỉ tồn tại dưới dạng tệp tĩnh (CSV đặc trưng đã trích xuất sẵn), không phải là một môi trường mạng sống có thể thực hiện tấn công trực tiếp lên đó.

Trong trường hợp môi trường lab được thiết lập vì lý do trình diễn, nhóm xác định nên tận dụng luôn dữ liệu thu được trong quá trình đó để phục vụ Thí nghiệm 6 (so sánh với Suricata), do khối lượng công việc bổ sung để thực hiện việc này gần như không đáng kể so với công sức đã bỏ ra để dựng lab — dữ liệu vốn đã được Zeek ghi lại một cách tự động trong suốt quá trình vận hành và trình diễn.

#### *Bảng tổng hợp khung ra quyết định*

| **Tiêu chí** | **Phương án A: có môi trường lab** | **Phương án B: không có môi trường lab** |
| --- | --- | --- |
| Điều kiện áp dụng | Có nhu cầu trình diễn tấn công real-time trước hội đồng | Không có nhu cầu trình diễn real-time |
| Phạm vi thí nghiệm | TN1, TN2, TN6 (đầy đủ) | TN1, TN2 (RQ1 và RQ2 vẫn được trả lời đầy đủ) |
| Khối lượng công việc bổ sung | Thiết lập VMware, cấu hình mạng, Zeek/Suricata, 5 kịch bản tấn công | Không phát sinh |
| Giá trị khoa học bổ sung | So sánh thực tế với hệ thống rule-based (Suricata) | Không mất giá trị khoa học cốt lõi (RQ1, RQ2 đã đủ) |
| Rủi ro kỹ thuật | Cấu hình VMware, Promiscuous Mode, độ chính xác gán nhãn thủ công | Không có rủi ro tương ứng |
| Phù hợp nếu | Muốn phần trình diễn ấn tượng, trực quan hơn | Muốn tối ưu thời gian, tập trung chiều sâu vào mô hình và thực nghiệm |

Việc trình bày tường minh khung ra quyết định này trong tài liệu nhằm đảm bảo mọi lựa chọn kỹ thuật của đề tài đều có căn cứ rõ ràng, có thể giải trình được trước giảng viên hướng dẫn và hội đồng, thay vì là các quyết định rời rạc thiếu nhất quán.

### 4.4. Ranh giới quan trọng cần tuân thủ nếu triển khai môi trường lab

Trong trường hợp quyết định thiết lập môi trường lab (Phương án A), có một ranh giới kỹ thuật và pháp lý bắt buộc phải tuân thủ: toàn bộ hoạt động tấn công, bao gồm cả kịch bản từ chối dịch vụ (DoS), chỉ được thực hiện trong môi trường mạng ảo cục bộ (VMware, mạng host-only, cô lập hoàn toàn khỏi Internet). Nếu đề tài có sử dụng hạ tầng đám mây (ví dụ AWS EC2) để lưu trữ và trình diễn hệ thống (API phục vụ dự đoán, giao diện dashboard), hạ tầng đám mây đó chỉ đóng vai trò lưu trữ và phục vụ mô hình đã huấn luyện xong, tuyệt đối không được sử dụng để thực hiện bất kỳ hình thức tấn công từ chối dịch vụ nào — đây là yêu cầu bắt buộc theo chính sách sử dụng dịch vụ của các nhà cung cấp đám mây, vi phạm có thể dẫn đến việc tài khoản bị đình chỉ.

## 5. Kiến trúc và phương pháp kỹ thuật

### 5.1. Xây dựng biểu diễn đồ thị từ luồng mạng

Bước chuyển đổi từ dữ liệu dạng bảng sang biểu diễn đồ thị là khâu kỹ thuật nền tảng, quyết định chất lượng đầu vào cho toàn bộ các mô hình GNN phía sau. Quy trình xây dựng đồ thị bao gồm các bước: mã hoá địa chỉ IP thành chỉ số nguyên (do các thư viện GNN yêu cầu chỉ số nút dạng số nguyên liên tục); xây dựng danh sách cạnh (edge index) biểu diễn các kết nối giữa những địa chỉ IP; tính toán các đặc trưng cho từng nút dựa trên vị trí của nút đó trong cấu trúc đồ thị tổng thể — bao gồm bậc vào (in-degree), bậc ra (out-degree), độ quan trọng theo thuật toán PageRank, và hệ số phân cụm (clustering coefficient); và cuối cùng là áp dụng kỹ thuật cửa sổ trượt (sliding window) với độ chồng lấp 50% để chia luồng dữ liệu liên tục thành các đồ thị con có kích thước quản lý được.

Việc lựa chọn các đặc trưng cấu trúc đồ thị nêu trên không phải ngẫu nhiên, mà dựa trên quan sát thực tế về hành vi tấn công: một địa chỉ IP thực hiện tấn công dò quét cổng (PortScan) thường có bậc ra rất cao (kết nối đến nhiều cổng đích khác nhau trong thời gian ngắn); một địa chỉ IP là mục tiêu của tấn công từ chối dịch vụ phân tán (DDoS) thường có bậc vào đột biến (nhiều nguồn cùng lúc kết nối đến); hệ số phân cụm thấp bất thường thường xuất hiện trong các kết nối do kẻ tấn công tạo ra, vì các kết nối này không có quan hệ tự nhiên với nhau như trong lưu lượng bình thường.

### 5.2. Bài toán phân loại cạnh (edge classification)

Một quyết định thiết kế quan trọng của đề tài là xác định bài toán phân loại ở mức cạnh (edge), không phải mức nút (node). Lý do xuất phát từ bản chất của hành vi tấn công: tấn công xảy ra trên một kết nối cụ thể (cạnh), không phải trên một máy chủ (nút) — một địa chỉ IP hoàn toàn có thể vừa là nạn nhân trong một kết nối, vừa là nguồn tấn công trong một kết nối khác cùng thời điểm. Cách thực hiện cụ thể: sau khi mô hình GNN tính toán được biểu diễn véc-tơ (embedding) cho từng nút thông qua cơ chế truyền thông điệp (message passing), với mỗi cạnh nối hai nút u và v, hệ thống ghép nối (concatenate) hai véc-tơ biểu diễn của u và v, sau đó đưa qua một bộ phân loại tuyến tính để tính xác suất thuộc về từng lớp tấn công.

### 5.3. Các kiến trúc mô hình được sử dụng

#### *5.3.1. Mô hình học máy truyền thống (baseline)*

Random Forest và XGBoost được sử dụng làm hai mô hình đối chứng (baseline). Cả hai đều là mô hình dựa trên cây quyết định, xử lý mỗi luồng mạng như một mẫu độc lập, không khai thác được mối quan hệ cấu trúc giữa các luồng. Việc đưa hai mô hình này vào so sánh nhằm mục đích trả lời trực tiếp câu hỏi nghiên cứu thứ nhất: liệu việc bổ sung thông tin cấu trúc đồ thị có thực sự mang lại lợi ích so với cách tiếp cận truyền thống hay không.

#### *5.3.2. Graph Convolutional Network (GCN)*

GCN là kiến trúc mạng nơ-ron đồ thị đơn giản nhất trong ba kiến trúc được sử dụng, đóng vai trò là mô hình GNN cơ sở. Nguyên lý hoạt động: mỗi nút cập nhật biểu diễn của mình bằng cách lấy trung bình có trọng số đặc trưng của bản thân và của các nút hàng xóm, sau đó nhân với một ma trận trọng số có thể học được. Hạn chế của GCN là coi tất cả hàng xóm có tầm quan trọng ngang nhau, không phân biệt được hàng xóm nào mang thông tin quan trọng hơn.

#### *5.3.3. Graph Attention Network (GAT)*

GAT khắc phục hạn chế của GCN bằng cơ chế chú ý (attention), cho phép mô hình tự học trọng số quan trọng khác nhau cho từng cặp nút thay vì coi mọi hàng xóm ngang nhau. Đây là mô hình được kỳ vọng cho kết quả tốt nhất trong đề tài, do lưu lượng mạng bình thường chiếm đa số áp đảo trong dữ liệu — cơ chế chú ý giúp mô hình tập trung vào những kết nối có dấu hiệu bất thường thay vì bị nhiễu bởi khối lượng lớn lưu lượng bình thường xung quanh.

#### *5.3.4. GraphSAGE (triển khai nếu còn thời gian)*

GraphSAGE là kiến trúc hướng đến khả năng mở rộng quy mô, phù hợp với bối cảnh lưu lượng mạng thực tế liên tục phát sinh dữ liệu mới. Thay vì yêu cầu nhìn thấy toàn bộ đồ thị trong quá trình huấn luyện như GCN và GAT, GraphSAGE lấy mẫu một số lượng hàng xóm cố định để học hàm tổng hợp có thể áp dụng cho các nút mới chưa từng xuất hiện trong tập huấn luyện. Do khối lượng công việc để triển khai đầy đủ ba kiến trúc GNN trong khung thời gian 6 tuần là đáng kể, GraphSAGE được xếp vào hạng mục triển khai có điều kiện: chỉ thực hiện nếu tiến độ huấn luyện GCN và GAT thuận lợi; nếu không, việc so sánh GCN (đơn giản) với GAT (có cơ chế chú ý) đã đủ để đưa ra kết luận khoa học có ý nghĩa.

### 5.4. Giải thích mô hình (Explainable AI)

Đề tài tích hợp GNNExplainer nhằm giải quyết vấn đề “hộp đen” thường gặp ở các mô hình học sâu. Trong bối cảnh bảo mật, một chuyên viên phân tích (analyst) không chỉ cần biết kết luận “đây là tấn công”, mà còn cần biết lý do hệ thống đưa ra kết luận đó để quyết định có nên xử lý ngay hay không. GNNExplainer hoạt động bằng cách tìm tập con nhỏ nhất của đồ thị (một số nút và cạnh nhất định) mà nếu loại bỏ sẽ làm thay đổi đáng kể kết quả dự đoán của mô hình — tập con này được xem là “bằng chứng” chính mà mô hình dựa vào để đưa ra kết luận, và được trực quan hoá trên giao diện giám sát dưới dạng đồ thị con được tô sáng kèm theo danh sách các đặc trưng đóng góp nhiều nhất.

## 6. Quy trình thực nghiệm và tiêu chí đánh giá

### 6.1. Các chỉ số đánh giá được sử dụng

Do dữ liệu tấn công mạng có đặc điểm mất cân bằng nghiêm trọng giữa các lớp (lưu lượng bình thường thường chiếm trên 80% tổng số mẫu, trong khi một số loại tấn công hiếm chỉ chiếm dưới 0,1%), việc lựa chọn chỉ số đánh giá phù hợp có ý nghĩa quyết định đến độ tin cậy của kết luận nghiên cứu. Đề tài sử dụng đồng thời nhiều chỉ số bổ trợ lẫn nhau:

- **Accuracy:** chỉ mang tính tham khảo, không dùng làm chỉ số quyết định do dễ gây hiểu lầm khi dữ liệu mất cân bằng (một mô hình đoán mọi mẫu là “bình thường” vẫn có thể đạt accuracy cao).
- **Precision và Recall:** Precision đo tỷ lệ báo động đúng trong số các cảnh báo được đưa ra; Recall đo tỷ lệ tấn công thật sự được phát hiện trong số toàn bộ tấn công thật sự tồn tại. Trong bối cảnh bảo mật, bỏ sót tấn công (Recall thấp) nguy hiểm hơn nhiều so với báo động nhầm (Precision thấp).
- **F1-macro:** chỉ số chính được dùng để so sánh và lựa chọn mô hình tốt nhất trong toàn bộ đề tài, do đây là trung bình không trọng số của F1-score từng lớp, giúp phản ánh công bằng hiệu quả trên cả các lớp tấn công hiếm.
- **AUC-ROC và MCC (Matthews Correlation Coefficient):** hai chỉ số bổ sung, đặc biệt MCC được xem là chỉ số đáng tin cậy nhất khi dữ liệu mất cân bằng nghiêm trọng.

### 6.2. Thí nghiệm 1 — Đánh giá trong phạm vi một tập dữ liệu (Within-Dataset)

Thí nghiệm này đánh giá bốn mô hình (Random Forest, XGBoost, GCN, GAT) trên cả hai bộ dữ liệu, với tập huấn luyện và tập kiểm thử đều thuộc cùng một nguồn thu thập. Đây là bước đánh giá cơ sở, trả lời câu hỏi nghiên cứu thứ nhất về hiệu quả phân loại thuần tuý, chưa xét đến khả năng tổng quát hoá.

### 6.3. Thí nghiệm 2 — Đánh giá chéo tập dữ liệu (Cross-Dataset)

Thí nghiệm quan trọng nhất về mặt khoa học, thực hiện bốn tổ hợp huấn luyện và kiểm thử: huấn luyện trên CICIDS2017 kiểm thử trên CICIDS2017 (đối chứng), huấn luyện trên CICIDS2017 kiểm thử trên UNSW-NB15 (kiểm tra tổng quát hoá), và tương tự theo chiều ngược lại. Kết quả kỳ vọng: mô hình GNN duy trì hiệu quả ổn định hơn khi chuyển sang tập dữ liệu khác so với mô hình học máy truyền thống, do GNN học được các mẫu hình cấu trúc mang tính bản chất của hành vi tấn công (ví dụ hình sao đặc trưng của DDoS) thay vì học các ngưỡng đặc trưng cụ thể dễ bị thay đổi giữa các môi trường mạng khác nhau.

### 6.4. Thí nghiệm 6 — Kiểm tra trên môi trường lab và so sánh với Suricata (thực hiện có điều kiện)

Trong trường hợp môi trường lab được thiết lập (theo khung quyết định tại mục 4.3), thí nghiệm này đánh giá mô hình đã huấn luyện trên dữ liệu công khai khi áp dụng lên dữ liệu thu thập tại môi trường lab — dữ liệu mà mô hình chưa từng tiếp xúc trong bất kỳ giai đoạn nào của quá trình huấn luyện. Song song, hệ thống Suricata với bộ luật ET Open Rules được chạy trên cùng dữ liệu lưu lượng để so sánh trực tiếp số lượng phát hiện đúng (True Positive), báo động nhầm (False Positive) và bỏ sót (False Negative) giữa hai cách tiếp cận — học máy dựa trên đồ thị và phát hiện dựa trên luật cố định.

### 6.5. Kiểm định thống kê

Nhằm đảm bảo các kết luận về sự khác biệt hiệu quả giữa các mô hình có ý nghĩa thống kê chứ không phải do ngẫu nhiên, đề tài áp dụng kiểm định McNemar cho ít nhất hai cặp mô hình quan trọng nhất: GAT so với Random Forest, và GAT so với GCN. Ngưỡng ý nghĩa thống kê được sử dụng là p-value nhỏ hơn 0,05.

## 7. Định hướng triển khai theo từng giai đoạn

Toàn bộ quá trình triển khai được tổ chức thành sáu giai đoạn tương ứng với sáu tuần thực hiện. Nguyên tắc xuyên suốt là: mỗi giai đoạn phải tạo ra một đầu ra kiểm chứng được (verifiable output) trước khi chuyển sang giai đoạn tiếp theo, và phần báo cáo tương ứng với nội dung đã hoàn thành phải được viết ngay trong cùng giai đoạn đó — không dồn việc viết báo cáo vào giai đoạn cuối, nhằm tránh thất thoát chi tiết kỹ thuật và số liệu thực nghiệm theo thời gian.

### 7.1. Giai đoạn 1 — Thiết lập nền tảng và chuẩn bị dữ liệu

Giai đoạn này tập trung vào ba nhóm công việc song song: thiết lập môi trường phát triển (viết code trên VS Code, thực thi và huấn luyện thông qua Google Colab — tận dụng GPU miễn phí, kết nối notebook Colab trực tiếp vào VS Code để vừa giữ được trải nghiệm chỉnh sửa mã nguồn quen thuộc vừa tận dụng tài nguyên tính toán của Colab; mount Google Drive ngay từ đầu để lưu trữ dữ liệu và checkpoint lâu dài, không phụ thuộc vào bộ nhớ tạm của phiên làm việc Colab), nghiên cứu lý thuyết nền tảng thông qua việc đọc và tóm tắt các bài báo khoa học cốt lõi (GCN, GAT, E-GraphSAGE), và thực hiện quy trình trích xuất — biến đổi — nạp dữ liệu (ETL) cho cả hai bộ dữ liệu công khai, với dữ liệu đã xử lý được lưu trực tiếp lên Google Drive để tái sử dụng cho các phiên làm việc sau. Về mặt tổ chức tài liệu, đây cũng là giai đoạn khung báo cáo chính thức được dựng lên với đầy đủ cấu trúc chương mục, để các giai đoạn sau chỉ cần điền nội dung vào đúng vị trí đã quy hoạch.

Đầu ra kiểm chứng được của giai đoạn: môi trường chạy không phát sinh lỗi phụ thuộc; hai bộ dữ liệu đã được làm sạch, chuẩn hoá và chia theo tỷ lệ 70/15/15, lưu ở định dạng Parquet để đảm bảo tốc độ đọc; tài liệu tóm tắt lý thuyết của các bài báo nền tảng; và phần Mở đầu cùng Chương 1 của báo cáo đã có bản nháp đầu tiên.

### 7.2. Giai đoạn 2 — Xây dựng biểu diễn đồ thị và quyết định về môi trường lab

Giai đoạn thứ hai gồm hai nhánh công việc có thể triển khai song song. Nhánh thứ nhất là xây dựng module chuyển đổi dữ liệu dạng bảng sang biểu diễn đồ thị (Graph Builder), áp dụng thống nhất cho cả hai bộ dữ liệu, kèm theo bộ kiểm thử đơn vị (unit test) để đảm bảo tính đúng đắn của cấu trúc đồ thị sinh ra. Nhánh thứ hai, được thực hiện có điều kiện theo khung ra quyết định tại mục 4.3, là việc thiết lập môi trường mạng thực nghiệm bằng VMware (nếu nhóm xác nhận có nhu cầu trình diễn tấn công theo thời gian thực) và tiến hành thu thập dữ liệu thông qua việc thực hiện lần lượt các kịch bản tấn công đã hoạch định.

Đầu ra kiểm chứng được: module xây dựng đồ thị hoạt động chính xác trên cả hai bộ dữ liệu với toàn bộ kiểm thử đơn vị đạt yêu cầu; và, nếu áp dụng Phương án A, một bộ dữ liệu thu thập tại môi trường lab với đầy đủ các lớp nhãn cần thiết và độ chính xác gán nhãn được xác nhận theo mốc thời gian thực hiện.

### 7.3. Giai đoạn 3 — Xây dựng và huấn luyện mô hình

Đây là giai đoạn trọng tâm về mặt kỹ thuật, tập trung toàn bộ nguồn lực vào việc hiện thực hoá và huấn luyện các mô hình đã hoạch định: Graph Convolutional Network, Graph Attention Network (kèm tối ưu siêu tham số tự động thông qua Optuna), và hai mô hình đối chứng Random Forest cùng XGBoost. Toàn bộ quá trình huấn luyện được thực hiện trên Google Colab (tận dụng GPU miễn phí để rút ngắn đáng kể thời gian huấn luyện so với chạy trên CPU cục bộ) và được theo dõi, ghi nhận có hệ thống thông qua nền tảng MLflow, với thư mục theo dõi (tracking directory) trỏ vào Google Drive đã mount để không bị mất dữ liệu khi phiên làm việc Colab kết thúc. Do Colab có giới hạn thời gian phiên làm việc và có thể ngắt kết nối đột ngột, trọng số mô hình được lưu thành checkpoint ngay sau mỗi epoch thay vì chỉ lưu khi huấn luyện hoàn tất, đảm bảo mọi kết quả thực nghiệm đều có thể truy vết và khôi phục lại được.

Đầu ra kiểm chứng được: các tệp trọng số mô hình đã huấn luyện hoàn chỉnh cho cả bốn mô hình trên cả hai bộ dữ liệu; nhật ký thực nghiệm trên MLflow với số lượng lượt chạy tối thiểu đã quy định; và phần nội dung lý thuyết cùng kiến trúc mô hình trong báo cáo được cập nhật ngay sau khi từng mô hình huấn luyện xong, đảm bảo các tham số được ghi nhận là tham số thực tế đã sử dụng, không phải tham số lý thuyết trích từ bài báo gốc.

### 7.4. Giai đoạn 4 — Thực nghiệm và giải thích mô hình

Giai đoạn này triển khai đầy đủ các thí nghiệm đã thiết kế tại mục 6 (Thí nghiệm 1, Thí nghiệm 2, và Thí nghiệm 6 nếu áp dụng), đồng thời tích hợp module giải thích mô hình GNNExplainer cho kiến trúc cho kết quả tốt nhất. Việc hoàn thiện GraphSAGE, nếu còn đủ thời gian sau khi hoàn tất GCN và GAT, cũng được thực hiện trong giai đoạn này.

Đầu ra kiểm chứng được: các bảng kết quả thực nghiệm đầy đủ số liệu, không có ô trống; các trường hợp minh hoạ (case study) về khả năng giải thích của mô hình; và phần Thực nghiệm trong báo cáo được viết ngay khi từng thí nghiệm hoàn tất, với số liệu và nhận xét được ghi nhận trong lúc thông tin còn mới, tránh sai lệch do phải hồi tưởng lại ở giai đoạn sau.

### 7.5. Giai đoạn 5 — Xây dựng hệ thống minh hoạ và triển khai

Giai đoạn này tập trung vào việc hiện thực hoá một hệ thống minh hoạ hoàn chỉnh, bao gồm một dịch vụ API cung cấp chức năng dự đoán và giải thích, cùng một giao diện giám sát trực quan cho phép người dùng theo dõi kết quả phát hiện. Hệ thống được đóng gói và triển khai lên hạ tầng đám mây để có địa chỉ truy cập công khai phục vụ buổi báo cáo. Cần nhấn mạnh lại ranh giới đã nêu tại mục 4.4: hạ tầng đám mây chỉ đóng vai trò lưu trữ và phục vụ hệ thống đã huấn luyện xong, không được sử dụng để thực hiện bất kỳ hoạt động tấn công nào, đặc biệt là các kịch bản liên quan đến từ chối dịch vụ.

Đầu ra kiểm chứng được: dịch vụ API và giao diện giám sát hoạt động ổn định; hệ thống có địa chỉ truy cập công khai; và phần thiết kế hệ thống trong báo cáo được hoàn thiện với hình ảnh chụp màn hình thực tế từ hệ thống đã triển khai, không sử dụng hình ảnh minh hoạ.

### 7.6. Giai đoạn 6 — Hoàn thiện báo cáo và chuẩn bị bảo vệ

Do phần lớn nội dung báo cáo đã được viết song song trong suốt năm giai đoạn trước, giai đoạn cuối cùng chỉ còn lại các công việc mang tính rà soát và hoàn thiện: đối chiếu toàn bộ số liệu trình bày trong phần thực nghiệm với dữ liệu kết quả thô để đảm bảo khớp tuyệt đối; hoàn thiện các phần còn thiếu như mục lục, danh mục hình vẽ, danh mục bảng biểu, danh mục tài liệu tham khảo; viết phần kết luận tổng kết những gì đã đạt được, những hạn chế còn tồn tại, và hướng phát triển tiếp theo; chuẩn bị và diễn tập bài trình bày; và chuẩn bị phương án dự phòng cho phần trình diễn hệ thống.

## 8. Quản lý rủi ro

Việc nhận diện sớm các rủi ro tiềm ẩn và chuẩn bị phương án dự phòng tương ứng là một phần không thể tách rời của định hướng triển khai. Bảng dưới đây tổng hợp các rủi ro chính đã được nhận diện, kèm theo đánh giá về khả năng xảy ra, mức độ ảnh hưởng, và phương án xử lý.

| **Rủi ro** | **Khả năng** | **Ảnh hưởng** | **Phương án xử lý** |
| --- | --- | --- | --- |
| Mô hình GNN cho kết quả kém hơn mô hình học máy truyền thống ở Thí nghiệm 1 | Cao | Thấp | Không xem là thất bại — là kết luận khoa học có giá trị. Lập luận dựa trên kết quả Thí nghiệm 2 (khả năng tổng quát hoá), nơi ưu thế của GNN được kỳ vọng thể hiện rõ hơn. |
| Phiên làm việc Google Colab bị ngắt kết nối hoặc hết thời gian sử dụng GPU miễn phí giữa lúc đang huấn luyện | Trung bình | Cao | Lưu checkpoint mô hình sau mỗi epoch (không đợi huấn luyện xong mới lưu), lưu trực tiếp vào Google Drive đã mount thay vì bộ nhớ tạm của Colab; theo dõi hạn mức GPU còn lại, ưu tiên chạy các lượt huấn luyện quan trọng trước. |
| Dữ liệu và kết quả thực nghiệm (MLflow, checkpoint) bị mất khi phiên Colab kết thúc do lưu nhầm vào bộ nhớ tạm thay vì Google Drive | Thấp nếu tuân thủ quy trình | Cao | Toàn bộ dữ liệu ETL, checkpoint mô hình, và thư mục theo dõi MLflow đều trỏ vào đường dẫn trong Google Drive đã mount, không lưu ở thư mục tạm cục bộ của Colab. |
| Môi trường lab (nếu triển khai) gặp sự cố kỹ thuật khi bắt lưu lượng mạng | Thấp đến trung bình | Cao nếu xảy ra gần thời điểm báo cáo | Chuẩn bị phương án ghi hình dự phòng cho phần trình diễn; kiểm tra và diễn tập môi trường lab trước thời điểm báo cáo tối thiểu hai lần. |
| Vi phạm chính sách sử dụng dịch vụ của nhà cung cấp đám mây nếu thực hiện tấn công từ chối dịch vụ trên hạ tầng đám mây | Thấp nếu tuân thủ định hướng | Rất cao (có thể dẫn đến đình chỉ tài khoản) | Tuân thủ tuyệt đối ranh giới đã nêu tại mục 4.4: mọi hoạt động tấn công thực hiện tại môi trường lab cục bộ, hạ tầng đám mây chỉ phục vụ lưu trữ và trình diễn. |
| Số liệu trong báo cáo không khớp với kết quả thực nghiệm thực tế do viết báo cáo dồn vào giai đoạn cuối | Trung bình nếu không tuân thủ định hướng viết song song | Cao | Áp dụng nguyên tắc viết báo cáo song song với tiến độ kỹ thuật đã trình bày tại mục 7; thực hiện đối chiếu số liệu ở giai đoạn cuối. |

## 9. Kết luận

Tài liệu này đã trình bày một cách có hệ thống toàn bộ quá trình phân tích đề tài nghiên cứu và ứng dụng mạng nơ-ron đồ thị trong phát hiện xâm nhập mạng, từ việc xác định loại hình và phương pháp nghiên cứu, phân tích phạm vi và câu hỏi nghiên cứu, đến việc xây dựng cơ sở lựa chọn dữ liệu, thiết kế kiến trúc kỹ thuật, quy trình thực nghiệm, và định hướng triển khai cụ thể theo từng giai đoạn.

Một số nguyên tắc xuyên suốt cần được ghi nhớ trong suốt quá trình triển khai: mọi quyết định kỹ thuật đều phải xuất phát từ một câu hỏi nghiên cứu cụ thể; dữ liệu dùng để đánh giá khả năng tổng quát hoá không bao giờ được sử dụng trong quá trình huấn luyện; báo cáo được viết song song với tiến độ kỹ thuật thay vì dồn vào giai đoạn cuối; và mọi hoạt động mang tính tấn công mạng, dù phục vụ mục đích nghiên cứu hay trình diễn, đều phải được thực hiện trong phạm vi môi trường có kiểm soát, tuân thủ đầy đủ các ranh giới kỹ thuật và pháp lý đã được xác lập.

Tài liệu này được xem là điểm tham chiếu xuyên suốt quá trình thực hiện đề tài, và có thể được điều chỉnh, cập nhật khi có những thay đổi phát sinh trong thực tế triển khai — miễn là mọi thay đổi đều được ghi nhận và có căn cứ rõ ràng, nhất quán với các nguyên tắc đã trình bày.
