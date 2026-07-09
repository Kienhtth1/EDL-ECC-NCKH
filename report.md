# A SURVEY AND IMPLEMENTATION OF MULTI-LABEL CLASSIFICATION: 
# TOWARDS THE INTEGRATION OF EVIDENTIAL DEEP LEARNING AND ENSEMBLE CLASSIFIER CHAINS
**(Khảo sát và Triển khai Phân loại Đa nhãn: Tích hợp Evidential Deep Learning và Ensemble Classifier Chains)**

---

## 1. INTRODUCTION

### 1.1. Research motivation
Trong kỷ nguyên dữ liệu đa chiều, nhiều bài toán học có giám sát (supervised learning) không còn thỏa mãn giả định một thực thể chỉ thuộc về một lớp duy nhất. Một văn bản có thể đề cập đến nhiều chủ đề, một hình ảnh y tế có thể chứa nhiều loại bệnh lý cùng lúc. Các bài toán này thuộc nhóm Phân loại Đa nhãn (Multi-Label Classification - MLC), trong đó mỗi đầu vào $x \in \mathbb{R}^d$ được gán với một vector nhãn $y \in \{0, 1\}^K$.

MLC phải đối mặt với 3 thách thức lớn. Thứ nhất, không gian đầu ra có kích thước $2^K$, gây bùng nổ tổ hợp. Thứ hai, dữ liệu thường xuyên bị mất cân bằng nghiêm trọng (imbalanced data), với nhiều nhãn thiểu số hiếm khi xuất hiện. Thứ ba, các nhãn thường có sự phụ thuộc lẫn nhau (label dependencies).
Các phương pháp cổ điển như Binary Relevance (BR) bỏ qua sự phụ thuộc nhãn. Classifier Chains (CC) khai thác tốt sự phụ thuộc bằng cách học tuần tự, nhưng lại mắc phải nhược điểm chí mạng là **hiện tượng lan truyền sai số (error propagation)**. Trong các bài toán rủi ro cao hoặc dữ liệu mất cân bằng, một dự đoán sai ở đầu chuỗi sẽ làm hỏng toàn bộ các dự đoán phía sau.

Động lực của báo cáo này xuất phát từ việc tích hợp **Evidential Deep Learning (EDL)** vào **Ensemble Classifier Chains (ECC)**. EDL không chỉ đưa ra xác suất dự đoán mà còn định lượng được Độ bất định (Uncertainty). Sự tích hợp EDL-ECC trả lời cho câu hỏi trung tâm: *Làm thế nào một mô hình có thể khai thác sự phụ thuộc nhãn, đồng thời tự động cắt đứt chuỗi lan truyền sai số khi đối mặt với các nhãn có độ bất định cao do thiếu dữ liệu?*

### 1.2. Research objectives
Báo cáo này có 4 mục tiêu chính:
1) Hệ thống hóa các phương pháp cơ sở của MLC và các thách thức về mất cân bằng dữ liệu.
2) Làm rõ vai trò của Evidential Deep Learning trong việc định lượng độ bất định thông qua Thuyết logic chủ quan.
3) Đề xuất kiến trúc lai EDL-ECC (Evidential Ensemble Classifier Chains) để ngăn chặn lan truyền sai số.
4) Trình bày kết quả thực nghiệm chi tiết trên tập dữ liệu chuẩn (BibTeX) để so sánh hiệu năng.

### 1.3. Research object and scope
Đối tượng nghiên cứu là họ các phương pháp MLC liên quan đến sự phụ thuộc nhãn và định lượng bất định. Phạm vi lý thuyết bao gồm BR, CC, ECC, và EDL. Phạm vi thực nghiệm bao gồm tập dữ liệu đa nhãn BibTeX, các kỹ thuật phân tích EDA, hàm mất mát tùy chỉnh (Expected MSE + KL Divergence), và các thước đo chuẩn (Macro-F1, Micro-F1, Hamming Loss).

### 1.4. Contributions of the report
Báo cáo này đóng góp 3 điểm chính:
1) Đề xuất một khung phân tích kết hợp giữa mô hình hóa sự phụ thuộc nhãn (ECC) và nhận thức rủi ro (EDL).
2) Hiện thực hóa cơ chế **Uncertainty-Aware Propagation** (Truyền dẫn nhận thức bất định), nơi các nhãn bất định bị hạ trọng số trước khi truyền vào chuỗi tiếp theo.
3) Cung cấp kết quả thực nghiệm và trực quan hóa toàn diện trên tập dữ liệu BibTeX.

---

## 2. LITERATURE REVIEW

### 2.1. Theoretical basis of Multi-Label Classification
Theo ký hiệu chuẩn, một tập dữ liệu MLC được biểu diễn là $D = \{(x_i, y_i)\}_{i=1}^N$, với $x_i \in \mathbb{R}^d$ và $y_i \in \{0, 1\}^K$.
Hai thước đo thống kê quan trọng nhất là Label Cardinality (số lượng nhãn trung bình trên mỗi mẫu) và Label Density (mật độ nhãn).
Các phương pháp MLC được chia làm hai nhóm lớn: Problem Transformation (Biến đổi bài toán) và Algorithm Adaptation (Thích ứng thuật toán). BR là baseline phổ biến nhất do tính đơn giản nhưng lại bỏ qua sự phụ thuộc nhãn. 

### 2.2. Label Dependencies and Error Propagation in Classifier Chains
Classifier Chains (CC) xây dựng một chuỗi $K$ bộ phân loại nhị phân $h_1, h_2, \dots, h_K$. Bộ phân loại thứ $j$ nhận đặc trưng gốc và kết quả dự đoán của các nhãn trước đó làm đầu vào:
$$h_j: \mathbb{R}^d \times \{0, 1\}^{j-1} \to \{0, 1\}$$
CC khai thác tốt sự phụ thuộc nhãn với chi phí thấp. Tuy nhiên, nó bị giới hạn bởi sự lan truyền sai số. Nếu $h_1$ dự đoán sai do dữ liệu mất cân bằng, sai số này sẽ truyền đến $h_2, \dots, h_K$.

### 2.3. Theoretical Foundation of Evidential Deep Learning (EDL)
EDL dựa trên Thuyết Logic Chủ quan (Subjective Logic). Đầu ra của mạng neural không phải là xác suất (Softmax), mà là "bằng chứng" (evidence) $e_k \ge 0$ (thông qua hàm ReLU). 
Bằng chứng này được dùng để tham số hóa phân phối Dirichlet: $\alpha_k = e_k + 1$. 
Tổng sức mạnh của bằng chứng là $S = \sum_{k=1}^K \alpha_k$. 
Mức độ bất định (Uncertainty) được tính bằng:
$$u = \frac{K}{S}$$
Hệ quả là, khi mô hình gặp một dữ liệu OOD (Out-of-Distribution) hoặc nhãn thiểu số chưa từng học, tổng bằng chứng $S$ nhỏ, kéo theo độ bất định $u$ tiến dần về 1.

---

## 3. PROPOSED METHOD AND EXPERIMENTAL DESIGN

### 3.1. Overview of EDL-ECC Pipeline
Phương pháp được đề xuất (EDL-ECC) tích hợp 2 thành phần: Bộ học cơ sở EDL và Cấu trúc Ensemble Classifier Chains.
Pipeline gồm 4 giai đoạn:
1) **Tiền xử lý (Preprocessing):** Trích xuất Top-K nhãn thường xuyên, tạo PyTorch DataLoader.
2) **Huấn luyện EDL Base Learner:** Thay vì cross-entropy, mô hình tối ưu hóa hàm loss Dirichlet (Expected MSE + KL Divergence).
3) **Lắp ráp ECC:** Tạo ra các chuỗi phân loại ngẫu nhiên.
4) **Truyền dẫn có nhận thức (Uncertainty-Aware Propagation):** Vector truyền đi trong chuỗi không phải là giá trị nhị phân $\{0,1\}$, mà là cặp giá trị liên tục $[p_i, u_i]$ (xác suất và độ bất định).

### 3.2. Datasets and compared models
Thực nghiệm sử dụng tập dữ liệu **BibTeX**, một tập dữ liệu chuẩn mực trong MLC với độ mất cân bằng cao.
Các mô hình được so sánh bao gồm:
*   **Base 1 (BR-BCE):** Mạng phân loại độc lập dùng Binary Cross Entropy.
*   **Base 2 (ECC-BCE):** Mạng Classifier Chains tiêu chuẩn, dùng để đo lường mức độ lan truyền sai số.
*   **Proposed (EDL-ECC):** Chuỗi phân loại kết hợp truyền dẫn độ bất định.

### 3.3. Evaluation metrics
Vì dữ liệu mất cân bằng, Accuracy thông thường không được sử dụng. Báo cáo sử dụng:
*   **Hamming Loss ($\downarrow$):** Tỷ lệ các nhãn bị phân loại sai trên tổng số nhãn.
*   **Micro-F1 và Macro-F1 ($\uparrow$):** F1-score tính trên toàn cục và tính trung bình theo từng nhãn (Macro-F1 rất nhạy cảm với các nhãn hiếm).
*   **Uncertainty Calibration:** So sánh phân bố độ bất định giữa dự đoán Đúng và Sai.

---

## 4. EXPERIMENTAL RESULTS

### 4.1. Exploratory Data Analysis (EDA) Results
Phân tích dữ liệu BibTeX cho thấy:
*   **Label Imbalance:** Tồn tại sự chênh lệch khổng lồ giữa các nhãn phổ biến nhất và nhãn hiếm (< 1% xuất hiện).
*   **Label Correlation:** Ma trận tương quan (Correlation Heatmap) cho các Top Labels xác nhận nhiều cặp nhãn có hệ số đồng xuất hiện rất cao, biện minh cho sự cần thiết của ECC thay vì BR.

### 4.2. Training and Loss Optimization
Đồ thị Learning Curves cho thấy mô hình EDL hội tụ ổn định sau 50 epochs. Sự kết hợp của KL Divergence như một bộ điều chuẩn (regularizer) giúp Loss trên tập Validation không bị tăng vọt trở lại (tránh overfitting trên tập dữ liệu thưa thớt).

### 4.3. Uncertainty Distribution Analysis
Kết quả thực nghiệm trực quan hóa (Uncertainty Density Plot) chỉ ra đặc tính quan trọng nhất của EDL:
*   Đối với các mẫu mô hình **dự đoán đúng**, phân bố độ bất định tập trung rất cao ở gần mức 0 (mô hình tự tin).
*   Đối với các mẫu mô hình **dự đoán sai** (thường rơi vào nhãn hiếm), phân bố độ bất định dịch chuyển mạnh sang bên phải (gần 1). Mô hình có khả năng nhận thức được "sự thiếu hiểu biết" của mình.

### 4.4. Confusion Matrix and Metric Comparison
Trên tập Validation/Test, EDL-ECC cho thấy sự vượt trội ở chỉ số **Macro-F1** so với ECC-BCE truyền thống.
Tại các nút mạng tương ứng với nhãn hiếm, ECC-BCE bị nhiễu do các dự đoán sai từ trước truyền tới, làm hỏng hoàn toàn dự đoán. Ngược lại, EDL-ECC nhận được tín hiệu $u_i \approx 1$ từ các dự đoán sai đó, tự động bỏ qua tín hiệu nhiễu và dựa vào đặc trưng gốc $x$, giúp duy trì độ chính xác của các nhãn phía sau.

---

## 5. DISCUSSION AND RESEARCH GAP

### 5.1. Label independence versus label dependence
BR đơn giản và nhanh, nhưng bỏ qua cấu trúc. Việc lập mô hình phụ thuộc nhãn thông qua CC mang lại lợi thế lớn về F1-score nhưng tiềm ẩn rủi ro lây nhiễm chéo. Thực nghiệm chứng minh rằng *Label dependence chỉ thực sự an toàn khi đi kèm với Uncertainty Estimation*.

### 5.2. Error correction by Uncertainty Gating
Trong khi các phương pháp mã hóa sửa lỗi (Error-Correcting Codes) xử lý sai số bằng cách giải mã lại từ không gian dự đoán, EDL-ECC xử lý sai số ngay trong quá trình *forward pass* bằng cơ chế Soft-gating. Khi một nhãn không chắc chắn, sức mạnh của nó bị giảm thiểu trước khi trở thành đầu vào cho nhãn tiếp theo.

### 5.3. Limitations of current EDL-ECC
Báo cáo cũng ghi nhận 2 hạn chế:
1) **Chi phí tính toán (Computational Cost):** Việc huấn luyện phân phối Dirichlet qua hàm Expected MSE tốn nhiều thời gian và tài nguyên phần cứng hơn so với BCE thông thường.
2) **Phụ thuộc thứ tự (Order Sensitivity):** Dù ECC đã dùng ngẫu nhiên nhiều chuỗi, nhưng đối với các tập dữ liệu có hàng nghìn nhãn (Extreme MLC), việc tạo ensemble trở nên quá tải.

---

## 6. CONCLUSION AND FUTURE WORK

### 6.1. Main findings
Nghiên cứu này đã hệ thống hóa và đề xuất thành công quy trình kết hợp Evidential Deep Learning vào Ensemble Classifier Chains cho Phân loại Đa nhãn. Bằng cách định lượng độ bất định cho từng dự đoán đơn lẻ, kiến trúc EDL-ECC đã giải quyết thành công nút thắt lớn nhất của Classifier Chains: sự lan truyền sai số do dữ liệu mất cân bằng. Thực nghiệm trên tập BibTeX xác nhận rằng mô hình không chỉ đạt hiệu suất cao ở thước đo Macro-F1 mà còn cung cấp khả năng cảnh báo mức độ rủi ro (Uncertainty) minh bạch cho từng quyết định.

### 6.2. Future work
Định hướng nghiên cứu trong tương lai sẽ tập trung vào ba nhiệm vụ:
1) Tích hợp cơ chế **Partial Abstention (PA)**: Sử dụng trực tiếp độ bất định $u_i$ để mô hình chủ động từ chối dự đoán (abstain) đối với các nhãn rủi ro cao, tạo ra hệ thống phân loại đa nhãn an toàn (Safe MLC).
2) **Dynamic Ordering:** Sắp xếp lại thứ tự chuỗi động (Dynamic CC) theo thời gian thực dựa trên độ tự tin, các nhãn dễ đoán (Uncertainty thấp) sẽ được xếp đầu chuỗi.
3) Đánh giá trên các tập dữ liệu Extreme MLC quy mô lớn (ví dụ: EUR-Lex, AmazonCat) để kiểm chứng khả năng mở rộng của thuật toán.
