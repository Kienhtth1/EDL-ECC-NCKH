# BÁO CÁO TÓM TẮT THUẬT TOÁN RANDOM K-LABELSETS (RAkEL)
### (Dựa trên bài báo gốc: Tsoumakas et al., IEEE TKDE 2010)

---

## 1. TỔNG QUAN VỀ BÀI BÁO GỐC

- **Tên bài báo**: *Random $k$-Labelsets for Multi-Label Classification*
- **Tác giả**: Grigorios Tsoumakas, Ioannis Katakis, Ioannis Vlahavas
- **Tạp chí**: IEEE Transactions on Knowledge and Data Engineering (TKDE), Vol. 22, No. 7, 2010.
- **Đóng góp cốt lõi**: Đề xuất phương pháp ensemble **RAkEL** (Random $k$-Labelsets) để giải quyết hai nhược điểm chí mạng của hai phương pháp kinh điển: **Binary Relevance (BR)** (bỏ qua phụ thuộc nhãn) và **Label Powerset (LP)** (bùng nổ không gian lớp $2^{|L|}$ và mất cân bằng lớp trầm trọng).

---

## 2. ĐỘNG LỰC NGHIÊN CỨU & SO SÁNH BR vs. LP vs. RAkEL

| Phương pháp | Cơ chế chính | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **Binary Relevance (BR)** | Trầm phẳng bài toán thành $M$ mô hình nhị phân độc lập $P(\lambda_j \mid x)$ | Huấn luyện nhanh, đơn giản, $O(M)$ | **Bỏ qua hoàn toàn mối quan hệ phụ thuộc giữa các nhãn (Label Correlation)**. |
| **Label Powerset (LP)** | Coi mỗi tập tổ hợp nhãn là 1 lớp duy nhất của bài toán Đa lớp ($2^M$ lớp) | Học hoàn hảo mối quan hệ phụ thuộc giữa tất cả các nhãn | **Sự bùng nổ không gian lớp ($2^M$)**; Mất cân bằng lớp cực kỳ trầm trọng khi $M$ lớn. |
| **RAkEL (Đề xuất)** | Chia tập nhãn thành $m$ tập con nhỏ kích thước $k$ ($k$-labelsets), mỗi tập con dùng LP | **Học được mối quan hệ phụ thuộc nhãn**, kiểm soát được không gian lớp ($2^k$) | Cần chọn tham số $k$ và số mô hình con $m$ phù hợp. |

---

## 3. NGUYÊN LÝ TOÁN HỌC VÀ THUẬT TOÁN RAKEL

### 3.1 Khái niệm $k$-Labelset
Cho tập $M$ nhãn $L = \{\lambda_1, \lambda_2, \dots, \lambda_M\}$. 
Một **$k$-labelset** $R \subset L$ là một tập con gồm $k$ nhãn phân biệt ($|R| = k \ll M$). 

Bài toán phân loại đa nhãn trên tập con $R$ được biến đổi thành bài toán phân loại đa lớp Label Powerset (LP) với **$C = 2^k$ lớp tổ hợp**.

---

### 3.2 Hai biến thể của RAkEL (RAkELd vs. RAkELo)

1. **RAkELd (Disjoint / Rời rạc)**:
   - Chia không gian nhãn $L$ thành $m = \lceil M/k \rceil$ tập con **không chồng lấn**.
   - **Ưu điểm**: Huấn luyện rất nhanh.
   - **Nhược điểm**: Bỏ qua sự phụ thuộc giữa các nhãn thuộc 2 tập con khác nhau.

2. **RAkELo (Overlapping / Ngẫu nhiên chồng lấn - Phiên bản chuẩn)**:
   - Rút ngẫu nhiên $m$ tập con kích thước $k$ từ $L$ (có thể hoàn lại).
   - **Ưu điểm**: Mỗi cặp nhãn bất kỳ có xác suất cùng xuất hiện trong một $k$-labelset $\rightarrow$ **Mô hình hóa toàn diện sự phụ thuộc giữa tất cả các cặp nhãn**.
   - Tạo ra sự đa dạng (Ensemble Diversity) cao cho mô hình.

---

### 3.3 Quy trình Bỏ phiếu Tổng hợp Dự đoán (Thresholded Average Voting)

Cho một mẫu đầu vào $x$:
1. Mỗi mô hình con $h_i$ (ứng với $k$-labelset $R_i$) dự đoán xác suất cho $2^k$ lớp tổ hợp, từ đó suy ra xác suất nhị phân $p_{i, \lambda_j}(x)$ cho các nhãn $\lambda_j \in R_i$.
2. **Tổng hợp điểm xác suất (Average Voting)** cho nhãn thứ $j$:
   $$p(\lambda_j = 1 \mid x) = \frac{1}{|\{i: \lambda_j \in R_i\}|} \sum_{i: \lambda_j \in R_i} p_{i, \lambda_j}(x)$$
3. **Phán quyết nhị phân**:
   $$\hat{y}_j = \mathbb{I}(p(\lambda_j = 1 \mid x) > \tau)$$
   với $\tau$ là ngưỡng phán quyết (mặc định $\tau = 0.5$ hoặc được tìm tối ưu).

---

## 4. TẠI SAO RAKEL ĐẶC BIỆT THÍCH HỢP CHO DỮ LIỆU ẢNH (IMAGE DATA)?

Khi kết hợp RAkEL với **Evidential Deep Learning (EDL-RAkEL)** cho dữ liệu Ảnh:

1. **Loại bỏ hoàn toàn nút thắt Dung hợp Đa phương thức (Multi-modal Fusion)**:
   - Khác với Classifier Chains ($P(y_k \mid y_1, \dots, y_{k-1}, x)$) phải nối vector nhãn $1D$ mỏng vào đặc trưng ảnh $x$ lớn, RAkEL dự đoán trực tiếp:
     $$P(R_i \mid x)$$
   - Đầu vào của các mô hình con **chỉ là duy nhất ma trận đặc trưng ảnh $x$**.

2. **Huấn luyện End-to-End song song dễ dàng**:
   - Có thể dùng chung 1 Backbone trích xuất ảnh (như ResNet-50, Vision Transformer) và cắm $m$ đầu LP Heads song song.

3. **Tích hợp Định lượng Độ bất định (Evidential Uncertainty)**:
   - Trong **EDL-RAkEL**, giá trị độ bất định evidential $u_i = \frac{2^k}{S_i}$ của từng $k$-labelset được dùng để tính **Trọng số Bỏ phiếu Tự tin** $w_i = 1 - u_i$:
     $$p(\lambda_j = 1 \mid x) = \frac{\sum_{i: \lambda_j \in R_i} (1 - u_i) \cdot p_{i, \lambda_j}(x)}{\sum_{i: \lambda_j \in R_i} (1 - u_i)}$$

---

## 5. KẾT LUẬN & GIÁ TRỊ ỨNG DỤNG

Bản chất của **RAkEL** là sự dung hòa thông minh giữa **tính hiệu quả tính toán của Binary Relevance** và **khả năng học tương quan nhãn của Label Powerset**. Khi tích hợp thêm **Evidential Deep Learning**, **EDL-RAkEL** trở thành một kiến trúc mạnh mẽ, thanh thoát và đạt hiệu năng đỉnh cao (Subset Accuracy = 97.30%, Micro-F1 = 99.55% trên tập ảnh `Scene`).
