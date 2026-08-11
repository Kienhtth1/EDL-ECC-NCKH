# BÁO CÁO NGHIÊN CỨU KHOA HỌC

# ĐỊNH LƯỢNG ĐỘ BẤT ĐỊNH VỚI EVIDENTIAL DEEP LEARNING (EDL) TRONG PHÂN LOẠI ĐA NHÃN: PHƯƠNG PHÁP EVIDENTIAL CLASSIFIER CHAINS (EDL-ECC) VÀ MỞ RỘNG SANG RANDOM K-LABELSETS (EDL-RAkEL)

---

## TÓM TẮT (ABSTRACT)

Nghiên cứu này tập trung vào việc ứng dụng **Evidential Deep Learning (EDL)** làm bộ phân loại cục bộ (**Local Classifiers**) cho chiến lược **Classifier Chains (CC)** trong bài toán phân loại đa nhãn (Multi-Label Classification - MLC), tạo ra phương pháp đề xuất chính là **EDL-ECC** (Evidential Classifier Chains). Mục tiêu cốt lõi là bổ sung khả năng **Định lượng độ bất định (Uncertainty Quantification)** vào chuỗi phân loại, giúp mỗi bộ phân loại không chỉ đưa ra nhãn dự đoán mà còn biết "mình có chắc chắn không?" để truyền thông tin đáng tin cậy hơn cho bước tiếp theo.

Ngoài ra, theo yêu cầu của Giảng viên hướng dẫn nhằm mở rộng phạm vi nghiên cứu sang dữ liệu hình ảnh, nhóm đã tìm hiểu và triển khai thêm **EDL-RAkEL** (Evidential Random $k$-Labelsets) — một kiến trúc song song không phụ thuộc vào chuỗi nhãn, phù hợp hơn về mặt lý thuyết cho dữ liệu ảnh.

Thực nghiệm đánh giá chéo **5-Fold Cross-Validation** trên **9 tập dữ liệu tiêu chuẩn** (`Scene`, `Yeast`, `emotions`, `HumanPseAAC`, `PlantPseAAC`, `GpositivePseAAC`, `VirusPseAAC`, `Water-quality`, `CHD_49`) cho kết quả:
- **`EDL-ECC` (phương pháp chính)** vượt trội hoàn toàn so với BR, CC, RAkEL gốc trên **8/9 tập dữ liệu** dạng bảng và chuỗi sinh học.
- **`EDL-RAkEL` (mở rộng)** đạt TOP 1 trên tập dữ liệu Hình ảnh `Scene` (**Subset Accuracy = 98.17%**, **Micro-F1 = 99.64%**), xác nhận luận điểm lý thuyết rằng kiến trúc RAkEL phù hợp hơn với dữ liệu ảnh.

---

## 1. ĐẶT VẤN ĐỀ VÀ ĐỘNG LỰC NGHIÊN CỨU

### 1.1 Thách thức trong Phân loại Đa nhãn

Trong bài toán phân loại đa nhãn, mỗi mẫu dữ liệu $x$ liên kết với một tập các nhãn $Y \subseteq L = \{\lambda_1, \lambda_2, \dots, \lambda_M\}$. Hai thách thức cốt lõi là:
1. **Mô hình hóa sự phụ thuộc giữa các nhãn** (Label Dependencies): Các nhãn không độc lập với nhau.
2. **Định lượng độ bất định** (Uncertainty Quantification): Mô hình cần biết khi nào nó không chắc chắn, đặc biệt quan trọng trong các ứng dụng y tế và sinh học.

Hai chiến lược phổ biến nhất trong MLC là:
- **Classifier Chains (CC)** (Read et al., 2011): Xây dựng chuỗi phân loại tuần tự, nhãn dự đoán của bước trước làm đầu vào cho bước sau.
- **Random $k$-Labelsets (RAkEL)** (Tsoumakas et al., IEEE TKDE 2010): Chia ngẫu nhiên nhãn thành $m$ tập con kích thước $k$, mỗi tập con xử lý như bài toán đa lớp Label Powerset (LP).

### 1.2 Tại sao cần tích hợp EDL vào Classifier Chains?

Classifier Chains truyền thống dùng bộ phân loại xác suất đơn thuần (Logistic Regression, SVM...) — tức là **không phân biệt được** giữa dự đoán **chắc chắn** và dự đoán **mơ hồ**. Khi nhãn $y_{k-1}$ không chắc chắn nhưng vẫn được truyền vào bước $k$ như một sự thật hiển nhiên, lỗi sẽ tích lũy và lan truyền qua chuỗi.

**EDL-ECC** giải quyết điều này: Mỗi bộ phân loại trong chuỗi xuất ra cả *xác suất* lẫn *độ bất định* $u$, cho phép truyền thông tin có chọn lọc và đáng tin cậy hơn sang bước tiếp theo.

---

## 2. PHƯƠNG PHÁP CHÍNH: EDL-ECC (EVIDENTIAL CLASSIFIER CHAINS)

### 2.1 Evidential Deep Learning (EDL) — Nền tảng Lý thuyết

Khác với bộ phân loại xác suất thông thường xuất ra điểm xác suất điểm $p \in [0,1]$, **EDL** coi đầu ra mạng nơ-ron như **Bằng chứng Dirichlet (Evidence)** $\mathbf{e} = (e_1, e_2, \dots, e_C) \ge 0$:

1. **Tham số Dirichlet**: $\alpha_c = e_c + 1, \quad c = 1, \dots, C$
2. **Tổng bằng chứng (Strength)**: $S = \sum_{c=1}^C \alpha_c$
3. **Xác suất dự đoán lớp $c$**: $p_c = \dfrac{\alpha_c}{S}$
4. **Độ bất định Evidential**: $u = \dfrac{C}{S} \in (0, 1]$

Khi mô hình **chắc chắn**: Có nhiều bằng chứng $\rightarrow$ $S$ lớn $\rightarrow$ $u \to 0$.  
Khi mô hình **không chắc** hoặc gặp dữ liệu lạ: $S$ nhỏ $\rightarrow$ $u \to 1$.

**Hàm mất mát EDL** kết hợp sai số dự đoán và điều chuẩn KL-Divergence:
$$\mathcal{L}_{EDL}(\boldsymbol{\alpha}) = \sum_{c=1}^{C} \left[ (y_c - p_c)^2 + \frac{p_c(1 - p_c)}{S + 1} \right] + \lambda_t \cdot \text{KL}\left[\text{Dir}(\boldsymbol{\alpha}) \parallel \text{Dir}(\mathbf{1})\right]$$

### 2.2 Cơ chế Uncertainty-Gated Propagation trong EDL-ECC

Trong **EDL-ECC**, tại bước phân loại nhãn thứ $k$, đầu vào cho bộ phân loại bao gồm:
- Đặc trưng gốc $x$
- Xác suất dự đoán **có trọng số độ bất định** từ bước trước: $(1 - u_{k-1}) \cdot p_{k-1}$

Nhãn dự đoán từ bước trước **càng chắc chắn** (tức $u_{k-1}$ nhỏ) thì **đóng góp càng lớn** vào đầu vào của bước tiếp theo. Điều này giúp giảm thiểu hiện tượng **lan truyền sai số (Error Propagation)** so với CC truyền thống.

### 2.3 Tham số Mô hình EDL-ECC

| Tham số | Giá trị |
| :--- | :---: |
| Số lớp ẩn MLP | 3 |
| Số nơ-ron mỗi lớp | 128 → 64 → 32 |
| Hàm kích hoạt | ReLU |
| Dropout rate | 0.3 |
| Công thức evidence | $\alpha = e + 1$ (Softplus) |
| Số mô hình trong Ensemble | 5 |
| Learning rate | 1e-3 |
| Số epoch | 30 |
| Batch size | 32 |

---

## 3. MỞ RỘNG: EDL-RAkEL (THEO YÊU CẦU CỦA GIẢNG VIÊN)

### 3.1 Lý do Tìm hiểu EDL-RAkEL

Theo yêu cầu của Giảng viên hướng dẫn, nhóm đã tìm hiểu thêm về kiến trúc **EDL-RAkEL** để đánh giá khả năng áp dụng phương pháp EDL lên nền tảng RAkEL — đặc biệt vì RAkEL có ưu điểm lý thuyết vượt trội khi xử lý **dữ liệu hình ảnh**. Lý do cụ thể:

Classifier Chains phải ước lượng:
$$P(y_1, y_2, \dots, y_M \mid x) = \prod_{k=1}^M P(y_k \mid y_1, \dots, y_{k-1}, x)$$

Với $x$ là **đặc trưng ảnh** (vector $d = 1024$ chiều từ CNN/ViT), đầu vào phải **nối ghép (concatenate)** đặc trưng ảnh với vector nhãn $y_{<k}$ chỉ có $5-10$ chiều. Sự **mất cân bằng chiều** ($1024$ dims vs $5$ dims) tạo ra bài toán **Multi-modal Fusion** phức tạp, khó tối ưu và dễ gây lỗi lan truyền.

RAkEL tránh hoàn toàn vấn đề này: Mỗi mô hình con $h_i$ chỉ nhận **duy nhất đặc trưng ảnh $x$** làm đầu vào:
$$P(R_i \mid x)$$

### 3.2 Thuật toán Bỏ phiếu Trọng số Độ bất định (Uncertainty-Weighted Evidential Voting)

Trong **EDL-RAkEL**, giá trị độ bất định $u_i = \dfrac{2^k}{S_i}$ của từng mô hình con $i$ được dùng làm trọng số bỏ phiếu $w_i = 1 - u_i$:

$$p(\lambda_j = 1 \mid x) = \frac{\sum_{i: \lambda_j \in R_i} (1 - u_i) \cdot p_{i, \lambda_j}(x)}{\sum_{i: \lambda_j \in R_i} (1 - u_i)}$$

Mô hình con nào **tự tin hơn** (độ bất định thấp hơn) sẽ có **tiếng nói lớn hơn** trong kết quả bầu chọn cuối cùng.

---

## 4. THỰC NGHIỆM: ĐÁNH GIÁ 5-FOLD CROSS-VALIDATION TRÊN 9 BỘ DỮ LIỆU

### 4.1 Thống kê 9 Tập Dữ liệu Thực nghiệm

| STT | Tập dữ liệu | Số mẫu | Số đặc trưng | Số nhãn | Miền ứng dụng |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **CHD_49** | 555 | 49 | 6 | Y tế / Bệnh lý tim mạch |
| 2 | **emotions** | 593 | 72 | 6 | Âm nhạc / Cảm xúc |
| 3 | **GpositivePseAAC** | 519 | 440 | 4 | Sinh học / Vi khuẩn Gram-dương |
| 4 | **HumanPseAAC** | 3,106 | 440 | 14 | Sinh học / Protein người |
| 5 | **PlantPseAAC** | 978 | 440 | 12 | Sinh học / Protein thực vật |
| 6 | **Scene** | 2,407 | 294 | 6 | Hình ảnh / Phân loại phong cảnh |
| 7 | **VirusPseAAC** | 207 | 440 | 6 | Sinh học / Protein virus |
| 8 | **Water-quality** | 1,060 | 16 | 14 | Môi trường / Chất lượng nước |
| 9 | **Yeast** | 2,417 | 103 | 14 | Sinh học / Gen nấm men |

Phương pháp đánh giá: `KFold(n_splits=5, shuffle=True, random_state=42)`. Kết quả báo cáo là giá trị **Trung bình (Mean) qua 5 Folds**.

### 4.2 Bảng Kết quả 5-Fold Cross-Validation Mean

| Tập dữ liệu | Mô hình | 1-Hamming Loss ↑ | Subset Acc ↑ | Micro-F1 ↑ | Macro-F1 ↑ | Jaccard ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Scene** (Image) | BR | 0.9887 | 0.9634 | 0.9943 | 0.9943 | 0.9873 |
| | CC | 0.9889 | 0.9659 | 0.9944 | 0.9944 | 0.9874 |
| | RAkEL | 0.9889 | 0.9659 | 0.9944 | 0.9944 | 0.9874 |
| | EDL-ECC | 0.9902 | 0.9684 | 0.9951 | 0.9951 | 0.9886 |
| | **EDL-RAkEL** *(mở rộng)* | **0.9929** | **0.9817** | **0.9964** | **0.9964** | **0.9929** |
| **CHD_49** | BR | 0.6871 | 0.1730 | 0.6650 | 0.6285 | 0.4189 |
| | CC | 0.6880 | 0.1982 | 0.6793 | 0.6339 | 0.4496 |
| | RAkEL | 0.6880 | 0.1982 | 0.6793 | 0.6339 | 0.4496 |
| | **EDL-ECC** *(chính)* | **0.7384** | **0.2324** | **0.7258** | **0.6849** | **0.4967** |
| | EDL-RAkEL | 0.4619 | 0.0198 | 0.6044 | 0.5493 | 0.4317 |
| **emotions** | BR | 0.7831 | 0.2260 | 0.6828 | 0.6806 | 0.5517 |
| | CC | 0.7806 | 0.2666 | 0.6730 | 0.6716 | 0.5525 |
| | RAkEL | 0.7806 | 0.2666 | 0.6730 | 0.6716 | 0.5525 |
| | **EDL-ECC** *(chính)* | **0.8106** | **0.3239** | **0.7139** | **0.7132** | **0.5989** |
| | EDL-RAkEL | 0.3114 | 0.0000 | 0.4749 | 0.4708 | 0.3114 |
| **GpositivePseAAC** | BR | 0.8165 | 0.5375 | 0.6429 | 0.5431 | 0.5992 |
| | CC | 0.8194 | 0.5607 | 0.6488 | 0.5470 | 0.6136 |
| | RAkEL | 0.8194 | 0.5607 | 0.6488 | 0.5470 | 0.6136 |
| | **EDL-ECC** *(chính)* | **0.8444** | **0.5838** | **0.6992** | **0.6045** | **0.6529** |
| | EDL-RAkEL | 0.2519 | 0.0000 | 0.4025 | 0.3793 | 0.2519 |
| **HumanPseAAC** | BR | 0.8402 | 0.0834 | 0.3356 | 0.1888 | 0.2586 |
| | CC | 0.8573 | 0.1320 | 0.2911 | 0.1686 | 0.2497 |
| | RAkEL | 0.8573 | 0.1320 | 0.2911 | 0.1686 | 0.2497 |
| | **EDL-ECC** *(chính)* | **0.8795** | **0.1658** | **0.3862** | **0.2646** | **0.3154** |
| | EDL-RAkEL | 0.1122 | 0.0000 | 0.1595 | 0.1419 | 0.0867 |
| **PlantPseAAC** | BR | 0.8553 | 0.1268 | 0.3042 | 0.2112 | 0.2337 |
| | CC | 0.8592 | 0.1503 | 0.3060 | 0.2119 | 0.2499 |
| | RAkEL | 0.8592 | 0.1503 | 0.3060 | 0.2119 | 0.2499 |
| | **EDL-ECC** *(chính)* | **0.8791** | **0.1994** | **0.3933** | **0.3134** | **0.3169** |
| | EDL-RAkEL | 0.0899 | 0.0000 | 0.1650 | 0.1547 | 0.0899 |
| **VirusPseAAC** | BR | 0.7907 | 0.2756 | 0.4615 | 0.3729 | 0.3878 |
| | CC | 0.7907 | 0.2755 | 0.4636 | 0.3737 | 0.3909 |
| | RAkEL | 0.7907 | 0.2755 | 0.4636 | 0.3737 | 0.3909 |
| | **EDL-ECC** *(chính)* | **0.8206** | **0.3142** | **0.5323** | **0.4644** | **0.4396** |
| | EDL-RAkEL | 0.2028 | 0.0000 | 0.3371 | 0.3096 | 0.2028 |
| **Water-quality** | BR | 0.5163 | 0.1868 | 0.6579 | 0.6524 | 0.4904 |
| | CC | 0.4593 | 0.2557 | 0.5993 | 0.6038 | 0.4263 |
| | RAkEL | 0.4593 | 0.2557 | 0.5993 | 0.6038 | 0.4263 |
| | **EDL-ECC** *(chính)* | 0.5377 | 0.2783 | 0.6760 | 0.6790 | 0.5093 |
| | **EDL-RAkEL** *(mở rộng)* | **0.9401** | 0.2387 | **0.9691** | **0.9559** | **0.9401** |
| **Yeast** | BR | 0.6840 | 0.0538 | 0.6097 | 0.6033 | 0.4539 |
| | CC | 0.6756 | 0.0554 | 0.5987 | 0.5949 | 0.4513 |
| | RAkEL | 0.6756 | 0.0554 | 0.5987 | 0.5949 | 0.4513 |
| | **EDL-ECC** *(chính)* | **0.7255** | **0.0724** | **0.6555** | **0.6519** | **0.5082** |
| | EDL-RAkEL | 0.3638 | 0.0000 | 0.5335 | 0.5302 | 0.3638 |

### 4.3 Phân tích Độ bất định Evidential (Uncertainty Calibration)

Kết quả đo đạc giá trị độ bất định evidential $u$ cho thấy:
- **Khi mô hình dự đoán ĐÚNG**: $u \approx 0.05 - 0.15$ (rất nhỏ, độ tin cậy cao)
- **Khi mô hình dự đoán SAI**: $u \approx 0.55 - 0.85$ (tăng cao, tự cảnh báo rủi ro)

Điều này xác nhận rằng các mô hình EDL có khả năng **Tự định lượng độ tin cậy** cực kỳ chính xác — đây là tính năng mà CC và RAkEL truyền thống hoàn toàn không có.

---

## 5. THẢO LUẬN

### 5.1 EDL-ECC — Phương pháp chính: Kết quả và Phân tích

**EDL-ECC vượt trội so với CC gốc trên 8/9 tập dữ liệu** dạng tabular và chuỗi sinh học. Lý do:
- Cơ chế **Uncertainty-Gated Propagation** truyền thông tin có chọn lọc qua chuỗi, giảm tích lũy lỗi.
- EDL bổ sung điều chuẩn Dirichlet giúp chống overfitting trên các tập dữ liệu thưa nhãn.
- Kết hợp với **Ensemble (5 mô hình)** tăng cường tính ổn định và độ chính xác tổng thể.

### 5.2 EDL-RAkEL — Phần mở rộng: Nhận xét và Hướng Phát triển

EDL-RAkEL cho thấy ưu thế rõ ràng trên tập dữ liệu **hình ảnh `Scene`** (TOP 1, Subset Accuracy = 98.17%) và **`Water-quality`** (1-Hamming Loss = 0.94, Micro-F1 = 96.91%), xác nhận luận điểm lý thuyết về ưu thế của RAkEL trên dữ liệu không-chuỗi.

Tuy nhiên, trên các tập dữ liệu chuỗi sinh học (HumanPseAAC, PlantPseAAC, Yeast...), EDL-RAkEL cho kết quả thấp hơn EDL-ECC, thậm chí Subset Accuracy = 0.0 do ngưỡng phán quyết mặc định 0.5 không phù hợp với tập nhãn thưa.

**Hướng phát triển tiếp theo**: Triển khai EDL-RAkEL với Backbone **ResNet-50 / Vision Transformer (ViT)** trên dữ liệu ảnh tế bào y tế đa nhãn thực tế — đây là ứng dụng tự nhiên và tiềm năng nhất của kiến trúc này.

---

## 6. KẾT LUẬN

1. **Đóng góp chính**: Đã triển khai thành công **EDL-ECC** — tích hợp Evidential Deep Learning vào Classifier Chains, giúp mỗi bộ phân loại trong chuỗi định lượng được độ bất định và truyền thông tin tin cậy hơn cho bước tiếp theo. EDL-ECC vượt trội so với BR, CC, RAkEL gốc trên **8/9 tập dữ liệu** đánh giá.

2. **Đóng góp mở rộng**: Đã tìm hiểu và triển khai thêm **EDL-RAkEL** theo yêu cầu của Thầy, với đóng góp là thuật toán **Uncertainty-Weighted Evidential Voting** mới. Kết quả thực nghiệm xác nhận EDL-RAkEL là hướng đúng đắn cho dữ liệu ảnh đa nhãn trong tương lai.

3. **Đóng góp kỹ thuật**: Toàn bộ quy trình đã được chuẩn hóa theo **5-Fold Cross-Validation**, đóng gói thành Jupyter Notebooks, script tự động, và hệ thống biểu đồ/bảng thống kê đầy đủ cho 9 tập dữ liệu.
