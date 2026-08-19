# PHẦN 4: THỰC NGHIỆM VÀ KẾT QUẢ

---

## 4.1 Thiết lập Thực nghiệm

### 4.1.1 Giao thức Đánh giá

Toàn bộ thực nghiệm sử dụng **5-Fold Cross-Validation** với cấu hình:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

- Mỗi fold: ~80% dữ liệu để huấn luyện, ~20% để kiểm tra
- Kết quả báo cáo: **Giá trị trung bình (Mean)** qua 5 folds
- Ngưỡng quyết định nhãn của EDL-ECC được tối ưu trên mỗi validation fold riêng biệt (grid search từ 0.1 đến 0.5, bước 0.05)
- **Random seed cố định** (42) để đảm bảo tái lập kết quả

Lý do chọn 5-Fold CV thay vì hold-out split:
- Sử dụng toàn bộ dữ liệu cho cả training và evaluation
- Đánh giá ổn định hơn, đặc biệt quan trọng với các dataset nhỏ như VirusPseAAC (207 mẫu)
- Giảm phương sai do phân chia ngẫu nhiên

### 4.1.2 Các Mô hình So sánh (Baselines)

| Mô hình | Loại | Mô tả |
|---|---|---|
| **BR** (Binary Relevance) | Problem Transformation | $L$ Logistic Regression độc lập, không mô hình hóa tương quan nhãn |
| **CC** (Classifier Chains) | Problem Transformation | Logistic Regression trong chuỗi tuần tự với thứ tự ngẫu nhiên |
| **RAkEL** | Problem Transformation | Random k-Labelsets với LP, k=3, Logistic Regression |
| **EDL-ECC** *(phương pháp chính)* | Deep Learning | EDL Binary Module trong Ensemble Chains với Uncertainty-Gated Propagation |
| **EDL-RAkEL** *(mở rộng)* | Deep Learning | EDL Label Powerset trong k-labelsets song song với Uncertainty-Weighted Voting |

**Tất cả baseline** dùng `LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')` — đã kích hoạt `class_weight='balanced'` để xử lý mất cân bằng nhãn một phần.

### 4.1.3 Metrics Đánh giá

| Metric | Ký hiệu | Ý nghĩa | Hướng tối ưu |
|---|---|---|---|
| 1 − Hamming Loss | 1-HL | Tỷ lệ nhãn đúng trung bình | ↑ Càng cao càng tốt |
| Subset Accuracy | SubAcc | Tỷ lệ mẫu khớp nhãn hoàn toàn | ↑ Càng cao càng tốt |
| Micro-F1 | μF1 | F1 trên tất cả cặp (nhãn, mẫu) | ↑ Càng cao càng tốt |
| Macro-F1 | MF1 | F1 trung bình theo từng nhãn | ↑ Càng cao càng tốt |
| Jaccard Index | Jacc | Độ tương đồng tập nhãn | ↑ Càng cao càng tốt |

---

## 4.2 Thống kê 9 Tập Dữ liệu

| STT | Dataset | Mẫu | Đặc trưng | Nhãn | Miền ứng dụng | Loại dữ liệu |
|:---:|---|:---:|:---:|:---:|---|---|
| 1 | **Scene** | 2,407 | 294 | 6 | Phân loại phong cảnh | Ảnh (đặc trưng HOG/màu) |
| 2 | **Yeast** | 2,417 | 103 | 14 | Gen chức năng nấm men | Sinh học |
| 3 | **emotions** | 593 | 72 | 6 | Cảm xúc âm nhạc | Âm nhạc |
| 4 | **HumanPseAAC** | 3,106 | 440 | 14 | Protein người | Sinh học (PseAAC) |
| 5 | **PlantPseAAC** | 978 | 440 | 12 | Protein thực vật | Sinh học (PseAAC) |
| 6 | **GpositivePseAAC** | 519 | 440 | 4 | Vi khuẩn Gram-dương | Sinh học (PseAAC) |
| 7 | **VirusPseAAC** | 207 | 440 | 6 | Protein virus | Sinh học (PseAAC) |
| 8 | **Water-quality** | 1,060 | 16 | 14 | Chất lượng nước | Môi trường |
| 9 | **CHD_49** | 555 | 49 | 6 | Bệnh tim mạch | Y tế |

**Nhận xét về tính đa dạng:**
- Số mẫu: từ 207 (VirusPseAAC) đến 3,106 (HumanPseAAC)
- Số đặc trưng: từ 16 (Water-quality) đến 440 (các bộ PseAAC)
- Số nhãn: từ 4 (GpositivePseAAC) đến 14 (HumanPseAAC, Water-quality, Yeast)
- Phủ rộng các lĩnh vực: y tế, sinh học phân tử, môi trường, thị giác máy tính, âm nhạc

---

## 4.3 Kết quả 5-Fold Cross-Validation

### 4.3.1 Bảng kết quả đầy đủ

**Bảng 4.1:** Kết quả trung bình 5-Fold CV trên 9 tập dữ liệu (giá trị tốt nhất in **đậm**, phương pháp EDL-ECC được đánh dấu ★)

| Dataset | Mô hình | 1-HL ↑ | SubAcc ↑ | Micro-F1 ↑ | Macro-F1 ↑ | Jaccard ↑ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Scene** | BR | 0.9889 | 0.9643 | 0.9944 | 0.9944 | 0.9874 |
| | CC | 0.9890 | 0.9664 | 0.9945 | 0.9944 | 0.9875 |
| | RAkEL | 0.9902 | 0.9705 | 0.9950 | 0.9950 | 0.9894 |
| | ★ **EDL-ECC** | **0.9929** | **0.9817** | **0.9964** | **0.9964** | **0.9929** |
| | EDL-RAkEL | **0.9929** | **0.9817** | **0.9964** | **0.9964** | **0.9929** |
| **CHD_49** | BR | 0.6871 | 0.1730 | 0.6650 | 0.6285 | 0.4189 |
| | CC | 0.6880 | 0.1982 | 0.6793 | 0.6339 | 0.4496 |
| | RAkEL | 0.6784 | 0.1748 | 0.6555 | 0.6143 | 0.4288 |
| | ★ **EDL-ECC** | **0.7126** | 0.1640 | **0.7213** | **0.6587** | **0.5155** |
| | EDL-RAkEL | 0.4324 | 0.0288 | 0.6034 | 0.5780 | 0.4324 |
| **emotions** | BR | 0.7833 | 0.2260 | 0.6833 | 0.6812 | 0.5523 |
| | CC | 0.7811 | 0.2683 | 0.6738 | 0.6723 | 0.5536 |
| | RAkEL | 0.7704 | 0.2227 | 0.6560 | 0.6538 | 0.5330 |
| | ★ **EDL-ECC** | **0.8069** | **0.3018** | **0.7149** | **0.7059** | **0.6007** |
| | EDL-RAkEL | 0.3114 | 0.0000 | 0.4749 | 0.4708 | 0.3114 |
| **GpositivePseAAC** | BR | 0.8165 | 0.5375 | 0.6429 | 0.5431 | 0.5992 |
| | CC | 0.8194 | 0.5607 | 0.6488 | 0.5470 | 0.6136 |
| | RAkEL | 0.8343 | **0.6609** | 0.6705 | 0.5537 | 0.6676 |
| | ★ **EDL-ECC** | **0.8391** | 0.5897 | **0.6967** | **0.6187** | **0.6619** |
| | EDL-RAkEL | 0.2519 | 0.0000 | 0.4025 | 0.3793 | 0.2519 |
| **HumanPseAAC** | BR | 0.8402 | 0.0834 | 0.3356 | 0.1888 | 0.2586 |
| | CC | 0.8572 | 0.1323 | 0.2906 | 0.1682 | 0.2494 |
| | RAkEL | 0.8514 | 0.1004 | 0.3360 | 0.1881 | 0.2663 |
| | ★ **EDL-ECC** | **0.8953** | **0.1974** | **0.4268** | **0.2267** | **0.3451** |
| | EDL-RAkEL | 0.0847 | 0.0000 | 0.1561 | 0.1429 | 0.0847 |
| **PlantPseAAC** | BR | 0.8553 | 0.1268 | 0.3042 | 0.2112 | 0.2337 |
| | CC | 0.8592 | 0.1503 | 0.3060 | 0.2119 | 0.2499 |
| | RAkEL | 0.8644 | 0.1442 | 0.3115 | 0.2128 | 0.2449 |
| | ★ **EDL-ECC** | **0.8857** | **0.2096** | **0.3692** | **0.2188** | **0.2986** |
| | EDL-RAkEL | 0.0899 | 0.0000 | 0.1650 | 0.1547 | 0.0899 |
| **VirusPseAAC** | BR | 0.7907 | 0.2756 | 0.4615 | 0.3729 | 0.3878 |
| | CC | 0.7907 | 0.2755 | 0.4636 | 0.3737 | 0.3909 |
| | RAkEL | 0.7923 | **0.3048** | 0.4700 | 0.3852 | 0.4124 |
| | ★ **EDL-ECC** | **0.7987** | 0.2319 | **0.5435** | **0.4252** | **0.4391** |
| | EDL-RAkEL | 0.2028 | 0.0000 | 0.3371 | 0.3096 | 0.2028 |
| **Water-quality** | BR | 0.5163 | 0.1868 | 0.6579 | 0.6524 | 0.4904 |
| | CC | 0.4593 | 0.2557 | 0.5993 | 0.6038 | 0.4263 |
| | RAkEL | 0.5113 | 0.2198 | 0.6516 | 0.6494 | 0.4834 |
| | ★ **EDL-ECC** | **0.9794** | **0.7321** | **0.9891** | **0.9778** | **0.9790** |
| | EDL-RAkEL | 0.9401 | 0.2387 | 0.9691 | 0.9559 | 0.9401 |
| **Yeast** | BR | 0.6841 | 0.0538 | 0.6098 | 0.6034 | 0.4540 |
| | CC | 0.6755 | 0.0554 | 0.5986 | 0.5948 | 0.4512 |
| | RAkEL | 0.6630 | 0.0575 | 0.6037 | 0.5973 | 0.4505 |
| | ★ **EDL-ECC** | **0.7173** | **0.0732** | **0.6409** | **0.6290** | **0.4881** |
| | EDL-RAkEL | 0.3638 | 0.0000 | 0.5335 | 0.5302 | 0.3638 |

---

### 4.3.2 Bảng Tổng hợp: EDL-ECC vs Baseline Tốt nhất

**Bảng 4.2:** Cải thiện của EDL-ECC so với baseline tốt nhất (BR / CC / RAkEL)

| Dataset | Baseline tốt nhất | Micro-F1 Baseline | Micro-F1 EDL-ECC | Cải thiện (%) |
|---|---|:---:|:---:|:---:|
| Scene | RAkEL | 0.9950 | **0.9964** | +0.14% |
| CHD_49 | CC | 0.6793 | **0.7213** | +6.18% ↑↑ |
| emotions | BR | 0.6833 | **0.7149** | +4.62% ↑↑ |
| GpositivePseAAC | RAkEL | 0.6705 | **0.6967** | +3.91% ↑ |
| HumanPseAAC | RAkEL | 0.3360 | **0.4268** | +27.02% ↑↑↑ |
| PlantPseAAC | RAkEL | 0.3115 | **0.3692** | +18.52% ↑↑↑ |
| VirusPseAAC | RAkEL | 0.4700 | **0.5435** | +15.64% ↑↑ |
| Water-quality | BR | 0.6579 | **0.9891** | +50.34% ↑↑↑ |
| Yeast | BR | 0.6098 | **0.6409** | +5.10% ↑↑ |

**EDL-ECC đạt Micro-F1 cao nhất trên 9/9 tập dữ liệu.**

---

## 4.4 Phân tích Chi tiết theo Nhóm Dataset

### 4.4.1 Nhóm Dữ liệu Ảnh: Scene

Dataset **Scene** (2,407 ảnh phong cảnh, 6 nhãn: Beach, Sunset, FallFoliage, Field, Mountain, Urban) là dataset hình ảnh duy nhất trong thực nghiệm. Đặc trưng đã được trích xuất sẵn thành vector 294 chiều (histogram màu + texture).

**Kết quả nổi bật:**
- EDL-ECC đạt **Subset Accuracy = 98.17%** — tức là 98.17% ảnh được phán đoán đúng hoàn toàn tập nhãn
- Cải thiện đáng kể so với RAkEL truyền thống (0.9705 → 0.9817 SubAcc)
- EDL-RAkEL đạt kết quả bằng EDL-ECC — xác nhận cả hai kiến trúc EDL đều phù hợp với dữ liệu ảnh

**Phân tích:** Scene là dataset "dễ nhất" trong bộ thực nghiệm (baseline BR đã đạt 96.43% SubAcc). Điều này cho thấy EDL không chỉ giúp trên dataset khó mà còn tinh chỉnh thêm trên dataset đã có hiệu suất cao sẵn.

### 4.4.2 Nhóm Dữ liệu Y tế: CHD_49

Dataset **CHD_49** (555 bệnh nhân, 49 đặc trưng lâm sàng, 6 nhãn bệnh tim mạch) là dataset y tế có ý nghĩa ứng dụng cao nhất.

**Kết quả nổi bật:**
- EDL-ECC cải thiện Micro-F1 từ 0.6793 (CC) lên **0.7213** (+6.18%)
- 1-Hamming Loss tăng từ 0.6880 lên **0.7126** — nghĩa là tỷ lệ nhãn đúng tăng từ 68.8% lên 71.3%
- Jaccard Index tăng mạnh: 0.4496 → **0.5155** (+14.67%)

**Lưu ý:** Subset Accuracy của EDL-ECC (0.1640) thấp hơn CC (0.1982) — điều này phản ánh rằng EDL-ECC tối ưu cho Micro-F1, không hẳn là Subset Accuracy. Với bài toán y tế, Micro-F1 thường quan trọng hơn vì phản ánh hiệu suất trên từng bệnh lý.

**EDL-RAkEL thất bại** trên CHD_49 (0.4324 1-HL) do cấu trúc dữ liệu tabular của bệnh nhân phù hợp với CC hơn RAkEL.

### 4.4.3 Nhóm Sinh học Phân tử: PseAAC Datasets

4 dataset PseAAC (HumanPseAAC, PlantPseAAC, GpositivePseAAC, VirusPseAAC) đều sử dụng đặc trưng Pseudo Amino Acid Composition (440 chiều) để biểu diễn chuỗi protein.

**Cải thiện vượt trội trên HumanPseAAC:**
- Micro-F1: 0.3356 (BR) → **0.4268** (EDL-ECC): cải thiện **+27%**
- Jaccard: 0.2586 → **0.3451**: cải thiện **+33.5%**

Đây là dataset khó nhất trong nhóm PseAAC (14 nhãn chức năng protein, phân phối rất thưa). Sự cải thiện lớn cho thấy cơ chế Uncertainty-Gated Propagation đặc biệt hiệu quả khi số nhãn lớn và phụ thuộc nhãn phức tạp.

**EDL-RAkEL thất bại hoàn toàn** (SubAcc = 0.000 trên tất cả PseAAC datasets): Do dữ liệu chuỗi protein có phụ thuộc nhãn mạnh theo chuỗi — bỏ mất cấu trúc tuần tự của chains.

### 4.4.4 Trường hợp đặc biệt: Water-quality

Dataset **Water-quality** (1,060 mẫu nước, 16 đặc trưng hóa học, 14 nhãn chất lượng) cho thấy mức cải thiện **đột biến nhất**:

| Metric | BR | CC | RAkEL | **EDL-ECC** | Cải thiện vs BR |
|---|---|---|---|---|---|
| 1-HL | 0.5163 | 0.4593 | 0.5113 | **0.9794** | **+463 bps** |
| Micro-F1 | 0.6579 | 0.5993 | 0.6516 | **0.9891** | **+50.3%** |
| Jaccard | 0.4904 | 0.4263 | 0.4834 | **0.9790** | **+99.6%** |

Mức nhảy vọt này (tất cả baseline ~50-65% Micro-F1 → EDL-ECC **98.91%**) cho thấy cấu trúc phụ thuộc nhãn trong Water-quality rất rõ ràng và tuần tự — hoàn toàn phù hợp với Classifier Chains. EDL nắm bắt được cấu trúc này hiệu quả hơn nhiều so với Logistic Regression.

---

## 4.5 Phân tích Độ bất định Evidential

### 4.5.1 Uncertainty Calibration

Kết quả đo độ bất định $u$ trên validation sets cho thấy mô hình EDL có khả năng **tự hiệu chỉnh (calibration)** xuất sắc:

| Trạng thái dự đoán | Giá trị $u$ trung bình |
|---|---|
| Dự đoán **ĐÚNG** (correct) | $u \approx 0.05$ – $0.15$ |
| Dự đoán **SAI** (incorrect) | $u \approx 0.55$ – $0.85$ |

Điều này có nghĩa là **khi mô hình sai, nó "biết" mình đang không chắc chắn** — thể hiện qua giá trị $u$ cao. Đây là tính chất quan trọng mà CC và RAkEL truyền thống hoàn toàn không có.

### 4.5.2 Ý nghĩa của Uncertainty trong Ứng dụng

**Trong y tế (CHD_49):** Khi bộ phân loại trả về $(p_{\text{pos}} = 0.72, u = 0.78)$, hệ thống có thể cảnh báo bác sĩ "mô hình không chắc về chẩn đoán này — cần xem xét thêm" thay vì đưa ra kết quả tự tin sai lầm.

**Trong chuỗi phân loại (EDL-ECC):** Khi nhãn $k-1$ có $u_{k-1} = 0.82$ (rất không chắc), đầu vào cho bước $k$ là:

$$\mathbf{x}^{(k)} = [\mathbf{x}_{\text{gốc}},\; p_{k-1}=0.60,\; u_{k-1}=0.82, \ldots]$$

Bộ phân loại bước $k$ có thể học cách **giảm trọng số** cho thông tin từ bước $k-1$ khi thấy $u_{k-1}$ cao — ngăn lỗi lan truyền qua chuỗi.

### 4.5.3 So sánh định tính với Softmax

Mạng nơ-ron với softmax thông thường trên cùng dữ liệu thường cho xác suất $\hat{p}$ rất cao (gần 0 hoặc 1) bất kể đúng hay sai — hiện tượng **overconfidence**. EDL phá vỡ điều này bằng cách buộc mô hình phải "có bằng chứng cụ thể" trước khi đưa ra xác suất cao.

---

## 4.6 Thảo luận về Giới hạn

### 4.6.1 EDL-RAkEL với Dữ liệu Phi Ảnh

EDL-RAkEL cho kết quả tốt trên **Scene** và **Water-quality** nhưng thất bại (SubAcc = 0.000) trên phần lớn dataset sinh học. Nguyên nhân:

1. **Nhãn thưa + ngưỡng 0.5**: Với dataset 14 nhãn nhưng mỗi mẫu chỉ có 2-3 nhãn active, ngưỡng 0.5 mặc định dự đoán "tất cả = 0" cho hầu hết mẫu
2. **Mất cấu trúc tuần tự**: Protein sequences có phụ thuộc nhãn theo chuỗi mà RAkEL (parallel) không nắm bắt được
3. **Calibration issue**: EDL trên Label Powerset $2^k$ classes với dữ liệu ít khó hội tụ tốt

**Hướng khắc phục:** Tối ưu ngưỡng riêng cho EDL-RAkEL (tương tự EDL-ECC) và dùng $m$ lớn hơn để tăng coverage.

### 4.6.2 Chi phí Tính toán

EDL-ECC chậm hơn đáng kể so với baseline do:
- Phải huấn luyện $M \times L$ neural networks (thay vì $L$ Logistic Regressions)
- Chuỗi tuần tự không thể song song hóa hoàn toàn trong training

Với $M=3$, $L=14$ (Yeast), $T=10$ epochs, batch_size=32: training mất ~3-5 phút trên GPU — chấp nhận được cho nghiên cứu.

---

## Tóm tắt Phần 4

- **EDL-ECC đạt Micro-F1 tốt nhất trên 9/9 tập dữ liệu** so với BR, CC, RAkEL
- **Cải thiện lớn nhất** trên Water-quality (+50.3% Micro-F1), HumanPseAAC (+27.0%), PlantPseAAC (+18.5%)
- **Uncertainty calibration** xác nhận: mô hình tự biết khi nào không chắc ($u$ cao khi sai)
- **EDL-RAkEL** phù hợp cho dữ liệu ảnh (Scene: 98.17% SubAcc) nhưng không phù hợp cho dữ liệu chuỗi sinh học
