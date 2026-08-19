# PHẦN 2: NỀN TẢNG LÝ THUYẾT

---

## 2.1 Lý thuyết Bằng chứng và Độ bất định trong Deep Learning

### 2.1.1 Phân loại các dạng Độ bất định

Trong học máy, có hai dạng độ bất định cơ bản cần phân biệt:

**Aleatoric Uncertainty (Độ bất định ngẫu nhiên):**
Phát sinh từ sự ngẫu nhiên vốn có của dữ liệu — tiếng ồn đo lường, biến động tự nhiên của hiện tượng. Dạng này **không thể giảm** dù có thêm dữ liệu.

**Epistemic Uncertainty (Độ bất định nhận thức — còn gọi là Model Uncertainty):**
Phát sinh từ sự thiếu thông tin hoặc thiếu dữ liệu trong vùng không gian đặc trưng mà mô hình chưa quan sát. Dạng này **có thể giảm** khi có thêm dữ liệu hoặc kiến thức phù hợp. Đây là dạng độ bất định mà EDL hướng đến đo lường.

Sensoy et al. (2018) nhận định: mạng nơ-ron sâu truyền thống — dù kết hợp với softmax — thường **overconfident**: xác suất đầu ra luôn gần 0 hoặc 1, ngay cả khi mẫu đầu vào hoàn toàn lạ so với dữ liệu huấn luyện. Điều này nguy hiểm trong các ứng dụng như y tế, nơi mô hình cần biết khi nào nên nói "Tôi không chắc" thay vì luôn đưa ra câu trả lời tự tin.

### 2.1.2 Lý thuyết Bằng chứng Dempster-Shafer

**Evidential Deep Learning (EDL)** được xây dựng trên nền tảng **Lý thuyết Dempster-Shafer về Bằng chứng** (Dempster-Shafer Theory of Evidence, DST). Khác với xác suất Bayesian truyền thống, DST cho phép biểu diễn trạng thái "không biết" rõ ràng.

Trong framework DST, một **mass function** $m: 2^\Omega \to [0,1]$ phân bổ niềm tin cho các tập con của không gian sự kiện $\Omega$, trong đó:
- $m(A)$: Lượng bằng chứng hỗ trợ trực tiếp tập $A \subseteq \Omega$
- $m(\Omega)$: Phần bằng chứng chưa được phân bổ cho lớp cụ thể nào — đây chính là **đo lường sự không biết (ignorance)**

**Kết nối với EDL:** Sensoy et al. (2018) đề xuất diễn giải đầu ra mạng nơ-ron như các bằng chứng Dirichlet, trong đó lượng bằng chứng $e_c \geq 0$ cho lớp $c$ phản ánh mức độ hỗ trợ của đặc trưng đầu vào cho lớp đó. Khi không có bằng chứng ($e_c \approx 0$ với mọi $c$), độ bất định tiến về 1.

---

## 2.2 Phân phối Dirichlet và vai trò trong EDL

### 2.2.1 Định nghĩa phân phối Dirichlet

Phân phối Dirichlet $\text{Dir}(\boldsymbol{\alpha})$ với tham số $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \ldots, \alpha_K)$ ($\alpha_k > 0$) là **phân phối trên đơn hình xác suất** $K$ chiều — tức là phân phối của các vector xác suất $\mathbf{p} = (p_1, \ldots, p_K)$ với $\sum_k p_k = 1$, $p_k \geq 0$.

Hàm mật độ xác suất:
$$\text{Dir}(\mathbf{p};\, \boldsymbol{\alpha}) = \frac{\Gamma\!\left(\sum_{k=1}^K \alpha_k\right)}{\prod_{k=1}^K \Gamma(\alpha_k)} \prod_{k=1}^K p_k^{\alpha_k - 1}$$

**Các tính chất quan trọng:**
- **Giá trị kỳ vọng**: $\mathbb{E}[p_k] = \dfrac{\alpha_k}{S}$ với $S = \sum_{k=1}^K \alpha_k$ (gọi là **Dirichlet strength**)
- **Phương sai**: $\text{Var}[p_k] = \dfrac{\alpha_k(S - \alpha_k)}{S^2(S+1)}$ — giảm khi $S$ tăng
- **Tập trung**: Khi $S \to \infty$, phân phối Dirichlet tập trung về điểm $\mathbb{E}[\mathbf{p}]$ — mô hình rất chắc chắn
- **Đồng nhất**: Khi $\alpha_k = 1$ với mọi $k$, $\text{Dir}(\mathbf{1})$ là phân phối đều trên đơn hình — mô hình hoàn toàn không biết gì

### 2.2.2 Tại sao Dirichlet phù hợp cho MLC?

Phân phối Dirichlet là **conjugate prior** của phân phối Multinomial — nghĩa là nếu dữ liệu phân phối Multinomial và prior là Dirichlet, thì posterior cũng là Dirichlet. Tính chất này cho phép cập nhật niềm tin một cách phân tích (analytical) khi có thêm bằng chứng.

Trong bài toán phân loại nhị phân (binary) cho từng nhãn — như trong EDL-ECC — phân phối Dirichlet với $K=2$ trở thành phân phối **Beta**:
$$p_{\text{pos}} \sim \text{Beta}(\alpha_1, \alpha_0)$$

với $\alpha_1 = e_1 + 1$ (bằng chứng cho lớp positive), $\alpha_0 = e_0 + 1$ (bằng chứng cho lớp negative).

---

## 2.3 Mô hình Evidential Deep Learning (EDL)

### 2.3.1 Kiến trúc tổng quát

EDL thay thế lớp softmax cuối cùng của mạng nơ-ron bằng một chuỗi biến đổi để xuất ra **tham số Dirichlet** thay vì điểm xác suất trực tiếp:

```
Input x → [Deep Network f_θ] → Logits z → ReLU → Evidence e ≥ 0
                                                        ↓
                                              α = e + 1  (α_k ≥ 1)
                                                        ↓
                                              S = Σ α_k  (Dirichlet strength)
                                                        ↓
                                          p_k = α_k / S  (Expected probability)
                                          u = K / S      (Uncertainty)
```

**Lưu ý quan trọng:** Hàm kích hoạt **ReLU** (thay vì softmax) được dùng để đảm bảo $e_k \geq 0$. Một số biến thể dùng **Softplus** để đảm bảo đạo hàm liên tục.

### 2.3.2 Các công thức cốt lõi của EDL

Với mạng nơ-ron $f_\theta(\mathbf{x})$ có đầu ra logits $\mathbf{z} \in \mathbb{R}^K$, EDL định nghĩa:

**Bước 1 — Tính Evidence:**
$$e_k = \text{ReLU}(z_k) + \epsilon, \quad \epsilon = 10^{-4}$$

**Bước 2 — Tính tham số Dirichlet:**
$$\alpha_k = e_k + 1, \quad k = 1, 2, \ldots, K$$

**Bước 3 — Tính Dirichlet Strength:**
$$S = \sum_{k=1}^K \alpha_k = \sum_{k=1}^K e_k + K$$

**Bước 4 — Tính xác suất dự đoán kỳ vọng:**
$$\hat{p}_k = \frac{\alpha_k}{S} = \mathbb{E}_{\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})}[p_k]$$

**Bước 5 — Tính độ bất định Evidential:**
$$u = \frac{K}{S} = \frac{K}{\sum_{k=1}^K e_k + K} \in \left(0, 1\right]$$

**Diễn giải trực quan:**
- $u \to 0$: Tổng bằng chứng $S$ rất lớn → mô hình rất chắc chắn
- $u \to 1$: Tổng bằng chứng $S \approx K$ (mỗi $\alpha_k \approx 1$, tức $e_k \approx 0$) → mô hình không có bằng chứng → hoàn toàn không chắc

### 2.3.3 Trường hợp nhị phân (K=2) — áp dụng trong EDL-ECC

Trong EDL-ECC, mỗi nhãn $c_j$ được phân loại bởi một EDL binary classifier riêng với $K=2$ (positive/negative):

$$\alpha_0 = e_0 + 1, \quad \alpha_1 = e_1 + 1, \quad S = \alpha_0 + \alpha_1$$

$$\hat{p}_{\text{pos}} = \frac{\alpha_1}{S}, \quad u = \frac{2}{S}$$

Khi $S = 2$ (không có bằng chứng nào): $u = 1$, $\hat{p}_{\text{pos}} = 0.5$ — mô hình hoàn toàn phân vân.  
Khi $S = 100$ (nhiều bằng chứng): $u = 0.02$ — mô hình rất tự tin.

---

## 2.4 Hàm Mất mát trong EDL

### 2.4.1 Hàm mất mát EDL-MSE

Sensoy et al. (2018) đề xuất hàm mất mát kết hợp hai thành phần:

$$\mathcal{L}_{\text{EDL}}(\boldsymbol{\alpha}, \mathbf{y}) = \underbrace{\sum_{k=1}^K \left[ (y_k - \hat{p}_k)^2 + \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1} \right]}_{\text{(1) Expected MSE}} + \underbrace{\lambda_t \cdot \text{KL}\left[\text{Dir}(\tilde{\boldsymbol{\alpha}}) \;\|\; \text{Dir}(\mathbf{1})\right]}_{\text{(2) KL Regularization}}$$

**Thành phần (1) — Expected MSE:**
Tối thiểu hóa sai số bình phương **kỳ vọng** giữa phân phối Dirichlet và one-hot label thật. Thành phần $\frac{\hat{p}_k(1-\hat{p}_k)}{S+1}$ là phương sai của phân phối Beta — đảm bảo rằng khi mô hình không chắc, phương sai này được tính vào loss để khuyến khích mô hình thu thập thêm bằng chứng.

**Thành phần (2) — KL Divergence Regularization:**
Phạt các phân phối Dirichlet quá "lan rộng" đối với nhãn sai, kéo về phân phối đều $\text{Dir}(\mathbf{1})$ (trạng thái không biết gì) cho các lớp mà thực tế không xuất hiện.

$\tilde{\boldsymbol{\alpha}} = \mathbf{y} + (1 - \mathbf{y}) \odot \boldsymbol{\alpha}$ — modified alpha loại bỏ bằng chứng cho các lớp đúng trước khi tính KL.

$$\text{KL}\!\left[\text{Dir}(\tilde{\boldsymbol{\alpha}}) \| \text{Dir}(\mathbf{1})\right] = \ln\frac{\Gamma(S_{\tilde{\alpha}})}{\prod_k \Gamma(\tilde{\alpha}_k)} + \sum_k (\tilde{\alpha}_k - 1)\left[\psi(\tilde{\alpha}_k) - \psi(S_{\tilde{\alpha}})\right]$$

với $\psi(\cdot)$ là hàm digamma.

**Hệ số annealing $\lambda_t$:**
$$\lambda_t = \min\!\left(1,\; \frac{t}{t_{\text{step}}}\right)$$

Trong những epoch đầu ($t$ nhỏ), $\lambda_t \approx 0$ → mô hình tập trung học dự đoán đúng trước. Sau đó $\lambda_t$ tăng dần → điều chuẩn KL tác động đầy đủ để kiểm soát bằng chứng cho nhãn sai.

### 2.4.2 Điều chỉnh mất cân bằng nhãn (Positive Weight)

Trong dữ liệu đa nhãn, mỗi nhãn thường có nhiều mẫu negative hơn positive đáng kể. EDL-ECC bổ sung trọng số asymmetric:

$$\mathcal{L}_{\text{EDL-ECC}} = w_{\text{pos}} \cdot \text{ExpectedMSE}(\boldsymbol{\alpha}, \mathbf{y}) + \lambda_t \cdot \text{KL}(\cdot)$$

với $w_{\text{pos}} = 3.0$ cho mẫu positive và $w_{\text{pos}} = 1.0$ cho mẫu negative — tương tự tinh thần của Asymmetric Loss (ASL) trong tài liệu MLIC.

---

## 2.5 Classifier Chains — Nền tảng Kiến trúc

### 2.5.1 Classifier Chains (CC) cổ điển

Read et al. (2011) đề xuất Classifier Chains như một giải pháp khắc phục hạn chế của Binary Relevance: thay vì huấn luyện $L$ bộ phân loại hoàn toàn độc lập, CC xây dựng **chuỗi tuần tự** trong đó mỗi bộ phân loại nhận thêm nhãn đã được dự đoán từ các bước trước.

Với thứ tự nhãn $\pi = (c_{\pi(1)}, c_{\pi(2)}, \ldots, c_{\pi(L)})$, bộ phân loại thứ $k$ trong chuỗi học:

$$f_k\!\left(\mathbf{x},\, \hat{y}_{\pi(1)},\, \hat{y}_{\pi(2)},\, \ldots,\, \hat{y}_{\pi(k-1)}\right) \rightarrow \hat{y}_{\pi(k)}$$

**Cơ sở xác suất:** CC xấp xỉ xác suất kết hợp bằng chuỗi điều kiện:

$$P(y_1, y_2, \ldots, y_L \mid \mathbf{x}) = \prod_{k=1}^L P\!\left(y_{\pi(k)} \mid y_{\pi(1)}, \ldots, y_{\pi(k-1)}, \mathbf{x}\right)$$

**Vấn đề về thứ tự nhãn:** Kết quả của CC phụ thuộc vào thứ tự $\pi$ — không có thứ tự nào tối ưu cho mọi tình huống.

### 2.5.2 Ensemble Classifier Chains (ECC)

Để giải quyết sự nhạy cảm với thứ tự nhãn, Read et al. (2011) đề xuất **Ensemble Classifier Chains (ECC)**: huấn luyện $M$ chuỗi với $M$ thứ tự nhãn ngẫu nhiên khác nhau, rồi tổng hợp bằng biểu quyết trung bình:

$$P(y_j = 1 \mid \mathbf{x}) = \frac{1}{M} \sum_{m=1}^M P_m\!\left(y_j = 1 \mid \mathbf{x}\right)$$

Ensemble không chỉ làm giảm phụ thuộc vào thứ tự nhãn mà còn giảm phương sai dự đoán tổng thể.

### 2.5.3 Vấn đề Error Propagation trong CC

Hạn chế cơ bản của CC truyền thống là **Error Propagation (Lan truyền sai số)**: Khi bộ phân loại bước $k$ dự đoán sai $\hat{y}_{\pi(k)}$, sai số này được truyền sang bước $k+1$ như một sự thật hiển nhiên — không có cơ chế nào cảnh báo bước tiếp theo về mức độ tin cậy của thông tin vừa nhận.

Ví dụ trong chẩn đoán y tế: Nếu bước 1 dự đoán sai "viêm phổi = có" với xác suất 0.51 (ngưỡng ranh giới), bước 2 sẽ coi đây là sự thật và điều chỉnh dự đoán về "tràn dịch màng phổi" theo — dù thực ra bước 1 rất không chắc. Đây là vấn đề mà **EDL-ECC giải quyết** bằng cơ chế Uncertainty-Gated Propagation được trình bày ở Phần 3.

---

## Tóm tắt Phần 2

Phần này đã xây dựng nền tảng lý thuyết đủ để hiểu kiến trúc EDL-ECC:

1. **EDL** giải quyết overconfidence của deep learning bằng cách xuất ra phân phối Dirichlet thay vì điểm xác suất — từ đó suy ra cả $\hat{p}_k$ lẫn $u$.
2. **Phân phối Dirichlet** là công cụ toán học phù hợp: conjugate prior của Multinomial, biểu diễn được "không biết gì" qua $\text{Dir}(\mathbf{1})$.
3. **Hàm mất mát EDL-MSE** kết hợp Expected MSE (học dự đoán đúng) và KL Regularization (kiểm soát bằng chứng cho nhãn sai), với annealing schedule để ổn định huấn luyện.
4. **Classifier Chains** mô hình hóa phụ thuộc nhãn qua chuỗi điều kiện — nhưng bị hạn chế bởi Error Propagation, vấn đề trung tâm mà EDL-ECC giải quyết.
