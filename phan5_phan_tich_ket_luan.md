# PHẦN 5: PHÂN TÍCH, THẢO LUẬN VÀ KẾT LUẬN

---

## 5.1 Phân tích Tại sao EDL-ECC Vượt trội

### 5.1.1 Ba cơ chế cải thiện chính

**Cơ chế 1 — Uncertainty-Gated Propagation (UGP):**

Trong CC truyền thống, khi nhãn $k-1$ được dự đoán với ngưỡng ranh giới $p_{k-1} = 0.52$, toàn bộ thông tin này được truyền sang bước $k$ như sự thật hiển nhiên. Với EDL-ECC, bộ phân loại bước $k$ nhận thêm $u_{k-1}$. Nếu $u_{k-1} = 0.85$ (rất không chắc), bộ phân loại bước $k$ có thể học cách **không tin tưởng** thông tin này qua các trọng số $W_k$ — giảm thiệu lỗi tích lũy.

Bằng chứng thực nghiệm: Sự cải thiện của EDL-ECC so với CC lớn nhất trên các dataset nhiều nhãn (HumanPseAAC: 14 nhãn, +27% Micro-F1; Water-quality: 14 nhãn, +65%). Dataset ít nhãn hơn (CHD_49: 6 nhãn) có cải thiện nhỏ hơn — nhất quán với giả thuyết UGP hiệu quả nhất khi chuỗi dài.

**Cơ chế 2 — Điều chuẩn KL-Divergence:**

Hàm mất mát EDL bao gồm thành phần KL regularization giữa phân phối Dirichlet và phân phối đều:

$$\text{KL}\left[\text{Dir}(\tilde{\boldsymbol{\alpha}}) \| \text{Dir}(\mathbf{1})\right]$$

Thành phần này phạt mô hình khi tạo ra bằng chứng lớn cho các lớp **không đúng** — buộc mô hình phải "cẩn thận" hơn trong việc phân bổ bằng chứng. Điều này tương tự tác dụng của L2 regularization nhưng có cơ chế rõ ràng hơn trong không gian Dirichlet.

**Cơ chế 3 — Ensemble Diversity:**

3 chains với thứ tự nhãn ngẫu nhiên khác nhau tạo ra **diversity** trong ensemble. Mỗi chain mô hình hóa phụ thuộc nhãn theo một thứ tự khác nhau — chain trung bình cuối cùng ít bị ảnh hưởng bởi thứ tự nhãn "xấu" hơn so với một chain đơn.

### 5.1.2 Liên kết với Lý thuyết SLIC-MLIC Learning Game

Trong Phần 1, chúng ta đã giới thiệu lý thuyết SLIC-MLIC Learning Game từ Zhu et al. (2026): global representation tạo xung đột giữa SLIC và MLIC objectives.

**EDL-ECC giải quyết xung đột này như thế nào?**

Mỗi EDL Binary Module trong chuỗi về bản chất là một bài toán **phân loại nhị phân đơn lẻ** (binary SLC): dự đoán có/không nhãn $c_j$ cho mẫu hiện tại. Đây gần với SLIC hơn MLC — không có xung đột classifier separation. Phụ thuộc nhãn được mô hình hóa **không phải qua global representation** mà qua chuỗi điều kiện $[x, p_{<k}, u_{<k}]$.

Điều này giải thích tại sao EDL-ECC hoạt động tốt hơn các phương pháp dùng global representation: nó tránh được learning game bằng cách phân rã bài toán đa nhãn thành nhiều bài toán nhị phân.

### 5.1.3 Vai trò của Evidential vs Probabilistic trong Context Sinh học

Các dataset PseAAC (protein sequences) có đặc điểm: một protein thường chỉ có 2-3 chức năng trong số 4-14 nhãn có thể. Nhiều nhãn "không phù hợp" với protein hiện tại nhưng đặc trưng PseAAC có thể mơ hồ.

EDL trong bối cảnh này: Thay vì học softmax đưa ra xác suất cao cho mọi nhãn, EDL học cách "không có bằng chứng" ($e_k \approx 0, u_k \approx 1$) cho các chức năng protein không liên quan. Điều này tạo ra biểu diễn **sparse** hơn trong không gian bằng chứng — phù hợp với bản chất thưa nhãn của sinh học phân tử.

---

## 5.2 Phân tích Hạn chế Hiện tại

### 5.2.1 Hạn chế về Kiến trúc

**MLP với tabular features:**
Kiến trúc EDL Binary Module hiện tại là MLP 3 lớp ($d_k \to 128 \to 64 \to 2$). Với tabular data, MLP là lựa chọn hợp lý. Tuy nhiên, khi mở rộng sang **dữ liệu ảnh thực sự** (raw pixels), cần backbone CNN hoặc ViT để trích xuất đặc trưng visual trước khi áp dụng EDL — đây là hướng phát triển tự nhiên tiếp theo.

**Tăng chiều đầu vào theo chuỗi:**
Với dataset nhiều nhãn ($L = 14$), đầu vào của module cuối chuỗi có thêm $2 \times 13 = 26$ chiều từ các bước trước. Tuy không lớn với các dataset có $d_0 \geq 49$, nhưng có thể gây vấn đề nếu $d_0$ nhỏ (ví dụ Water-quality: $d_0 = 16$, module cuối có $d_{14} = 16 + 26 = 42$ chiều — vẫn ổn).

**Cơ chế Attention vắng mặt:**
Chuỗi hiện tại concatenate thông tin $[p_{k-1}, u_{k-1}]$ một cách cứng nhắc. Như được phân tích từ bài báo MLTC (IJCAI 2025), cơ chế **Label Attention** có thể giúp bộ phân loại bước $k$ **chọn lọc** thông tin từ các bước trước dựa trên độ liên quan ngữ nghĩa — không chỉ dựa vào vị trí trong chuỗi. Đây là hướng cải tiến tiềm năng.

### 5.2.2 Hạn chế về Dữ liệu

**Quy mô nhỏ:**
Phần lớn dataset trong thực nghiệm có dưới 3,000 mẫu. Hiệu suất EDL-ECC trên dataset lớn hơn (MS-COCO: 330,000 ảnh; EURLex: 15,000 văn bản với 3,956 nhãn) chưa được kiểm chứng.

**Chưa thử nghiệm dữ liệu ảnh thô:**
Dataset Scene đã được pre-extracted features (294-dim). Thực nghiệm trực tiếp trên ảnh nguyên với ResNet/ViT backbone chưa được thực hiện — đây là khoảng cách lớn so với state-of-the-art.

---

## 5.3 Hướng Nghiên cứu Tương lai

### 5.3.1 Tích hợp Cơ chế Attention (Ngắn hạn)

Dựa trên hai bài báo đã phân tích:

**Từ MLTC (IJCAI 2025 — UCLAF):**
Cơ chế Label Attention Aware Network (LAN) học trọng số attention $\beta_{ik}$ giữa nhãn $l_k$ và đặc trưng đầu vào $f_i$:
$$c_k = \sum_i \beta_{ik} f_i, \quad \beta_{ik} = \text{softmax}(v_b^\top \tanh(W_b[l_k; f_i]))$$

Có thể áp dụng tư tưởng này vào EDL-ECC: thay vì concatenate cứng $[p_{k-1}, u_{k-1}]$, dùng cross-attention với nhãn đang được dự đoán làm query và các nhãn đã dự đoán làm key/value.

**Từ MLIC Survey (2026):**
Attention tạo **label-specific visual representation** — tránh SLIC-MLIC Learning Game. Với EDL, có thể kết hợp uncertainty $u$ như một điều kiện để điều chỉnh attention weights.

**Đề xuất nghiên cứu cụ thể — "EDL-AECC" (Attentive Evidential Classifier Chains):**

Thay thế phép concatenation thuần túy bằng Uncertainty-Weighted Cross-Attention:
$$\text{Attn}(Q_k, K_{<k}, V_{<k}) = \text{softmax}\!\left(\frac{Q_k K_{<k}^\top}{\sqrt{d}}\right) \odot (1-\mathbf{u}_{<k}) \cdot V_{<k}$$

Đây là hướng nghiên cứu mới, chưa được công bố — có giá trị đóng góp học thuật.

### 5.3.2 Backbone CNN/ViT cho Dữ liệu Ảnh Thực (Trung hạn)

Hướng phát triển tự nhiên và quan trọng nhất:

```
Ảnh thực → ResNet-50 / ViT-Base → Feature map 1024-dim
                                          ↓
                                     EDL-ECC / EDL-RAkEL
                                          ↓
                              Multi-label prediction + Uncertainty
```

Survey MLIC (2026) xác nhận: **Medical imaging** là ứng dụng triển vọng nhất của MLIC — X-quang ngực đa bệnh lý là test case lý tưởng cho EDL (cần biết khi nào không chắc).

**Dataset đề xuất:** CheXpert (224,316 X-quang ngực, 14 nhãn bệnh lý) — quy mô đủ lớn để kiểm chứng EDL-ECC với ViT backbone.

### 5.3.3 Asymmetric EDL Loss (Ngắn hạn, dễ implement)

Survey MLIC (2026) nhắc đến **ASL (Asymmetric Loss)** là phương pháp hiệu quả cho long-tail multi-label. Có thể kết hợp asymmetric decay vào hàm mất mát EDL:

$$\mathcal{L}_{\text{ASL-EDL}} = w_+ \cdot \mathcal{L}_{\text{EDL}}^{(+)} + w_- \cdot \mathcal{L}_{\text{EDL}}^{(-)} + \lambda_t \cdot \text{KL}(\cdot)$$

với $w_+ > w_-$ để ưu tiên học các positive samples hiếm hơn. Điều này đặc biệt hữu ích cho các dataset sinh học có nhãn rất thưa.

---

## 5.4 Kết luận

### 5.4.1 Tóm tắt Đóng góp

Nghiên cứu này đề xuất và thực nghiệm **EDL-ECC (Evidential Ensemble Classifier Chains)** — phương pháp mới tích hợp Evidential Deep Learning vào kiến trúc Ensemble Classifier Chains cho bài toán phân loại đa nhãn. Ba đóng góp chính:

**Đóng góp 1 — Cơ chế Uncertainty-Gated Propagation:**
Thay thế truyền nhãn cứng nhắc trong CC bằng cơ chế truyền $[p_k, u_k]$ — cho phép giảm thiểu error propagation qua chuỗi nhờ lượng hóa độ bất định evidential tại mỗi bước.

**Đóng góp 2 — Lượng hóa độ bất định trong MLC:**
Lần đầu tiên áp dụng EDL (Dirichlet-based uncertainty) làm local classifier cho Classifier Chains — mở ra khả năng cảnh báo độ không tin cậy trong dự đoán đa nhãn, đặc biệt quan trọng cho ứng dụng y tế.

**Đóng góp 3 — Kiến trúc EDL-RAkEL:**
Mở rộng tư tưởng EDL sang kiến trúc song song RAkEL với Uncertainty-Weighted Voting — xác nhận lý thuyết về ưu thế của RAkEL trên dữ liệu ảnh.

### 5.4.2 Kết quả Chính

- **EDL-ECC đạt Micro-F1 cao nhất trên 9/9 tập dữ liệu** (tabular và chuỗi sinh học) so với BR, CC, RAkEL truyền thống
- Cải thiện **từ +0.14% đến +50.3% Micro-F1** tùy dataset — lớn nhất trên Water-quality và các dataset sinh học nhiều nhãn
- **Uncertainty calibration tốt**: $u \approx 0.05$–$0.15$ khi đúng, $u \approx 0.55$–$0.85$ khi sai
- **EDL-RAkEL** đạt **98.17% Subset Accuracy** trên Scene — xác nhận phù hợp với dữ liệu ảnh

### 5.4.3 Ý nghĩa Thực tiễn

Khả năng lượng hóa độ bất định của EDL-ECC mở ra hướng ứng dụng thực tiễn quan trọng:

| Lĩnh vực | Ứng dụng EDL-ECC | Giá trị từ Uncertainty |
|---|---|---|
| **Y tế** | Chẩn đoán đa bệnh lý từ xét nghiệm | Cảnh báo bác sĩ khi mô hình không chắc |
| **Sinh học** | Dự đoán chức năng protein | Đánh dấu protein cần thực nghiệm thêm |
| **Môi trường** | Phân loại chất lượng nước đa chỉ tiêu | Phát hiện mẫu bất thường |
| **Hình ảnh** | Phân loại cảnh quan đa nhãn | Tăng độ tin cậy dự đoán tự động |

### 5.4.4 Câu kết

Nghiên cứu chứng minh rằng việc **biết mình không biết gì** — thông qua độ bất định evidential — không chỉ là một tính chất lý thuyết thú vị mà còn mang lại cải thiện thực nghiệm rõ ràng trong phân loại đa nhãn. Khi một bộ phân loại trong chuỗi thành thật về sự không chắc chắn của mình, toàn bộ chuỗi phân loại hoạt động tốt hơn — đây là bài học cốt lõi mà EDL-ECC mang lại.

---

## Tài liệu tham khảo Tổng hợp

1. Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *NeurIPS*, 31.
2. Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. *Machine Learning*, 85(3), 333–359.
3. Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Random k-labelsets for multi-label classification. *IEEE TKDE*, 23(7), 1079–1089.
4. Tsoumakas, G., & Katakis, I. (2007). Multi-label classification: An overview. *Int. J. Data Warehousing and Mining*, 3(3), 1–13.
5. Zhu, X., Wei, X.-S., Ge, J., Xu, S., & Wang, B. (2026). Rethinking multi-label image classification with deep learning: Taxonomy, challenge, and outlook. *arXiv:2607.00839*.
6. Zhu, Z., Zhou, P., Li, Z., Chen, K., & Zhu, J. (2025). Multi-label text classification with label attention aware and correlation aware contrastive learning. *IJCAI 2025*, 8420–8428.
7. Wang, J. et al. (2016). Cnn-rnn: A unified architecture for multi-label image classification. *CVPR*.
8. Chen, Z. M., Wei, X.-S., Wang, P., & Guo, Y. (2019). Multi-label image recognition with graph convolutional networks. *CVPR*.
9. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
10. Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.
