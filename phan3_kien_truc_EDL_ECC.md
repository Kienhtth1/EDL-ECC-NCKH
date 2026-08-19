# PHẦN 3: KIẾN TRÚC EDL-ECC CHI TIẾT

---

## 3.1 Tổng quan Kiến trúc

**EDL-ECC (Evidential Ensemble Classifier Chains)** là sự tích hợp của hai thành phần:
1. **EDL (Evidential Deep Learning)** làm **local classifier** cho từng nhãn trong chuỗi — thay thế Logistic Regression truyền thống trong CC
2. **Ensemble Classifier Chains (ECC)** làm **framework** mô hình hóa phụ thuộc nhãn — mở rộng bằng cơ chế Uncertainty-Gated Propagation

**Đóng góp cốt lõi** so với ECC truyền thống: Thay vì truyền nhãn dự đoán $\hat{y}_{k-1}$ như một sự thật hiển nhiên, EDL-ECC truyền cả **xác suất** $p_{k-1}$ lẫn **độ bất định** $u_{k-1}$ — cho phép bộ phân loại bước $k$ tự quyết định mức độ tin tưởng vào thông tin nhận được.

### Sơ đồ tổng thể

```
                    ┌────────────────────────────────────────────────────┐
                    │             EDL-ECC: 1 Chain (trong M chains)       │
                    │                                                      │
  Input x ──────►  │  ┌──────────┐   p₁, u₁   ┌──────────┐  p₂, u₂    │
  (features)       │  │ EDL_Bin  │ ──────────► │ EDL_Bin  │ ─────────► ··· ─► p_L
                   │  │ Module 1 │             │ Module 2 │             │
                   │  │ (nhãn π₁)│             │ (nhãn π₂)│             │
                   │  └──────────┘             └──────────┘             │
                   │      ▲                        ▲                     │
                   │      │ [x]                    │ [x, p₁, u₁]        │
                   └──────┼────────────────────────┼─────────────────────┘
                          │                        │
                   Uncertainty-Gated Propagation:
                   Input ở bước k = [x  ||  p_{k-1}  ||  u_{k-1}]
                                         (gate: nhãn chắc → đóng góp nhiều)

  Ensemble M chains → Average → Final Probability Vector p̂ ∈ [0,1]^L
```

---

## 3.2 EDL Binary Module — Bộ phân loại cơ bản

Mỗi bước trong chuỗi sử dụng một **EDL Binary Module** — mạng MLP nhỏ với đầu ra là tham số Dirichlet cho bài toán phân loại nhị phân (có/không nhãn $c_j$).

### 3.2.1 Kiến trúc MLP

```
Input: x^(k) ∈ R^{d_k}          (d_k = d_0 + 2(k-1) với k ≥ 2)
         ↓
  Linear(d_k → 128) + ReLU + Dropout(0.2)
         ↓
  Linear(128 → 64) + ReLU + Dropout(0.2)
         ↓
  Linear(64 → 2)               ← 2 logits: [z₀, z₁]
         ↓
  ReLU(·) + 1e-4 + 1.0        ← α = evidence + 1
         ↓
  α = [α₀, α₁] ∈ R²₊₊         ← tham số Dirichlet
```

**Kích thước đầu vào $d_k$ tăng dần theo chiều chuỗi:**
- Bước $k=1$: $d_1 = d_0$ (chỉ features gốc)
- Bước $k=2$: $d_2 = d_0 + 2$ (features gốc + $p_1$ + $u_1$)
- Bước $k=j$: $d_j = d_0 + 2(j-1)$ (features gốc + 2 giá trị từ mỗi bước trước)

Tổng số tham số của mỗi EDL Binary Module (bỏ qua phần tăng chiều theo chuỗi):
$$|\theta_k| \approx 128 \times d_k + 128 + 64 \times 128 + 64 + 2 \times 64 + 2$$

### 3.2.2 Đầu ra của EDL Binary Module

Từ $\boldsymbol{\alpha} = [\alpha_0, \alpha_1]$, tính:

$$S = \alpha_0 + \alpha_1, \quad \hat{p}_{\text{pos}} = \frac{\alpha_1}{S}, \quad u = \frac{2}{S}$$

Vector gate truyền sang bước tiếp theo: $\mathbf{g}_k = [p_k, u_k] \in \mathbb{R}^2$

---

## 3.3 Cơ chế Uncertainty-Gated Propagation

Đây là **đóng góp cốt lõi** của EDL-ECC so với CC và ECC truyền thống.

### 3.3.1 So sánh cơ chế truyền thông tin

| Phương pháp | Thông tin truyền qua chuỗi | Chất lượng thông tin |
|---|---|---|
| **CC truyền thống** | $\hat{y}_{k-1} \in \{0,1\}$ (nhãn cứng) | Mất mát thông tin xác suất |
| **Probabilistic CC** | $p_{k-1} \in [0,1]$ (nhãn mềm) | Chưa biết độ tin cậy |
| **EDL-ECC** | $[p_{k-1}, u_{k-1}] \in \mathbb{R}^2$ | **Biết cả giá trị lẫn độ tin cậy** |

### 3.3.2 Cơ chế truyền thông tin chi tiết

Tại bước $k$, đầu vào cho EDL Binary Module thứ $k$ được xây dựng bằng **phép nối (concatenation)**:

$$\mathbf{x}^{(k)} = \left[\,\mathbf{x}_{\text{gốc}}\; \Big\|\; p_1\; \Big\|\; u_1\; \Big\|\; p_2\; \Big\|\; u_2\; \Big\|\; \cdots \Big\|\; p_{k-1}\; \Big\|\; u_{k-1}\,\right] \in \mathbb{R}^{d_0 + 2(k-1)}$$

**Tại sao concatenation với $[p, u]$ hiệu quả hơn chỉ $p$?**

Xét hai trường hợp bộ phân loại ở bước trước:
- **Trường hợp A**: $p_1 = 0.9$, $u_1 = 0.05$ → Rất chắc chắn nhãn positive
- **Trường hợp B**: $p_1 = 0.9$, $u_1 = 0.85$ → Không chắc chắn, có thể là dữ liệu lạ

Nếu chỉ truyền $p_1 = 0.9$, bước 2 không phân biệt được hai trường hợp. Khi có thêm $u_1$, bước 2 có thể học cách **giảm trọng số** thông tin từ trường hợp B — tránh error propagation từ dự đoán không đáng tin cậy.

Cơ chế này MLP học **tự động** qua các trọng số $W_k$ — bộ phân loại bước $k$ học cách tương tác giữa $\mathbf{x}_{\text{gốc}}$, $p_{k-1}$ và $u_{k-1}$ để tối ưu loss.

---

## 3.4 Thuật toán Huấn luyện EDL-ECC

### 3.4.1 Thuật toán Training (1 Chain)

```
Algorithm: EDL-ECC Training (single chain)
Input:  X_train ∈ R^{N×d}, Y_train ∈ {0,1}^{N×L}, epochs T, lr η
Output: chain_models = [(lbl_idx₁, model₁), ..., (lbl_idxL, modelL)]

1. Khởi tạo ngẫu nhiên thứ tự nhãn: π = permutation(L)
2. X_current ← X_train  ▷ Bắt đầu với features gốc
3. For pos = 1, 2, ..., L do:
   a. lbl_idx ← π[pos]
   b. Khởi tạo: model ← EDL_Binary_Module(in_dim=dim(X_current))
   c. Optimizer: Adam(model.parameters(), lr=η)
   d. For epoch t = 1, ..., T do:
       For each mini-batch (X_b, y_b) from DataLoader(X_current, Y_train[:,lbl_idx]):
           α ← model(X_b)                    ▷ Forward pass
           L ← edl_binary_mse_loss(α, y_b, t) ▷ Compute loss
           optimizer.zero_grad()
           L.backward()                        ▷ Backward pass
           optimizer.step()
   e. model.eval()
   f. With no_grad():
       α_pred ← model(X_current)
       p_pos, u ← predict_edl_binary(α_pred)  ▷ [N,1], [N,1]
       X_current ← concat([X_current, p_pos, u], dim=-1)  ▷ Augment input
   g. chain_models.append((lbl_idx, model))
4. Return chain_models
```

### 3.4.2 Ensemble Training

```
For chain_id = 1, ..., M do:
    chain_models = EDL_ECC_Train_Single_Chain(X_train, Y_train, π_random)
    self.chains.append(chain_models)
```

**M = 3 chains** với 3 thứ tự nhãn $\pi_1, \pi_2, \pi_3$ được lấy mẫu ngẫu nhiên độc lập.

### 3.4.3 Complexity Analysis

- **Training**: $O(M \times L \times T \times N \times d_{\max})$ với $d_{\max} = d_0 + 2(L-1)$
- **Inference**: $O(M \times L \times d_{\max})$ per sample
- **Bộ nhớ mô hình**: $M \times L$ EDL Binary Modules

Với $M=3$, $L=14$ (Yeast), $d_0=103$, $T=10$: **tổng 42 EDL modules** được huấn luyện, mỗi module ~30K parameters.

---

## 3.5 Thuật toán Suy luận (Inference)

### 3.5.1 Inference cho 1 Chain

```
Algorithm: EDL-ECC Inference (single chain)
Input:  X_val ∈ R^{N×d}, chain_models
Output: chain_prob ∈ [0,1]^{N×L}

1. X_curr ← X_val
2. chain_prob ← zeros(N, L)
3. For (lbl_idx, model) in chain_models do:
   a. α ← model(X_curr)
   b. p_pos, u ← predict_edl_binary(α)
   c. chain_prob[:, lbl_idx] ← p_pos.squeeze()
   d. X_curr ← concat([X_curr, p_pos, u], dim=-1)
4. Return chain_prob
```

### 3.5.2 Ensemble Averaging

$$\hat{P}(y_j = 1 \mid \mathbf{x}) = \frac{1}{M} \sum_{m=1}^M \hat{p}^{(m)}_j(\mathbf{x}), \quad j = 1, \ldots, L$$

### 3.5.3 Quyết định nhãn với Ngưỡng Tối ưu

Thay vì dùng ngưỡng mặc định 0.5, EDL-ECC tìm **ngưỡng tối ưu** trên validation set:

$$\theta^* = \arg\max_{\theta \in \{0.1, 0.15, 0.2, \ldots, 0.5\}} \text{Micro-F1}\!\left(Y_{\text{val}},\; \mathbb{1}[\hat{P} \geq \theta]\right)$$

Điều này đặc biệt quan trọng với dữ liệu có nhãn thưa (sparse labels), nơi ngưỡng 0.5 thường quá cao và dẫn đến dự đoán tất cả là negative.

---

## 3.6 Kiến trúc EDL-RAkEL — Mở rộng Song song

### 3.6.1 Động lực và Thiết kế

EDL-RAkEL (Evidential Random k-Labelsets) là kiến trúc **song song** thay vì tuần tự. Mỗi sub-model $h_i$ xử lý **độc lập** một tập nhãn con $R_i \subset \mathcal{C}$ với $|R_i| = k$:

```
                    Input x
                   /   |   \
              h₁(x)  h₂(x)  h₃(x)   ...  hₘ(x)
              R₁={c₁,c₂,c₃}                R_m
                   \   |   /
              Uncertainty-Weighted Voting
                       ↓
                  p̂ ∈ [0,1]^L
```

Mỗi sub-model $h_i$ là một **EDL Label Powerset classifier**: nhận $\mathbf{x}$ làm đầu vào, xuất ra phân phối Dirichlet trên $2^k$ tổ hợp nhãn trong $R_i$.

**Ưu thế trên dữ liệu ảnh:** Không có vấn đề multi-modal fusion như CC — mỗi sub-model nhận nguyên vẹn vector đặc trưng ảnh (1024+ chiều) không bị "pha loãng" bởi concatenation nhãn.

### 3.6.2 Thuật toán Bỏ phiếu Trọng số Độ bất định

Khi nhãn $c_j$ xuất hiện trong nhiều k-labelsets $R_{i_1}, R_{i_2}, \ldots$, xác suất cuối cùng được tổng hợp **có trọng số uncertainty**:

$$\hat{p}(y_j = 1 \mid \mathbf{x}) = \frac{\displaystyle\sum_{i:\, c_j \in R_i} (1 - u_i) \cdot \hat{p}_{i,j}(\mathbf{x})}{\displaystyle\sum_{i:\, c_j \in R_i} (1 - u_i)}$$

với $u_i = \frac{2^k}{S_i}$ là độ bất định của sub-model $i$.

**Diễn giải:** Sub-model nào có $u_i$ nhỏ (tự tin hơn) sẽ có trọng số lớn hơn trong biểu quyết — tương tự "chuyên gia tự tin có tiếng nói lớn hơn".

### 3.6.3 So sánh EDL-ECC vs EDL-RAkEL

| Tiêu chí | EDL-ECC | EDL-RAkEL |
|---|---|---|
| **Cấu trúc** | Tuần tự (Sequential) | Song song (Parallel) |
| **Mô hình hóa tương quan nhãn** | Qua chuỗi điều kiện | Qua Label Powerset trong nhóm nhỏ |
| **Phù hợp với** | Tabular data, chuỗi sinh học | Dữ liệu ảnh, CNN/ViT features |
| **Error propagation** | Có (giảm nhờ $u$-gating) | Không (parallel) |
| **Multi-modal fusion** | Có vấn đề với ảnh | Không có vấn đề |
| **Hiệu suất (Scene)** | 98.17% SubAcc | 98.17% SubAcc |
| **Hiệu suất (Yeast)** | **71.73%** 1-HL | 36.38% 1-HL |

---

## 3.7 Tham số Mô hình và Siêu tham số

| Tham số | EDL-ECC | EDL-RAkEL |
|---|---|---|
| Số units lớp ẩn 1 | 128 | 128 |
| Số units lớp ẩn 2 | 64 | 64 |
| Dropout rate | 0.2 | 0.2 |
| Learning rate | 1e-3 | 1e-3 |
| Epochs per module | 10 | 10 |
| Batch size | 32 | 32 |
| Số chains / sub-models | M = 3 | m = L (đủ coverage) |
| k-labelset size | — | k = 3 |
| Annealing step $t_{\text{step}}$ | 5 | 5 |
| Positive weight | 3.0 | 3.0 |
| Threshold optimization | Có (grid: 0.1→0.5, step 0.05) | Có |
| Optimizer | Adam | Adam |

---

## Tóm tắt Phần 3

Kiến trúc EDL-ECC được xây dựng từ:
1. **EDL Binary Module** — MLP 3 lớp xuất ra $[\alpha_0, \alpha_1]$, từ đó tính $p_{\text{pos}}$ và $u$
2. **Uncertainty-Gated Propagation** — Truyền $[p_k, u_k]$ qua chuỗi thay vì $\hat{y}_k$ hay chỉ $p_k$
3. **Ensemble** — $M=3$ chains với thứ tự ngẫu nhiên, average probabilities
4. **Threshold optimization** — Tìm ngưỡng tốt nhất trên validation fold

Kiến trúc **EDL-RAkEL** là biến thể song song, phù hợp hơn cho dữ liệu ảnh, sử dụng uncertainty-weighted voting thay vì gated propagation.
