# 📄 DỊCH VÀ PHÂN TÍCH CHI TIẾT

## Rethinking Multi-Label Image Classification With Deep Learning: Taxonomy, Challenge, and Outlook
> Xuelin Zhu, Xiu-Shen Wei, Jiawei Ge, Shuai Xu, Bing Wang  
> arXiv:2607.00839v1 | Nộp: 1 tháng 7 năm 2026  
> The Hong Kong Polytechnic University & Southeast University & Nanjing University of Aeronautics

---

## 🗺️ Bản đồ tổng quan bài báo

```
MLIC Survey 2026
├── I.   Giới thiệu
├── II.  Nền tảng (Background)
│        ├── Định nghĩa bài toán
│        ├── Datasets
│        ├── Backbones
│        └── Evaluation Metrics
├── III. Phương pháp (6 nhóm lớn)
│        ├── A. Region-Oriented (Hướng vùng ảnh)
│        ├── B. Label-Oriented (Hướng nhãn)
│        ├── C. Architecture-Oriented (Hướng kiến trúc)
│        ├── D. Representation-Oriented (Hướng biểu diễn)
│        ├── E. Learning-Oriented (Hướng học)
│        └── F. Data-Oriented (Hướng dữ liệu)
├── IV.  Thách thức & Triển vọng
└── V.   Kết luận
```

---

## I. GIỚI THIỆU

### Bài toán MLIC là gì?
**Multi-Label Image Classification (MLIC)** — Phân loại ảnh đa nhãn — là bài toán nhận diện **nhiều đối tượng hoặc khái niệm** cùng lúc trong một ảnh. Khác với phân loại đơn nhãn (SLIC) chỉ dán 1 nhãn cho ảnh có 1 đối tượng nổi bật, MLIC dự đoán tập nhiều nhãn cho ảnh phức tạp.

### Ứng dụng thực tế
| Lĩnh vực | Ứng dụng cụ thể |
|---|---|
| Xe tự hành | Phát hiện đồng thời người đi bộ + xe + biển báo |
| Y tế | Chẩn đoán ECG bất thường đa bệnh lý |
| Hệ thống gợi ý | Pinterest board recommendation |
| Robot di động | Nhận dạng đồ vật trong môi trường |
| Viễn thám | Phân loại ảnh vệ tinh đa nhãn |
| Y khoa hình ảnh | Phân tích X-quang ngực |

### Tại sao MLIC khó hơn SLIC?
MLIC kế thừa toàn bộ độ khó của SLIC (cần trích xuất đặc trưng tốt), nhưng còn thêm:
1. **Mô hình hóa phụ thuộc nhãn**: Các nhãn không độc lập — "bàn phím" thường đi kèm "máy tính"
2. **Tương quan không gian**: Vị trí tương đối của các đối tượng mang thông tin ngữ nghĩa
3. **Phụ thuộc ngữ nghĩa**: Các nhãn có quan hệ ngữ nghĩa phức tạp qua ngôn ngữ
4. **Bối cảnh nền ảnh**: Nền ảnh chứa nhiều gợi ý ngữ nghĩa quan trọng

### Đóng góp chính của bài survey
1. **Review toàn diện** đầu tiên dành riêng cho MLIC dựa trên deep learning (survey trước [2017] chỉ có 37 công trình, toàn statistical learning)
2. **Phát hiện "Learning Game" ẩn** giữa SLIC và MLIC — một xung đột lý thuyết chưa ai đặt tên
3. **Thực nghiệm kiểm chứng** các thách thức và hướng nghiên cứu tiềm năng

---

## II. NỀN TẢNG

### Định nghĩa hình thức
- **Dataset**: $\mathcal{D} = \{(I_m, \mathbf{y}_m)\}_{m=1}^M$
- **Nhãn**: $\mathbf{y} = [y_0, y_1, \ldots, y_{N-1}]^\top$ với $y_i \in \{0, 1\}$
- **Mục tiêu**: Học hàm $f: \mathcal{I} \to \mathcal{Y}$ ánh xạ ảnh sang không gian multi-hot encoding

### Datasets tiêu chuẩn
| Dataset | Số ảnh | Số nhãn | Đặc điểm |
|---|---|---|---|
| **Pascal VOC** | ~17,000 | 20 | Benchmark kinh điển |
| **MS-COCO** | ~330,000 | 80 | Scale lớn, phổ biến nhất |
| **NUS-WIDE** | ~270,000 | 81 | Ảnh web thực tế |

### Backbones phổ biến
- **ResNet** (CNN): Trích xuất đặc trưng phân cấp với inductive bias tốt cho pattern cục bộ
- **Vision Transformer (ViT)**: Mô hình hóa ngữ cảnh toàn cục qua self-attention, linh hoạt hơn

### Metrics đánh giá
- **AP (Average Precision)**: Hiệu suất từng nhãn riêng biệt
- **mAP (mean Average Precision)**: Hiệu suất tổng thể qua tất cả nhãn

---

## III. PHÂN LOẠI PHƯƠNG PHÁP (6 NHÓM)

---

### A. NHÓM 1: Region-Oriented — Phương pháp Hướng Vùng Ảnh

> **Ý tưởng cốt lõi**: Biến MLIC thành nhiều bài toán phân loại đa lớp theo từng vùng ảnh riêng biệt.

#### A.1 Detection-Based Region Proposal (Đề xuất vùng dựa trên phát hiện đối tượng)

**Training-Free (Không cần huấn luyện thêm)**:
- Pipeline 3 bước: Đề xuất vùng → Trích xuất đặc trưng → Dự đoán nhãn
- Công cụ: BING, Edge Boxes, Selective Search, Mask R-CNN (pretrained)
- Tổng hợp dự đoán: category-wise max-pooling qua các vùng

**Training-Based (Tích hợp huấn luyện end-to-end)**:
- Tích hợp module phát hiện đối tượng vào mạng (Faster R-CNN style + RoI pooling)
- Phương pháp RLSD: Lớp localization fully convolutional → bilinear interpolation → max-pooling

#### A.2 Attention-Based Region Proposal (Đề xuất vùng dựa trên Attention)

**CAM (Class Activation Mapping)**:
- Học lớp conv 1×1 → tạo activation map theo nhãn → binarize → lấy vùng discriminative
- Tổng hợp cuối: category-wise max-pooling

**Spatial Transformer**:
- Module học ma trận affine transformation → định vị vùng đối tượng
- Kết hợp localization + classification trong kiến trúc end-to-end duy nhất

**Recurrent Attention (Attention Tuần tự)**:
- Dùng LSTM để lần lượt khám phá các vùng liên quan
- DRAM: Xử lý nhiều độ phân giải tại các vị trí cụ thể, cập nhật trạng thái ẩn

#### A.3 Region-Aware Relation Modeling (Mô hình quan hệ giữa các vùng)

**Semantic Dependency (Phụ thuộc ngữ nghĩa)**:
- Ví dụ: "board" trong cảnh tuyết → "snowboard"; trong phố → "skateboard"
- Phương pháp: Dùng LSTM xử lý tuần tự các vùng → học quan hệ ngữ nghĩa ẩn

**Spatial Correlation (Tương quan không gian)**:
- Ví dụ: Màn hình + "keyboard" → "computer"; màn hình + remote control → "TV"
- Phương pháp: Self-attention với positional encoding (IoU, relative direction)

---

### B. NHÓM 2: Label-Oriented — Phương pháp Hướng Nhãn

> **Ý tưởng cốt lõi**: Học embedding ngữ nghĩa hoặc biểu diễn thị giác đặc trưng cho từng nhãn, biến MLIC thành nhiều bài toán phân loại nhị phân theo từng nhãn.

#### B.1 Label-Specific Semantic Embedding (Embedding ngữ nghĩa đặc trưng theo nhãn)

**Label Embedding for Fusion**:
- One-hot vector → không gian liên tục → nối ghép với image features → LSTM → dự đoán
- MFB pooling: Kết hợp image features và label embeddings qua Hadamard product

**Label Embedding for Alignment**:
- Dùng GloVe/BERT để biểu diễn nhãn thành word vectors
- Transformer decoder: label embeddings làm queries, image features làm keys/values
- Cross-modal alignment: Minimize MSE giữa cosine similarity của embeddings và xác suất co-occurrence thống kê

**Label Embedding as Classifier**:
- Embeddings đóng vai trò trực tiếp là vector trọng số của classifier
- Inner product giữa image features và label embeddings → label scores

#### B.2 Label-Specific Visual Representation (Biểu diễn thị giác đặc trưng theo nhãn)

**Bilinear Pooling**:
- Kết hợp label embeddings và image features qua Hadamard product
- Normalized logits → re-weight và aggregate image features → biểu diễn thị giác theo nhãn
- Low-rank bilinear pooling: Giảm chi phí tính toán

**CAM-Based**:
- Học activation maps theo nhãn (conv layers + 1×1 conv, số kernels = số nhãn)
- Softmax theo chiều không gian → attention weights → aggregate image features

**Multi-Head Self-Attention (MSA)**:
- Transformer decoder: label embeddings → queries; image features → keys/values
- Sau nhiều lớp cross-attention → label embeddings tích lũy đặc trưng thị giác liên quan

#### B.3 Label-Aware Relation Modeling (Mô hình quan hệ giữa các nhãn)

**RNN-Based**:
- Đúc MLIC thành bài toán sinh chuỗi nhãn (LSTM với nhiều thứ tự: frequent-first, rare-first, dictionary order)
- Nhãn được dự đoán ở bước trước làm đầu vào cho bước tiếp theo

**GNN-Based**:
- Xây đồ thị nhãn từ label co-occurrence
- GAT: Tự động học hệ số tương quan từ features của hàng xóm

**⭐ GCN-Based — ML-GCN (Phương pháp nổi bật nhất)**:

$$A_{ij} = \begin{cases} p / \sum_{j \neq i} A'_{ij} & \text{nếu } i \neq j \\ 1 - p & \text{nếu } i = j \end{cases}$$

$$H^{l+1} = \sigma(A H^l W^l)$$

- Node: GloVe embeddings của nhãn
- Edge: Ma trận co-occurrence (binarized với ngưỡng $\tau$)
- Stacking GCN layers → học inter-dependent classifiers → match với image features

**Ba loại ma trận tương quan trong GCN:**
| Loại | Ví dụ | Ưu điểm | Nhược điểm |
|---|---|---|---|
| Hand-crafted | Co-occurrence từ training set | Đơn giản, hiệu quả | Cứng nhắc, không phụ thuộc instance |
| Adaptive | 1×1 conv + dot product | Linh hoạt hơn | Dễ over-smoothing |
| Dynamic | Từ image features (instance-aware) | Bắt quan hệ cụ thể từng ảnh | Phức tạp, khó tối ưu |

**Policy-Based**:
- Markov Decision Process: trạng thái = image feature + nhãn đã dự đoán
- Label mask training: Che ngẫu nhiên một số nhãn, yêu cầu mô hình dự đoán từ nhãn còn lại

---

### C. NHÓM 3: Architecture-Oriented — Phương pháp Hướng Kiến trúc

> **Ý tưởng cốt lõi**: Thiết kế kiến trúc mạng chuyên biệt và hiệu quả cho MLIC.

#### C.1 Transformer-Based Architectures

**Intra-Modal (Trong cùng một modality — ảnh)**:
- Transformer encoder để capture spatial correlations giữa patches ảnh
- Masked attention: Giới hạn MSA ở các nhãn high-confidence
- Multi-label Transformer: Window partitioning + in-window pixel attention + cross-window attention

**Inter-Modal (Giữa các modality — ảnh và nhãn)**:
- Transformer decoder: Fuse image features với label embeddings
- Graph Transformer: Dynamically refine ma trận tương quan nhãn
- Interactive visual-linguistic attention: Cross-modal interaction cho joint representations

#### C.2 Parallel Architectures

**Two-Branch (Hai nhánh song song)**:
- *Global-Local*: Nhánh global capture tổng thể ảnh; nhánh local tập trung vùng đối tượng → kết hợp
- *Two-Stream*: Hai nhánh phục vụ modality khác nhau (spatial vs. semantic; visual vs. textual)

**Multi-Branch (Nhiều nhánh)**:
- Ba nhánh song song: categorical memory + channel-relation + spatial-relation
- Mixture-of-Experts style: Nhiều CNN sub-networks → fuse qua multi-output heads

#### C.3 Multi-Scale Architectures

**Feature Pyramid**: Resize feature maps từ các lớp CNN khác nhau về cùng resolution → fuse

**Kernel Pyramid**: Nhiều conv kernels với kích thước khác nhau → multi-scale feature maps

**Loss Pyramid**: Dự đoán và tính loss ở mỗi scale riêng → tổng hợp

#### C.4 Other Architectures
- Hierarchical, Mamba-based, RWKV-based, Mixture-of-Experts, fully graph-based architectures

---

### D. NHÓM 4: Representation-Oriented — Phương pháp Hướng Biểu diễn

> **Ý tưởng cốt lõi**: Học biểu diễn đặc trưng mạnh để cải thiện MLIC.

#### D.1 Contrastive Representations (Học tương phản)

**Label-Correlation-Agnostic**: Pull features cùng nhãn lại gần, push features khác nhãn ra xa (không xét tương quan nhãn)

**Label-Correlation-Aware**: Pull các samples chia sẻ đủ nhiều nhãn lại gần nhau; sử dụng GMM để model uncertainty về sự hiện diện, vị trí, và phức tạp thống kê của nhãn

#### D.2 Debiased Representations (Biểu diễn phi thiên lệch)

**Confounding Effect (Hiệu ứng nhiễu)**:
- Vấn đề: Ngữ cảnh xung quanh đối tượng tạo thiên lệch (confounder) → dự đoán sai
- Giải pháp: Backdoor adjustment theo causal theory:

$$P(Y | do(X)) = \sum_c P(Y | X, C=c) \cdot P(C=c)$$

**Mediating Effect (Hiệu ứng trung gian)**:
- Vấn đề: Mediator M nằm giữa X và Y → dự đoán phụ thuộc vào context, không phải pure instance
- Giải pháp: TDE (Total Direct Effect) theo counterfactual reasoning:

$$\text{TDE}(Y) = P(Y|X=x, M=m) - P(Y|X=x_0, M=m)$$

#### D.3 Distilled Representations (Biểu diễn qua Knowledge Distillation)
- Distill từ detection model → MLIC model (feature-level + prediction-level)
- Uncertainty distillation: Hướng dẫn student network xử lý nhãn khó
- Multi-order label-pair dependency distillation: Batch-level và instance-level

#### D.4 Metric Representations
- Deep metric learning: Max-margin loss + max-correlation loss
- Bidirectional deep distance metric: Ảnh gần label embedding đúng hơn embedding sai

---

### E. NHÓM 5: Learning-Oriented — Phương pháp Hướng Học

> **Ý tưởng cốt lõi**: Các chiến lược học đa dạng và chuyên biệt cho MLIC.

#### E.1 Learning to Rank (Học xếp hạng)
- Biến MLIC từ tập quyết định nhị phân sang bài toán xếp hạng nhãn
- **Ranking-Loss-Only**: Weighted approximate ranking loss, smooth pairwise hinge loss
- **Joint-Loss**: Kết hợp BCE loss + ranking loss → cân bằng accuracy và ranking

#### E.2 Multi-Task/Instance Learning
- **Multi-Task**: Mỗi nhãn là một task riêng; shared representations + task-specific heads
- **Multi-Instance**: Ảnh là "bag" của nhiều vùng; region-level scoring → image-level prediction
- Mixture-of-Experts Transformer: Task-specialized experts + shared experts → tránh negative transfer

#### E.3 Other Learning Methods
- **Matching Learning**: MLIC = bài toán matching giữa instance patches và label embeddings
- **Dictionary Learning**: Từ điển visual atoms; sparse coding → associate với nhãn
- **Curriculum Learning**: Học từ dễ đến khó với RL agent
- **Compositional Learning**: Biểu diễn ảnh = kết hợp category-related features

---

### F. NHÓM 6: Data-Oriented — Phương pháp Hướng Dữ liệu

> **Ý tưởng cốt lõi**: Tập trung vào phân phối thống kê và tăng cường dữ liệu đa nhãn.

#### F.1 Data Imbalance (Mất cân bằng dữ liệu)

**Intra-Label Imbalance (Trong từng nhãn)**:
- Mỗi nhãn: Negative samples >> Positive samples
- Giải pháp: **ASL (Asymmetric Loss)** — Điều chỉnh decay rate khác nhau cho positive và negative

**Inter-Label Imbalance (Giữa các nhãn — Long-tail)**:
- Loss-Based: Re-weighting theo co-occurrence, focal term, balanced ASL
- Semantic-Based: CLIP để liên kết head labels với tail labels qua semantics

#### F.2 Data Augmentation
- **Transformation-Based**: Mixup (nội suy ảnh + nhãn), blended mixed images
- **Erasing-Based**: Xóa vùng discriminative → buộc mô hình khám phá vùng khác
  - Attention-guided erasing: Xóa dominant attention regions của high-confident labels

#### F.3 Large-Scale Pretrained Knowledge
- **Cross-Modal (CLIP)**: Dual-modal attention, gated dual-modal alignment, Transformer decoder với text queries
- **Visual Knowledge**: ImageNet pretrained networks, SAM segmentation priors
- **Linguistic Knowledge**: GloVe/BERT embeddings, LLMs cho category attributes và descriptions

---

## IV. THÁCH THỨC & TRIỂN VỌNG

### 🎮 "Learning Game" Ẩn giữa SLIC và MLIC — Phát hiện quan trọng nhất của bài báo

**Vấn đề đặt ra**: Tại sao local representations luôn vượt trội global representations trong MLIC?

**Lý giải của tác giả — SLIC-MLIC Learning Game**:

Trong paradigm **global representation**:
- Với ảnh đơn nhãn (SLIC): Ideal → Mỗi classifier nằm gần trung tâm cluster của nhãn đó, các classifiers cách xa nhau
- Với ảnh đa nhãn (MLIC): Features của ảnh phải gần với **cả hai** classifiers (ví dụ: vừa gần "cat" vừa gần "dog")
- **Xung đột**: Push classifiers ra xa nhau thì tốt cho SLIC; Pull chúng lại gần thì cần cho MLIC

→ Đây là **cuộc chơi (game)** không thể thắng tuyệt đối cùng một lúc!

**Tại sao local representations tránh được game này?**
- Phân rã global feature → các region-level local features riêng biệt
- Training degenerates thành tối ưu SLIC-style trên từng region riêng
- Mỗi region chỉ cần gần 1 classifier → không có xung đột

**Hạn chế của local representations**:
- Phụ thuộc chất lượng object localization
- Tạo conceptual gap giữa MLIC và SLIC → khó unification
- MLIC lag behind SLIC về performance

**Hướng giải quyết tiềm năng**: **Image-conditioned dynamic classifiers** — Classifier parameters được điều chỉnh theo nội dung ảnh: tự động tách xa với single-label images và dịch chuyển lại gần với multi-label images.

**Mở rộng Learning Game sang các bài toán khác**:
- Image retrieval, scene understanding, visual relationship detection
- Video action/event recognition
- Multimodal foundation models (CLIP-style)

---

### 📌 5 Thách thức chính được chỉ ra

#### 1. Small Objects (Đối tượng nhỏ)
- **Vấn đề**: Chiếm ít pixels, bị đối tượng lớn và background "lấn át"
- **Thực nghiệm**: mAP cho small objects ~32-41%, large objects ~80-85% — gap rất lớn
- **Observation**: Patch-based methods (ViT-Base, PAT) tốt hơn cho small objects
- **Hướng nghiên cứu**: Multi-scale patch tokenization, dùng co-occurrence từ large objects để refine small objects

#### 2. Contextual Bias (Thiên lệch ngữ cảnh)
- **Vấn đề**: Object thường đi kèm với ngữ cảnh cố định → classifier bị "dính" vào context, không phải object
- **Hướng nghiên cứu**: Metrics đo context-bias (stability dưới background perturbations), kiến trúc factorize objects và contexts, causal pipelines cho feature-level interventions

#### 3. Attention Mechanism
- **Vấn đề**: Không hiểu rõ attention học focus vào đâu và tại sao, khó diagnose failure
- **Hướng nghiên cứu**: Attention diagnostics, regularized attention (intra-label sparsity + inter-label separation), counterfactual interventions

#### 4. Prompt Learning
- **Vấn đề**: Category-level prompts yếu trong MLIC — không capture multi-label characteristics
- **Hướng nghiên cứu**: Adaptive prompt generation theo label co-occurrence + context; integrate label hierarchies vào prompts

#### 5. Novel Architectures
- **Vấn đề**: CNN (inductive bias cứng nhắc), RNN (sequential), ViT (quadratic attention) — đều có hạn chế
- **Kiến trúc mới tiềm năng**:
  - **Mamba**: Linear-time sequence modeling, capture label dependencies
  - **RWKV**: Recurrent + attention-like, parallelizable, order-agnostic
  - **KANs**: Function-approximation mới, high-order label interactions

---

### 🔭 3 Triển vọng tương lai

#### 1. Domain-Specialized MLIC
- Y tế (X-quang, ECG), khí tượng, viễn thám
- Hướng: Foundation models (SAM, CLIP) + domain-specific finetuning/prompt adaptation
- Yêu cầu: Reliable, generalizable, trustworthy cho safety-critical applications

#### 2. Data-Efficient Learning
- Paradigm mới: MLIC chỉ dùng textual data (không cần ảnh có nhãn) — CLIP-enabled
- Đặc biệt hữu ích: Medical imaging, scientific microscopy, industrial inspection
- Hướng: Cross-modal transferability của label prompts trong CLIP embedding space

#### 3. General Label Intelligence (Trí tuệ nhãn tổng quát)
- LLMs cung cấp: Conceptual reasoning, commonsense inference, structured knowledge
- Hướng: LLMs như "label intelligence" — tự động interpret, expand, restructure label vocabularies
- Kỳ vọng: Next-gen MLIC systems — scalable, interpretable, adaptable

---

## V. KẾT LUẬN

Survey là **review toàn diện đầu tiên** về deep learning-based MLIC, tổ chức theo 6 perspectives:
- **Label-oriented, Region-oriented, Architecture-oriented**: Nhóm thiết kế kiến trúc và học nhãn
- **Representation-oriented, Learning-oriented, Data-oriented**: Nhóm chiến lược học

Điểm nổi bật nhất: **Phát hiện SLIC-MLIC Learning Game** — một xung đột lý thuyết ẩn giải thích tại sao local representations luôn tốt hơn global trong MLIC, và mở ra hướng nghiên cứu **image-conditioned dynamic classifiers** đầy tiềm năng.

---

## 🔗 Liên hệ với Nghiên cứu EDL-RAkEL của Bạn

| Khía cạnh trong Survey | Điểm liên quan với EDL-RAkEL |
|---|---|
| **Label-Oriented → Label-Aware Relation Modeling** | EDL-RAkEL dùng RAkEL để mô hình hóa quan hệ nhãn qua $k$-labelsets — tương đương với GCN/GNN-based methods nhưng không cần graph construction rõ ràng |
| **Learning Game (SLIC-MLIC)** | Đây chính xác là lý do EDL-RAkEL tốt hơn EDL-ECC trên ảnh! RAkEL chia nhỏ bài toán → mỗi sub-model chỉ xử lý $k$ nhãn → không có global conflict |
| **Representation-Oriented → Distilled Representations** | EDL uncertainty $u$ tương tự uncertainty distillation — mô hình biết khi nào nó không chắc |
| **Data-Oriented → Data Imbalance → ASL** | Nếu dữ liệu ảnh y tế có long-tail, nên xem xét tích hợp ASL vào EDL loss function |
| **Novel Architecture → Mamba/RWKV cho MLIC** | Hướng nghiên cứu tiếp theo: EDL + Mamba backbone cho sequence-free label modeling |
| **Attention Mechanism challenge** | EDL cung cấp $u$ để diagnose khi nào attention không tin cậy — đây là đóng góp độc đáo của bạn |
| **Domain-Specialized MLIC → Medical Imaging** | Đây chính là target application của EDL-RAkEL với ViT backbone — được survey xác nhận là hướng tiềm năng |
