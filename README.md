# Evidential Deep Learning & Ensemble Classifier Chains (EDL-ECC) for Multi-Label Classification

This repository contains the implementation of a robust Multi-Label Classification (MLC) pipeline that integrates **Evidential Deep Learning (EDL)** with **Ensemble Classifier Chains (ECC)**. This approach aims to address the error propagation problem common in Classifier Chains, especially when dealing with highly imbalanced datasets.

## 🌟 Key Features
- **Uncertainty Quantification**: Uses EDL based on Subjective Logic and Dirichlet distributions to output an uncertainty score for every prediction.
- **Uncertainty-Aware Propagation**: Instead of hard-predicting 0 or 1, the model passes a continuous `[probability, uncertainty]` tuple down the classifier chain. This acts as a soft-gating mechanism to block noise from rare labels.
- **Robust against Imbalance**: The custom loss function (Expected MSE + KL Divergence) penalizes overconfident wrong predictions and provides better calibration.

## 📁 Repository Structure
```text
.
├── data/
│   └── bibtex/                    # Contains the BibTeX dataset (.arff format)
├── outputs/                       # Contains generated evaluation charts (Loss curves, Radar charts, Heatmaps, etc.)
├── complete_edl_ecc_pipeline.ipynb # The main executable Jupyter Notebook containing the full pipeline
├── report.md                      # Detailed academic report surveying MLC and the proposed EDL-ECC methodology
├── requirements.txt               # Required Python packages
└── README.md
```

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.10+ installed. Install the dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Pipeline
The entire workflow is encapsulated in `complete_edl_ecc_pipeline.ipynb`. Simply open the notebook in Jupyter or VS Code and run all cells sequentially:
1. **EDA**: Parses sparse `.arff` files, visualizes label distribution and correlation.
2. **Preprocessing**: Normalizes features and builds PyTorch DataLoaders.
3. **EDL Model**: Defines the custom architecture and Dirichlet loss function.
4. **ECC Wrapper**: Creates the ensemble chains.
5. **Training & Evaluation**: Trains the models and plots loss curves, uncertainty distributions, and confusion matrices.
6. **Advanced Visualization**: Generates Radar Charts, Boxplots, and Ranking Heatmaps.

## 📊 Evaluation
The model is evaluated using various metrics appropriate for imbalanced multi-label data, including:
- **Macro-F1 & Micro-F1**
- **Hamming Loss**
- **Subset Accuracy**
- **Jaccard Index**

All generated charts will be automatically saved to the `outputs/` directory.

## 📖 Documentation
For an in-depth understanding of the theoretical background, the mathematical formulation of EDL, and the Ablation study, please read the [report.md](report.md) included in this repository.
