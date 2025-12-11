# LNU Deep Learning 2025: German Text Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat&logo=huggingface)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)

This repository contains the source code and documentation for the **LNU Deep Learn 2 Text Classification Challenge (2025)**. The objective of this project was to classify German language texts into disjoint categories using State-of-the-Art (SOTA) Transformer architectures.

The final solution leverages **Large Language Models**, **Gradient Checkpointing** for memory optimization, and **Incremental Repeated Cross-Validation** to maximize data utilization and stability.

---

## Model Evolution Pipeline

The development process followed a rigorous experimental pipeline, progressing from baseline architectures to high-capacity ensemble models.

| Architecture | Configuration | Outcome |
| :--- | :--- | :--- |
| **gbert-base** | Baseline | **~89.0%** Accuracy. Established lower bound performance. |
| **gbert-large** | Gradient Accumulation (BS=32) | **~91.3%** Accuracy. Demonstrated the necessity of higher parameter counts. |
| **gelectra-large** | Generator-Discriminator | Validated performance on ambiguous tokens; used for diversity. |
| **dbmdz/bert-base** | Repeated K-Fold | Tested stability on formal German vocabulary. |
| **gbert-large** | **Incremental Repeated Ensemble** | **~92.6%+** Accuracy. Final winning strategy utilizing soft-voting ensembles. |

---

## Architectures Used

### 1. BERT (Bidirectional Encoder Representations from Transformers)
The core of our solution relies on BERT, specifically the **GBERT** (Deepset) and **DBMDZ** implementations. BERT utilizes a multi-layer bidirectional Transformer encoder to learn deep bidirectional representations by jointly conditioning on both left and right contexts in all layers.

<img width="2866" height="1825" alt="image" src="https://github.com/user-attachments/assets/b7bb9439-a21d-4795-8f1a-7cd40879d1a5" />


### 2. ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements)
We utilized **GELECTRA** to introduce architectural diversity. Unlike BERT, which uses Masked Language Modeling (MLM), ELECTRA employs a sample-efficient pre-training task called Replaced Token Detection (RTD). A generator network replaces tokens with plausible alternatives, and a discriminator network determines whether each token is original or replaced.

<img width="1106" height="380" alt="image" src="https://github.com/user-attachments/assets/4c326901-bdcf-4a5e-8c7f-c2cbfa5e461a" />


---

## Methodology & Technical Implementation

### Data Preprocessing
* **Normalization:** HTML entity decoding and whitespace normalization.
* **Tokenization:** Utilized `bert-base-german-cased` and `gbert-large` tokenizers with a maximum sequence length of 256 tokens to preserve full context.
* **Label Encoding:** Mapped categorical targets to integer indices using `LabelEncoder`.

### Optimization Techniques
To facilitate the training of Large models (340M+ parameters) on limited GPU resources (Tesla T4), the following optimizations were implemented:
1.  **Gradient Checkpointing:** Reduced VRAM usage by 50-60% by trading compute for memory during the backward pass.
2.  **Mixed Precision Training (FP16):** Utilized `torch.amp` to accelerate training throughput and reduce memory footprint.
3.  **Gradient Accumulation:** Simulated a batch size.

### Training Strategy: Incremental Repeated K-Fold
Instead of standard Cross-Validation, we implemented **Incremental Learning**:
1.  The dataset was split into **5 folds**.
2.  This process was **repeated 2 times** with different seeds.
3.  The model was **not reset** between folds; instead, it continued training, allowing it to progressively learn from the entire dataset distribution while maintaining validation rigor through unseen folds.

---

## Results and Evaluation

The final evaluation was conducted using a sequential sampling strategy on the full training set to generate a comprehensive Error Analysis report.

* **Metric:** Accuracy.
* **Best Single Model:** `deepset/gbert-large` (92.6%).
* **Ensemble Strategy:** Soft-voting averaging of probability distributions from 10 training cycles.

## Usage

1.  **Environment Setup**
    ```bash
    pip install transformers torch scikit-learn mlflow tqdm
    ```

2.  **Execution**
    Run the unified Jupyter Notebook `LNU_Text_Classification_Solution.ipynb`. The pipeline will automatically:
    * Download and extract data.
    * Tune hyperparameters.
    * Execute the Incremental Repeated CV training.
    * Generate the `submission_final.csv` file.

3.  **Inference**
    The final script performs inference on `test.csv` using the trained model weights resident in memory.

---
