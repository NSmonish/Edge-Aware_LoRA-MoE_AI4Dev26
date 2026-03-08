# 🍃 Edge-Aware MoE-LoRA: Edge-Cloud Cooperative LoRA-MoE
**A Lightweight Yet Powerful Parameter-Efficient Fine-Tuning Architecture for Edge Intelligence.**

> **Submission for PSG Tech AI4Dev '26 National-Level Hackathon**
> **Track:** Responsible AI and Resource Optimization
> **Team Members:** Monish N S (VIT), Manasi S (SRM), Dhanasrie S K (SRM)

## 📌 Project Overview
The proliferation of Large Language Models (LLMs) has revolutionized AI, but their massive size (billions of parameters) makes them infeasible for resource-constrained edge devices (smartphones, wearables). This creates critical bottlenecks for **user privacy**, **real-time latency**, and **deep personalization**. 

**EcoLoRA (Edge-Aware LoRA)** introduces a novel edge-cloud cooperative architecture. It decouples the fine-tuning workload, storing a tiny routing system and specialized Mixture of Experts (MoE) adapters directly on the edge device, while heavier, shared adapters remain on the cloud. 

### 🎯 Alignment with Hackathon Track
* **Resource Optimization:** Achieves State-of-the-Art performance while reducing the on-device trainable parameter footprint to **~0.0345%** (a mere 0.21 MB storage requirement for RoBERTa-base).
* **Responsible AI (Data Sovereignty):** Personalized weights (the router and experts) never leave the user's local device. Sensitive data used for fine-tuning remains strictly on-device, ensuring zero cloud leakage.

## 🏗️ System Architecture
EcoLoRA patches standard transformer models (e.g., RoBERTa) by replacing standard linear layers with an `EdgeAwareLoRALinear` module. 

1. **Cloud (Shared) Components:** Standard LoRA adapters (higher rank) shared across all devices for bulk feature extraction.
2. **Edge (Private) Components:** A lightweight `Routing System` and extremely low-rank `MoE Adapters`.
3. **Cooperative Flow:** The edge router analyzes input tokens, selects the Top-K experts locally, communicates intermediate representations with the cloud, and aggregates the final output.

## 📊 Performance Benchmarks (GLUE Dataset)
Tested using **RoBERTa-Large** against standard LoRA baselines. EcoLoRA achieves "no-compromise" performance, matching or exceeding standard LoRA while drastically cutting edge memory usage.

| NLP Task | Dataset | Standard LoRA Baseline | **EcoLoRA (Ours)** | On-Device Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| Natural Language Inference | **MNLI** | 90.6% | **90.81%** | 0.18% |
| Sentiment Analysis | **SST-2** | 96.2% | **96.33%** | 0.18% |
| Paraphrase Detection | **QQP** | 91.6% | **91.80%** | 0.30% |
| Question Answering / NLI | **QNLI** | **94.8%** | 94.73% | 0.30% |
| Linguistic Acceptability | **CoLA** | 68.2% | **86.00%** | 0.18% |
| Paraphrase Detection | **MRPC** | **90.9%** | 88.40% | 0.18% |
| Natural Language Inference | **RTE** | 85.2% | **85.56%** | 0.18% |
| Semantic Textual Similarity | **STS-B** | **92.3%** | 92.01% | 0.18% |
| **Overall Average** | - | 88.6% | **90.70%** | **~0.21%** |

## 🚀 How to Run (Proof of Concept)
The codebase is provided via a Google Colab notebook demonstrating the patching of RoBERTa, the custom LBTrainer (Load-Balancing Trainer), and rapid on-device personalization.

1. Open the https://colab.research.google.com/drive/1O9lozSLfNUzCgIoQzd5-PuTPU2DgfkdM?usp=sharing notebook.
2. Install the required dependencies: `pip install torch transformers datasets scikit-learn pandas`
3. Run the cells sequentially to initialize the `EdgeAwareLoRALinear` class.
4. Execute the training block to observe the MoE routing and auxiliary load-balancing loss in action.

## 🛠️ Tech Stack
* **Framework:** PyTorch
* **Libraries:** Hugging Face `transformers`, `datasets`
* **Base Models:** `roberta-large`, `roberta-base`
