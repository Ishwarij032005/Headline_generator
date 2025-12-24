# Headline Generator (Hybrid Seq2Seq + Extractive Model)

This project implements a **Hybrid Extractive–Abstractive Headline Generation model**
using a custom Seq2Seq architecture with attention.

## Features
- Lead-based extractive preprocessing
- Custom tokenizer and vocabulary
- Bi-directional LSTM encoder
- Attention-based decoder
- ROUGE evaluation (ROUGE-1, ROUGE-2, ROUGE-L)

## Dataset
- CNN/DailyMail (via HuggingFace Datasets)

## Tech Stack
- Python
- PyTorch
- HuggingFace Datasets
- NLTK
- ROUGE Score

## Results
- ROUGE-1 ≈ 30%
- ROUGE-2 ≈ 12%
- ROUGE-L ≈ 20%

## Author
**Ishwari Jamadade**
