# 📰 Headline Generator (Hybrid Extractive + Abstractive Model)

A deep learning project that generates news headlines using a **Hybrid Extractive–Abstractive approach** built on a custom Seq2Seq architecture with attention.

---

## 🌍 Problem Statement

Nowadays, millions of news articles are published every day, making it difficult for readers to go through all of them. As a result, most people rely on headlines to quickly understand the context of an article.

However, generating headlines manually:
- Requires significant time and human effort  
- Can be inconsistent or biased  
- Does not scale efficiently with large volumes of content  

💡 This is where AI can help.

An intelligent system can automatically generate **accurate, concise, and informative headlines**, enabling readers to quickly grasp the essence of an article while saving time and effort.

---

## 🎯 Solution

This project builds a **Hybrid Extractive–Abstractive Headline Generation system** that combines:
- Extractive summarization (to capture key content)
- Abstractive generation (to create human-like headlines)

The model leverages **Seq2Seq architecture with Attention** to generate meaningful and context-aware headlines.

---

## 🚀 Overview

The system follows a hybrid pipeline:
1. Extract key sentences using lead-based summarization  
2. Process input through a Bi-directional LSTM Encoder  
3. Apply Attention mechanism for context understanding  
4. Generate headline using LSTM Decoder  

---

## 🧠 Model Architecture

- Bi-directional LSTM Encoder  
- Attention-based Decoder  
- Seq2Seq Framework  
- Custom Tokenizer & Vocabulary  

---

## 📊 Dataset

- **CNN/DailyMail Dataset**
- Accessed using HuggingFace Datasets

---

## ⚙️ Tech Stack

- Python  
- PyTorch  
- HuggingFace Datasets  
- NLTK  
- ROUGE Score  

---

## 📈 Evaluation Metrics

| Metric   | Score |
|---------|------|
| ROUGE-1 | ~30% |
| ROUGE-2 | ~12% |
| ROUGE-L | ~20% |

---

## 💡 Key Highlights

- Built custom Seq2Seq model from scratch  
- Implemented Attention mechanism for better context learning  
- Combined extractive + abstractive techniques  
- Evaluated using industry-standard ROUGE metrics  

---

## 🎯 Learnings

- Sequence-to-Sequence modeling  
- Attention mechanisms in NLP  
- Text summarization techniques  
- Model evaluation using ROUGE  

---

## 🔗 Project Link

👉 https://github.com/Ishwarij032005/Headline_generator  

---

## 🏆 Conference Recognition

This work was presented at the  
**IEEE International Conference for Convergence in Computing Technology (I3CTCON 2026)**

---

## 👩‍💻 Author

**Ishwari Jamadade**
