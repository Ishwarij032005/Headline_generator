# 📰 Headline Generator (Hybrid Extractive + Abstractive Model)

A deep learning project that generates news headlines using a **Hybrid Extractive–Abstractive approach** built on a custom Seq2Seq architecture with attention.

---

## 🚀 Overview
This project combines:
- **Extractive summarization** (lead-based preprocessing)
- **Abstractive generation** (Seq2Seq with Attention)

👉 Goal: Generate concise, meaningful headlines from long news articles.

---

## 🧠 Model Architecture

- Bi-directional LSTM Encoder  
- Attention-based Decoder  
- Custom Tokenizer & Vocabulary  
- Hybrid Pipeline:
  1. Extract key sentences (lead-based)
  2. Generate headline using Seq2Seq model

---

## 📊 Dataset
- **CNN/DailyMail Dataset**
- Accessed via HuggingFace Datasets

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
- Implemented custom Seq2Seq from scratch  
- Integrated attention mechanism for better context learning  
- Combined extractive + abstractive techniques for improved performance  
- Evaluated using industry-standard ROUGE metrics  

---

## 🎯 Learnings
- Sequence-to-Sequence modeling  
- Attention mechanisms in NLP  
- Text summarization techniques  
- Model evaluation in NLP  

---

## 🔗 Project Link
👉 https://github.com/Ishwarij032005/Headline_generator  

---

## 👩‍💻 Author
**Ishwari Jamadade**
