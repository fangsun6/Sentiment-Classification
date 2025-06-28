
# 💬 Sentiment Analysis on Amazon Reviews using DistilBERT

This project tackles the task of **binary sentiment classification** using a layered approach: starting from classical models like TF-IDF + Logistic Regression, then scaling up to **transformer-based models** — specifically, DistilBERT. It emphasizes not only performance but also clarity, experimentation rigor, and real-world deployment readiness.

---

## 🎯 Project Motivation

Sentiment analysis has wide applications in product reviews, social media monitoring, and customer service analytics. My personal motivation was to:
- Compare **legacy NLP models** with modern **transfer learning** approaches.
- Understand bottlenecks and trade-offs in training large models.
- Push myself to reflect on model choice, optimization, and deployment.

Rather than jumping to BERT or any massive model, I deliberately followed a **progressive path**:
1. Build trust in the data.
2. Validate with simpler models.
3. Scale to more powerful models only when justified.

---

## 📦 Dataset

- **Size**: 360,000 Amazon customer reviews
- **Label**: Binary sentiment — `1` (positive), `0` (negative)
- **Source**: Preprocessed version loaded directly in the notebook
- **Challenge**: Natural variance in tone, sarcasm, negation — not trivial for classical models

---

## 🧠 Thought Process & Modeling Strategy

### Phase 1: Baseline with TF-IDF + Logistic Regression
- 🌱 **Goal**: Establish a fast, interpretable reference model.
- 🧪 **Finding**: TF-IDF performed decently but often failed on sarcastic or context-heavy phrases.
- ✅ **Takeaway**: Great for prototyping, but inadequate for nuanced semantics.

### Phase 2: Deep EDA
- Analyzed review lengths (avg. ≈ 80–100 words).
- Token count distribution guided `max_len=128` decision.
- Word cloud visualizations highlighted most common positive/negative tokens.

### Phase 3: DistilBERT Fine-tuning
- **Tokenizer**: `DistilBertTokenizerFast` — fast & handles subword tokenization
- **Model**: `DistilBertForSequenceClassification`
- **Training API**: Hugging Face `Trainer` for simplified training/validation loop

---

## 🧩 Engineering & Model Design Decisions

| Decision | Reason |
|---------|--------|
| Chose DistilBERT over full BERT | 40% smaller, 60% faster, almost same accuracy |
| Used Hugging Face Trainer API | Abstracts low-level training logic and handles metrics |
| Set `max_len=128` | Based on 90th percentile of review token lengths |
| Truncated & padded input | Avoid token overflow and maintain consistent batch size |
| Mixed-precision training | Used FP16 on Colab to reduce memory use and accelerate training |

---

## ⚠️ Key Challenges and How I Solved Them

| Challenge | How I Tackled It |
|----------|-------------------|
| **Subtle sentiment phrases** | Used transformer attention to capture long-range dependencies |
| **Out-of-vocabulary (OOV) words** in TF-IDF | Switched to subword tokenizer with DistilBERT |
| **Slow training time** on large dataset | Sampled dataset, enabled early stopping, reduced epochs |
| **Overfitting on small batch sizes** | Used dropout + weight decay + validation monitoring |
| **Monitoring metrics** during training | Used Hugging Face's built-in callbacks and logging |
| **Token limit (512)** | Analyzed length and chose 128 as optimal tradeoff |
| **Deployment readiness** | Explored TorchScript and ONNX export paths |

---

## 📊 Results

| Model | Accuracy | F1-Score | Notes |
|-------|----------|----------|-------|
| TF-IDF + Logistic Regression | ~88% | ~0.87 | Fast, interpretable, brittle |
| DistilBERT | ~95% | ~0.95 | Slower, robust to nuanced expressions |

---

## 🔍 Reflections

This project was a rich learning journey across:
- **Classical NLP foundations**
- **State-of-the-art transformers**
- **Hardware-aware optimization**
- **Model interpretability vs. power trade-off**

I learned that:
- Simpler models help build understanding and spot-check pipeline logic.
- DistilBERT is an excellent compromise between performance and efficiency.
- Thoughtful preprocessing and evaluation are as important as model choice.

---

## 📁 Directory Structure

```
.
├── sentiment-analysis-distilbert-amazon-reviews.ipynb  # Complete pipeline from EDA to fine-tuning
├── README.md                                            # Full project report & commentary
```

---

## ▶️ How to Run

Install all required libraries:

```bash
pip install pandas numpy matplotlib seaborn torch transformers datasets scikit-learn
```

Then open the notebook in Jupyter or Colab and run all cells in sequence. GPU is strongly recommended for fine-tuning.

---

## 🚀 Future Enhancements

- 💬 Add LIME/SHAP for interpretability on DistilBERT outputs
- 🌍 Extend to multilingual sentiment datasets (e.g., Amazon ES/FR)
- 🧠 Try zero-shot classification using `pipeline(task="zero-shot-classification")`
- 🔌 Serve model via FastAPI + TorchScript or ONNX for real-time use

---

## 📜 License

MIT License — for research, learning, and adaptation purposes.

---
