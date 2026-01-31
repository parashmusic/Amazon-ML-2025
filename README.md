# 🏷️ Smart Product Pricing - Amazon ML Challenge 2025

> **An efficient, feature-engineering-driven approach to e-commerce price prediction.**

## 📖 Overview
This repository contains our solution for the Amazon ML Challenge 2025. Our objective was to predict the price of diverse products based solely on their textual catalogue descriptions.

Instead of relying on computationally expensive Large Language Models (LLMs) or Transformers (like BERT/RoBERTa), we adopted a **"Feature Engineering First"** strategy, achieving a **SMAPE score of 48**. We built a custom extraction engine that parses unstructured text into structured technical specifications—such as RAM, Storage, Wattage, and Brand Quality—and fed these rich features into a **LightGBM** regressor.

## 🧠 Methodology
Our approach is a hybrid of Classical NLP and Rule-Based Extraction:

1.  **Text Parsing (`MaxAccuracyFeatureEngineer`)**:
    *   **Regex Extraction**: We developed over 15 targeted regex patterns to mine quantitative specs from the text (e.g., converting "1TB" and "1024GB" to a normalized value).
    *   **Heuristics**: Created flags for "Premium Brands", "Warranty", "Pack Quantity", and "Quality Tiers".
2.  **Vectorization**: Used TF-IDF (1-3 ngrams) to capture semantic keywords.
3.  **Modeling**:
    *   **LightGBM Regressor**: Fast, accurate gradient boosting capable of handling sparse inputs and missing values.
    *   **Log-Transformation**: Applied `log1p` to the target price to handle the extreme variance in product costs (long-tail distribution).
    *   **Robust Scaling**: Handled outliers in the feature space effectively.

## 💡 Key Learnings & "The Senior Feedback"
We received valuable insights from senior data scientists regarding our performance relative to the leaderboard.

### ✅ What We Did Right: Efficiency Over Complexity
Many participants tried to throw heavy Transformer models (DeBERTa, Electra) at the problem, often achieving error scores around **44%**. Our solution achieved a **SMAPE score of 48**, demonstrating that domain-specific feature engineering combined with efficient GBDT (Gradient Boosting) provides a strong alternative to heavy deep learning models with a fraction of the compute time. We effectively demonstrated that **understanding the data** (e.g., *16GB RAM is worth more than 8GB*) is often more valuable than generic semantic embeddings for pricing tasks.

### ⚠️ Mistakes & Future Improvements: The Ensemble Gap
**Our Mistake:** We relied too heavily on a single, well-tuned LightGBM model.
**The Lesson:** While our single-model approach was efficient and "pretty good," breaking the ceiling to reach the top-tier error rates (**42-42.5%**) requires **Ensembling**.
To improve further, we should have:
1.  **Diversified Models:** Trained distinct models (e.g., CatBoost, XGBoost, and a small Neural Net) alongside our LightGBM.
2.  **Stacking/Blending:** Combined predictions from these diverse models to cancel out individual biases.

## 🚀 How to Run
1.  Place `train.csv` and `test.csv` in the `dataset` folder.
2.  Run the notebook or script:
    ```bash
    python 7.py
    # OR
    jupyter notebook 7.ipynb
    ```
3.  The model will output a clean submission file in the dataset directory.

---
