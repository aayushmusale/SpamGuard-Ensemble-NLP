---

# Email Spam vs Ham Classifier 📧

A robust machine learning solution for identifying spam emails using Natural Language Processing (NLP) and Ensemble Learning. This project benchmarks 11 different algorithms and implements a high-performance **Voting Classifier** to achieve maximum reliability in spam detection.

## 🚀 Project Overview

In the context of spam filtering, **Precision** is the critical metric. A False Positive (marking an important "Ham" email as "Spam") can result in missing urgent information. This project is engineered to minimize those errors by combining the strengths of multiple tree-based ensembles.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, NLTK, Seaborn, Matplotlib
* **NLP Techniques:** Tokenization, Stemming, Stop-word removal, and TF-IDF Vectorization
* **Ensemble Methods:** Voting Classifier (Soft Voting), Random Forest, Extra Trees, and XGBoost

## 📊 Performance Benchmarking

Before selecting the final ensemble, 11 models were evaluated based on their Accuracy and Precision.

| Algorithm | Accuracy | Precision |
| --- | --- | --- |
| **K-Nearest Neighbors (KN)** | 0.896 | 1.000 |
| **Random Forest (RF)** | 0.970 | 0.991 |
| **Extra Trees (ETC)** | 0.972 | 0.975 |
| **XGBoost (xgb)** | 0.974 | 0.961 |
| **Voting Classifier (Final)** | **Optimized**(0.975) | **High-Reliability**(1.000) |

> **Note:** While **xgb** showed the highest individual accuracy, the **Voting Classifier** was chosen as the final model to leverage the combined predictive power and stability of the top three performing models.

## 🧠 Model Architecture: The Voting Classifier

The final model is a **Soft Voting Classifier** that aggregates the probability outputs of three core models:

1. **Random Forest (RF)**
2. **Extra Trees Classifier (ETC)**
3. **XGBoost (xgb)**

The ensemble makes a weighted decision, significantly reducing variance and improving the decision boundary compared to any single model.

## 📁 Repository Structure

* `Task2.ipynb`: The complete end-to-end pipeline, from data cleaning and NLP preprocessing to model evaluation and ensembling.
* `spam.csv`: The dataset containing labeled email messages.
* `model.joblib`: The serialized, production-ready version of the final **Voting Classifier**.

## ⚙️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/[Your-Username]/email-spam-classifier.git

```


2. Install required dependencies:
```bash
pip install pandas scikit-learn nltk xgboost joblib

```


3. Run the notebook or load the saved model:
```python
import joblib
model = joblib.load('model.joblib')

```



---
