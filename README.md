Fraud Detection under Class Imbalance

A research-oriented study on fraud detection under extreme class imbalance using machine learning, ensemble models, and explainability techniques.

📌 Problem Statement

Fraud detection is a highly imbalanced classification problem where fraudulent transactions represent less than 0.2% of the data.

This project investigates:

Model performance under extreme class imbalance
Impact of resampling techniques (SMOTE)
Comparison of linear vs ensemble models
Trade-offs between precision and recall
📊 Dataset
Credit Card Fraud Detection dataset
284,807 transactions
Fraud cases: ~0.17%

⚠️ Dataset not included due to size
👉 Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

⚙️ Methodology
1. Preprocessing
Stratified train-test split
Clean pipeline design
2. Models Used
Logistic Regression
Random Forest
XGBoost
3. Imbalance Handling
SMOTE (Synthetic Minority Oversampling Technique)
📈 Results
Model	Precision	Recall
Logistic Regression	0.83	0.63
Random Forest	0.94	0.82
XGBoost	0.92	0.78
XGBoost + SMOTE	0.38	0.88
⚖️ Precision–Recall Trade-off
Without SMOTE → High precision, lower recall
With SMOTE → High recall, very low precision

👉 Demonstrates real-world trade-off in fraud detection systems.

🔍 Explainability

SHAP analysis highlights key features contributing to fraud prediction, revealing hidden transactional patterns.

(Add SHAP image here)

🧠 Key Insights
Ensemble models outperform linear models
SMOTE improves recall but harms precision
Random Forest provides best real-world balance
🚀 How to Run
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/explainability.py
🎯 Conclusion

This project highlights the importance of model selection and evaluation metrics in imbalanced learning problems, emphasizing trade-offs between detecting fraud and minimizing false alarms.
