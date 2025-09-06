# 🎓 Student Stress Predictor

A machine learning–powered dashboard that predicts a student's **stress type** —  
**No Stress**, **Eustress (Positive Stress)**, or **Distress (Negative Stress)** —  
based on survey responses covering academic, emotional, and lifestyle factors.

The app is built with **Streamlit** for interactive deployment and uses **SHAP** for explainable AI, so predictions are transparent and actionable.
The detailed modeling process is provided in the attached Google Colaboratory file.

---

## ✨ Features

- 📥 **Input Student Responses** via an interactive questionnaire
- 🤖 **Predict Stress Type** using a trained ML model (SVM)
- 🔍 **Explain Predictions**
- 🎯 **Tailored Recommendations** for each stress type
- 📊 **Visual Insights**

---

## 🧠 How It Works

1. **Data Collection**  
   Survey questions capture academic confidence, workload, emotional well-being, and physical health.

2. **Model Training**  
   Multiple models (KNN, Random Forest, SVM, XGBoost) are trained and evaluated using:
   - Precision, Recall, F1-score
   - Macro-averaged metrics for class balance
   - ROC/AUC for Distress detection

3. **Interpretability**  
   SHAP values highlight the most influential features for each prediction.

4. **Deployment**  
   The best-performing model is deployed in a Streamlit app for real-time use.

---

## 📊 Example Visualizations

- **Bar Chart** for stress detection probability

---

## 🚀 Getting Started

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://intanannisa-student-stress-prediction.streamlit.app)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/intanurobiannisa/student-stress-predictor.git
cd student-stress-predictor
