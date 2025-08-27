import streamlit as st
import pandas as pd
import numpy as np
import joblib
# import shap
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# Load trained model (SVM)
model = joblib.load('data/best_model.pkl')

# Load SHAP explainer
# X_train = pd.read_csv("data/X_train.csv")
# feature_data_sample = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
# explainer = shap.Explainer(model.predict_proba, feature_data_sample)  # Use KernelExplainer if needed

# Define input questions
questions = [
    "Have you recently experienced stress in your life?",
    "Do you face any sleep problems or difficulties falling asleep?",
    "Are you facing any difficulties with your professors or instructors?",
    "Do you have trouble concentrating on your academic tasks?",
    "Have you gained/lost weight?",
    "Do you find that your relationship often causes you stress?",
    "Have you been getting headaches more often than usual?",
    "Is your working environment unpleasant or stressful?",
    "Have you been feeling sadness or low mood?",
    "Do you get irritated easily?",
    "Have you noticed a rapid heartbeat or palpitations?",
    "Have you been experiencing any illness or health issues?",
    "Academic and extracurricular activities conflicting for you?",
    "Do you attend classes regularly?",
    "Do you lack confidence in your academic performance?",
    "Have you been dealing with anxiety or tension recently?",
    "Is your hostel or home environment causing you difficulties?",
    "Do you often feel lonely or isolated?",
    "Do you struggle to find time for relaxation and leisure activities?",
    "Do you feel overwhelmed with your academic workload?",
    "Are you in competition with your peers, and does it affect you?",
    "Do you lack confidence in your choice of academic subjects?"
]

# Streamlit UI
st.title("🎓 Student Stress Prediction Dashboard")
st.write(
    """Created by Intan Nur Robi Annisa – student of Data Science and Data Analyst Bootcamp at Dibimbing.  
    [LinkedIn Profile](https://www.linkedin.com/in/intannurrobiannisa)"""
)

st.subheader("Predict Your Stress Type 😌😰😖")
st.write("Input your responses below to see your predicted stress type and personalized recommendations.")
st.markdown("All responses are on a five-point Likert scale, ranging from 1:‘Not at all’ to 5:‘Extremely’.")
    
# Collect user input
user_input = []

gender = st.selectbox("Select your Gender:", ["Male", "Female"])
gender_encoded = 0 if gender == "Male" else 1
user_input.append(gender_encoded)

age = st.number_input("Enter your Age:", min_value=10, max_value=100, value=20)
user_input.append(age)

for q in questions:
    response = st.slider(q, 1, 5, 3)
    user_input.append(response)

# Convert to DataFrame
input_df = pd.DataFrame([user_input], columns=['Gender', 'Age'] + questions)
input_df.rename(columns={
    'Have you been dealing with anxiety or tension recently?': 'Have you been dealing with anxiety or tension recently?.1'
}, inplace=True)
new_order = [
    "Gender", 
    'Age', 
    "Have you recently experienced stress in your life?",
    "Have you noticed a rapid heartbeat or palpitations?",
    "Do you face any sleep problems or difficulties falling asleep?",
    "Have you been dealing with anxiety or tension recently?.1",
    "Have you been getting headaches more often than usual?",
    "Do you get irritated easily?",
    "Do you have trouble concentrating on your academic tasks?",
    "Have you been feeling sadness or low mood?",
    "Have you been experiencing any illness or health issues?",
    "Do you often feel lonely or isolated?",
    "Do you feel overwhelmed with your academic workload?",
    "Are you in competition with your peers, and does it affect you?",
    "Do you find that your relationship often causes you stress?",
    "Are you facing any difficulties with your professors or instructors?",
    "Is your working environment unpleasant or stressful?",
    "Do you struggle to find time for relaxation and leisure activities?",
    "Is your hostel or home environment causing you difficulties?",
    "Do you lack confidence in your academic performance?",
    "Do you lack confidence in your choice of academic subjects?",
    "Academic and extracurricular activities conflicting for you?",
    "Do you attend classes regularly?",
    "Have you gained/lost weight?"]
input_df = input_df[new_order]

# # Display input summary
# st.subheader("📋 Your Input Summary")
# st.dataframe(input_df)

# Scale if needed
scaler = joblib.load('data/scaler.pkl')
input_df['Age_Scaled'] = scaler.transform(input_df[['Age']])
input_df = input_df.drop(columns='Age')

# Wait for user to click before predicting
if st.button("Predict Stress Level"):
    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # Map prediction to label
    stress_map = {0: "No Stress", 1: "Eustress", 2: "Distress"}
    st.subheader(f"🧠 Predicted Stress Type: **{stress_map[prediction]}**")

    # Show probabilities
    st.write("Prediction Confidence:")
    st.bar_chart(pd.Series(proba, index=[stress_map[i] for i in range(3)]))

    # SHAP explanation
    # st.subheader("🔍 Top Contributing Factors")
    # shap_values = explainer(input_df)
    # shap.plots.bar(shap_values[0], show=False)
    # st.pyplot(plt.gcf())
    # plt.clf()

    # Tailored recommendations
    st.subheader("🎯 Recommended Interventions")
    if prediction == 2:
        st.markdown("- Connect with a counselor or mental health professional")
        st.markdown("- Prioritize sleep and relaxation routines")
        st.markdown("- Seek academic support for workload management")
    elif prediction == 1:
        st.markdown("- Maintain healthy stress levels through time management")
        st.markdown("- Use stress as motivation—keep tracking your goals")
    else:
        st.markdown("- Keep up the good habits!")
        st.markdown("- Stay socially connected and monitor for any changes")
