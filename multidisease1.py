import streamlit as st
import numpy as np
import joblib

# Load the models
diabetes_model = joblib.load('diabetes.sav')
heart_disease_model = joblib.load('heartattack.sav')
parkinsons_model = joblib.load('parkisons.sav')

# Sidebar for navigation
st.sidebar.title("Multi Disease Prediction")
selected = st.sidebar.selectbox("Select the Disease", ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"])

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    # Input fields for diabetes
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Prediction
    diabetes_diagnosis = ''
    if st.button("Diabetes Test Result"):
        try:
            features = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diabetes_prediction = diabetes_model.predict([features])
            
            if diabetes_prediction[0] == 1:
                diabetes_diagnosis = "The person has diabetes"
            else:
                diabetes_diagnosis = "The person does not have diabetes"
            
            st.success(diabetes_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    # Input fields for heart disease
    Age = st.text_input('Age')
    Sex = st.text_input('Sex')
    CP = st.text_input('Chest Pain types')
    Trestbps = st.text_input('Resting Blood Pressure')
    Chol = st.text_input('Serum Cholestoral in mg/dl')
    Fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    Restecg = st.text_input('Resting Electrocardiographic results')
    Thalach = st.text_input('Maximum Heart Rate achieved')
    Exang = st.text_input('Exercise Induced Angina')
    Oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    Slope = st.text_input('Slope of the peak exercise ST segment')
    Ca = st.text_input('Major vessels colored by flourosopy')
    Thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # Prediction
    heart_disease_diagnosis = ''
    if st.button("Heart Disease Test Result"):
        try:
            features = [float(Age), float(Sex), float(CP), float(Trestbps), float(Chol), float(Fbs), float(Restecg), float(Thalach), float(Exang), float(Oldpeak), float(Slope), float(Ca), float(Thal)]
            heart_disease_prediction = heart_disease_model.predict([features])
            
            if heart_disease_prediction[0] == 1:
                heart_disease_diagnosis = "The person has heart disease"
            else:
                heart_disease_diagnosis = "The person does not have heart disease"
            
            st.success(heart_disease_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# Parkinson's Disease Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    # Input fields for Parkinson's disease
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DOP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input("Shimmer:APQ5")
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    # Prediction
    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            features = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs), float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
            parkinsons_prediction = parkinsons_model.predict([features])
            
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
            
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")