import streamlit as st
import pickle
import numpy as np

# Load the saved XGBoost model
with open('xgboost_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App Header
st.title("📊 **Customer Churn Prediction Dashboard**")
st.markdown("""
    **Welcome to the Customer Churn Prediction App!**  
    This app uses an advanced machine learning model to predict if a customer is likely to **churn** (leave the service).
    Simply fill in the customer data below and click **"Predict Churn"** to see the result.
""")

# Instructions
st.markdown("""
    ### 📌 **Instructions:**
    1. Fill in the **customer details** in the form below.
    2. Click the **"Predict Churn"** button to get the prediction.
    3. **Result:** The app will show if the customer is likely to **leave** (churn) or **stay** with the service.
    4. **Important:** Make sure all input values are within the expected range!
""")

# Input fields with validation, warnings, and tooltips

# Account Length
account_length = st.number_input(
    'Account Length (in months)', 
    min_value=0, 
    max_value=240, 
    value=100, 
    help="The total number of months the customer has been with the service."
)
if account_length > 240:
    st.warning("⚠️ **Warning:** Account length exceeds the maximum of 240 months.")
elif account_length == 0:
    st.warning("⚠️ **Warning:** Account length cannot be 0 months, please enter a valid value.")

# Voice Plan - Using "Yes" and "No"
voice_plan = st.radio(
    'Does the customer have a Voice Plan?', 
    ['No', 'Yes'],
    help="Select 'Yes' if the customer has a voice plan, otherwise select 'No'."
)
voice_plan = 1 if voice_plan == 'Yes' else 0  # Map 'Yes' to 1 and 'No' to 0

if voice_plan not in [0, 1]:
    st.warning("⚠️ **Warning:** Please select a valid option for the voice plan.")

# Voice Messages
voice_messages = st.number_input(
    'Number of Voice Messages', 
    min_value=0, 
    value=10,
    help="The number of voice messages the customer has sent or received."
)
if voice_messages < 0:
    st.warning("⚠️ **Warning:** Number of voice messages cannot be negative.")

# International Plan - Using "Yes" and "No"
intl_plan = st.radio(
    'Does the customer have an International Plan?', 
    ['No', 'Yes'],
    help="Select 'Yes' if the customer has an international calling plan, otherwise select 'No'."
)
intl_plan = 1 if intl_plan == 'Yes' else 0  # Map 'Yes' to 1 and 'No' to 0

if intl_plan not in [0, 1]:
    st.warning("⚠️ **Warning:** Please select a valid option for the international plan.")

# International Minutes Used
intl_mins = st.number_input(
    'International Minutes Used', 
    min_value=0.0, 
    value=12.5, 
    step=0.1,
    help="Total number of minutes used by the customer for international calls."
)
if intl_mins < 0:
    st.warning("⚠️ **Warning:** International minutes cannot be negative.")

# International Calls Made
intl_calls = st.number_input(
    'International Calls Made', 
    min_value=0, 
    value=5,
    help="Total number of international calls made by the customer."
)
if intl_calls < 0:
    st.warning("⚠️ **Warning:** Number of international calls cannot be negative.")

# International Charges
intl_charge = st.number_input(
    'International Charges (in USD)', 
    min_value=0.0, 
    value=3.0,
    help="The total charges (in USD) for the customer's international usage."
)
if intl_charge < 0:
    st.warning("⚠️ **Warning:** International charges cannot be negative.")

# Collect the features in the same format as trained model expects
sample_customer = np.array([[account_length, voice_plan, voice_messages, intl_plan, intl_mins, intl_calls, intl_charge, 200, 100, 45.5, 180, 90, 35.0, 150, 80, 20.0, 1, 600, 100.0, 0.5, 0, 1, 0]])

# Button to trigger prediction
if st.button('🔮 Predict Churn'):
    prediction = model.predict(sample_customer)
    if prediction[0] == 1:
        st.success("🚨 **Prediction:** The customer is likely to **churn** (leave).")
    else:
        st.success("✅ **Prediction:** The customer is likely to **stay** with the service.")