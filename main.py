import numpy as np
import pickle



# Load the saved XGBoost model
with open('xgboost_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

print("Model loaded successfully!")


# Example customer data (replace with actual values matching your columns)
sample_customer = np.array([[100, 1, 10, 0, 12.5, 5, 3.0, 200, 100, 45.5, 180, 90, 35.0, 150, 80, 20.0, 1, 600, 100.0, 0.5, 0, 1, 0]])

# Predict churn for this customer
prediction = model.predict(sample_customer)

print("Predicted Churn:", "Yes" if prediction[0] == 1 else "No")
