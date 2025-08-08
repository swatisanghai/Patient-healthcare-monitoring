import joblib
import pandas as pd

# Load trained model
model = joblib.load('model/health_monitor_model.pkl')

# Example patient data to predict:
# Format: [heart_rate, bp_systolic, bp_diastolic, oxygen_level, temperature]
sample_data = pd.DataFrame([[85, 130, 85, 96, 98.6]],
                           columns=['heart_rate', 'bp_systolic', 'bp_diastolic', 'oxygen_level', 'temperature'])

# Make prediction
prediction = model.predict(sample_data)[0]

risk_map = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}

print(f"Predicted patient risk level: {risk_map.get(prediction, 'Unknown')}")
