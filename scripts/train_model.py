import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('data/health_data.csv')

# Features and target
X = data[['heart_rate', 'bp_systolic', 'bp_diastolic', 'oxygen_level', 'temperature']]
y = data['risk_level']

# Split data (80% train, 20% test) with stratify to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save trained model
joblib.dump(clf, 'model/health_monitor_model.pkl')
print("Model trained and saved as 'model/health_monitor_model.pkl'")

# Optional: print training and testing accuracy
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Testing Accuracy: {test_acc:.2f}")

# Print label distribution to help debug imbalance
print("\nTrain label distribution:\n", y_train.value_counts(normalize=True))
print("\nTest label distribution:\n", y_test.value_counts(normalize=True))
