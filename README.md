# Patient Healthcare Monitoring - Risk Level Prediction

This project implements a machine learning solution to predict patient health risk levels based on vital signs such as heart rate, blood pressure, oxygen level, and body temperature.

## Project Overview

Accurately assessing patient risk levels can improve healthcare outcomes by enabling timely interventions. This project uses a **Random Forest Classifier** trained on patient vital sign data to classify risk into three categories:

- **Low Risk (0)**
- **Moderate Risk (1)**
- **High Risk (2)**

## Features

- Heart rate
- Systolic blood pressure
- Diastolic blood pressure
- Oxygen saturation level
- Body temperature

## Technologies Used

- Python
- pandas (data processing)
- scikit-learn (machine learning)
- joblib (model persistence)
- NumPy (numerical operations)

## Project Structure

- `data/health_data.csv` — Dataset containing patient vital signs and risk labels
- `scripts/train_model.py` — Script to train and save the Random Forest model
- `scripts/predict.py` — Script to load the saved model and predict risk level for new patient data
- `model/health_monitor_model.pkl` — Saved trained model (optional to include)
- `venv/` — Python virtual environment folder (ignored in Git)

## Usage

1. **Train the model:**

```bash
python scripts/train_model.py
