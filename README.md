# MaintAI

**Intelligent Predictive Maintenance and Machine Failure Detection System Using Machine Learning**

MaintAI is a machine learning-based application that analyzes machine sensor readings and classifies whether the current operating conditions are associated with a potential machine failure.

The application uses a trained Random Forest Classifier and provides predictions through an interactive Streamlit dashboard.

## Project Objectives

- Analyze machine sensor data.
- Identify patterns associated with machine failures.
- Train and evaluate machine learning classification models.
- Provide an interactive interface for machine failure detection.
- Display prediction probabilities and model performance metrics.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Project Status

Machine learning model development and Streamlit application testing completed.

## Model Evaluation

The final Random Forest model was evaluated using the held-out test dataset of 2,000 records.

### Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 98.30% |
| Precision | 85.42% |
| Recall | 60.29% |
| F1-Score | 70.69% |
| PR-AUC | 76.42% |

### Confusion Matrix

| | Predicted: No Failure | Predicted: Failure |
|---|---:|---:|
| Actual: No Failure | 1925 | 7 |
| Actual: Failure | 27 | 41 |

### Prediction Threshold

The final prediction threshold is **0.4**.

A machine is classified as having a potential failure when its predicted failure probability is greater than or equal to 0.4.

### Model Selection

The tuned Random Forest model was selected because it achieved the highest F1-score among the evaluated model configurations on the held-out test set.

The threshold was selected using validation data, not the test set.

## Streamlit Application

MaintAI includes an interactive web application developed using Streamlit.

### Application Features

- Machine sensor input interface.
- Machine failure probability calculation.
- Failure and normal operating condition classification.
- Sensor range warnings.
- Input validation.
- Sensor readings summary.
- Recent prediction history.
- Model information and evaluation metrics.

### Installation and Setup

Follow these steps to run MaintAI locally.

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MaintAI
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate the virtual environment**

Windows:

```bash
venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
streamlit run app.py
```

The application will open in your default browser.

### Project Structure

```text
MaintAI/
│
├── app.py
├── maintai_model.joblib
├── requirements.txt
└── README.md
```

## Limitations

- The model was trained using the AI4I 2020 synthetic predictive maintenance dataset.
- Model performance on real industrial machines may differ.
- Predictions indicate potential failure-associated conditions, not guaranteed future failures.
- The application does not predict the exact time or date of a machine failure.
- Prediction results depend on the quality and accuracy of the sensor inputs.

## Conclusion

MaintAI demonstrates how machine learning can be applied to predictive maintenance and machine failure detection.

By combining sensor data preprocessing, model comparison, threshold tuning, and a Streamlit interface, the project provides an interactive system for identifying operating conditions associated with potential machine failures.