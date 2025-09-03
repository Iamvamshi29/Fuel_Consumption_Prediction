🚗 Fuel Consumption & Driving Behavior Analyzer

This project is a Fuel Consumption and Driving Behavior Analyzer built with HTML, CSS, JavaScript, and Python (Flask).
It predicts fuel consumption and classifies driving behavior as Economical or Non-Economical using machine learning models.

🔹 Features

📊 Fuel Consumption Visualization: Interactive graphs showing fuel usage trends over time.

🚦 Driving Profile Classification: Real-time probability analysis of economical vs. non-economical driving.

⚙️ Custom Parameter Inputs: Users can provide engine RPM, speed, throttle position, acceleration, engine load, and other optional parameters.

🤖 Machine Learning Predictions: Uses regression and classification models for accurate predictions.

📈 Model Comparison Dashboard: Displays performance metrics (R², MSE, MAE) for different ML models.

🔹 Machine Learning Models Used

Ridge Regression – Best performance for fuel consumption prediction

XGBoost Regressor

SVR (Support Vector Regressor)

Random Forest Regressor

MLP Regressor (Neural Network)

Logistic Regression – For driving behavior classification

🔹 Tech Stack

Frontend: HTML, CSS, JavaScript

Backend: Python (Flask)

Visualization: Matplotlib, Plotly

ML/AI Libraries: Scikit-learn, XGBoost

🔹 Screenshots
Fuel Consumption & Driving Behavior Graphs




Input Parameters & Predictions




Prediction Results




🔹 Example Predictions

✅ Economical Driving

RPM: 2000, Speed: 60 km/h, Throttle: 40%, Acceleration: 0.5 m/s², Load: 50%

Predicted Consumption: 6.54 L/h (Economical)

❌ Non-Economical Driving

RPM: 3000, Speed: 90 km/h, Throttle: 60%, Acceleration: 1.6 m/s², Load: 65%

Predicted Consumption: 9.84 L/h (Non-Economical)

🔹 How to Run

Clone the repository

git clone https://github.com/yourusername/fuel-consumption-analyzer.git
cd fuel-consumption-analyzer


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py


Open in browser: http://127.0.0.1:5000/

🔹 Future Improvements

🔧 Integration with real-time OBD-II sensor data

📱 Mobile-friendly responsive dashboard

🚀 Deploy on cloud for live monitoring
