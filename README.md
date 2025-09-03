##🚗 Fuel Consumption & Driving Behavior Analyzer

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
<img width="1077" height="605" alt="drivinggraph" src="https://github.com/user-attachments/assets/c1468a5f-57c5-4693-8119-100aee16beba" />

Input Parameters & Predictions
<img width="1009" height="553" alt="graph" src="https://github.com/user-attachments/assets/ad6e8548-597c-461f-a002-f7a317273030" />
<img width="1056" height="593" alt="Predict1" src="https://github.com/user-attachments/assets/72cf5e5a-d360-4a02-92e8-05642b8e5d97" />
<img width="1056" height="594" alt="Predict2" src="https://github.com/user-attachments/assets/054b3458-0e95-4c0d-b86e-324b75cea642" />

Prediction Results
<img width="1076" height="605" alt="Result" src="https://github.com/user-attachments/assets/658abdf1-8f28-4ff0-9090-c0478f736646" />
<img width="1114" height="626" alt="Result2" src="https://github.com/user-attachments/assets/e47dff9a-da3d-412c-93da-2ab8e46be383" />




🔹 Example Predictions

✅ Economical Driving

RPM: 2000, Speed: 60 km/h, Throttle: 40%, Acceleration: 0.5 m/s², Load: 50%

Predicted Consumption: 6.54 L/h (Economical)

❌ Non-Economical Driving

RPM: 3000, Speed: 90 km/h, Throttle: 60%, Acceleration: 1.6 m/s², Load: 65%

Predicted Consumption: 9.84 L/h (Non-Economical)

🔹 How to Run

Clone the repository

git clone(https://github.com/Iamvamshi29/Fuel_Consumption_Prediction.git)
cd fuel-consumption-analyzer

Install dependencies

pip install -r requirements.txt

Open VSCode 
Run app

Run the Flask app

python app.py


Open in browser: http://127.0.0.1:5000/

🔹 Future Improvements

🔧 Integration with real-time OBD-II sensor data

📱 Mobile-friendly responsive dashboard

🚀 Deploy on cloud for live monitoring
