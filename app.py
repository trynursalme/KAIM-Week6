from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load both models
logistic_model = joblib.load("models/updated_logistic_regression_model.pkl")
random_forest_model = joblib.load("models/updated_random_forest_model.pkl")

# Choose the best model (Random Forest is the better model)
best_model = random_forest_model

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the form
            std_transaction_amount = float(request.form['std_transaction_amount'])
            total_transaction_amount = float(request.form['total_transaction_amount'])
            transaction_count = int(request.form['transaction_count'])
            average_transaction_amount = float(request.form['average_transaction_amount'])

            # Prepare the input features for prediction
            features = np.array([[std_transaction_amount, total_transaction_amount, 
                                  transaction_count, average_transaction_amount]])

            # Make prediction using the best model
            prediction = best_model.predict(features)[0]
            probability = best_model.predict_proba(features)[0][int(prediction)]

            # Return the result to the user
            result = f"Predicted Class: {int(prediction)}, Probability: {probability:.2f}"
        except Exception as e:
            result = f"Error in prediction: {str(e)}"
        return render_template('predict.html', result=result)
    return render_template('predict.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
