from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('cancer_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        input_features = [float(x) for x in request.form.values()]
        input_array = np.array([input_features])

        # Make a prediction using the trained model
        prediction = model.predict(input_array)[0]
        prediction_label = 'Malignant' if prediction == 'M' else 'Benign'

        return render_template('index.html', prediction_text=f'The tumor is likely: {prediction_label}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
