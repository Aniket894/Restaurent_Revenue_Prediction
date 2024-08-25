from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        # Load the pre-trained model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Transform the data using the preprocessor
        processed_data = self.preprocessor.transform(input_df)
        # Make predictions using the model
        predictions = self.model.predict(processed_data)
        return predictions.tolist()
    
# Initialize the prediction pipeline with model and preprocessor paths
prediction_pipeline = PredictionPipeline(
    model_path='artifacts/XGBRegressor.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    input_data = {
        'Location': request.form.get('Location'),
        'Cuisine': request.form.get('Cuisine'),
        'Rating': float(request.form.get('Rating')),
        'Seating Capacity': int(request.form.get('Seating Capacity')),
        'Average Meal Price': float(request.form.get('Average Meal Price')),
        'Marketing Budget': int(request.form.get('Marketing Budget')),
        'Social Media Followers': int(request.form.get('Social Media Followers')),
        'Chef Experience Years': int(request.form.get('Chef Experience Years')),
        'Number of Reviews': int(request.form.get('Number of Reviews')),
        'Avg Review Length': float(request.form.get('Avg Review Length')),
        'Ambience Score': float(request.form.get('Ambience Score')),
        'Service Quality Score': float(request.form.get('Service Quality Score')),
        'Parking Availability': request.form.get('Parking Availability'),
        'Weekend Reservations': int(request.form.get('Weekend Reservations')),
        'Weekday Reservations': int(request.form.get('Weekday Reservations'))
    }

    # Make prediction
    predictions = prediction_pipeline.predict(input_data)
    
    return render_template('results.html', predictions=predictions)
    
if __name__ == '__main__':
    app.run(debug=True)
