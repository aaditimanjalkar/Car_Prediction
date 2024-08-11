from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and preprocessor
model, preprocessor = joblib.load('car_price_model_preprocessor.pkl')

# Load the dataset to get unique values
dataset = pd.read_csv('indian-auto-mpg.csv')

# Extract unique values for dropdowns
unique_manufacturers = dataset['Manufacturer'].unique().tolist()
unique_locations = dataset['Location'].unique().tolist()
unique_fuel_types = dataset['Fuel_Type'].unique().tolist()
unique_owner_types = dataset['Owner_Type'].unique().tolist()
unique_transmissions = dataset['Transmission'].unique().tolist()
unique_seats = dataset['Seats'].unique().astype(int).tolist()  # Convert to int for display

@app.route('/')
def form():
    return render_template('form.html', 
                           manufacturers=unique_manufacturers,
                           locations=unique_locations,
                           fuel_types=unique_fuel_types,
                           owner_types=unique_owner_types,
                           transmissions=unique_transmissions,
                           seats=unique_seats)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    data = {
        'Manufacturer': [request.form['Manufacturer']],
        'Location': [request.form['Location']],
        'Year': [request.form['Year']],
        'Kilometers_Driven': [request.form['Kilometers_Driven']],
        'Fuel_Type': [request.form['Fuel_Type']],
        'Transmission': [request.form['Transmission']],
        'Owner_Type': [request.form['Owner_Type']],
        'Engine CC': [request.form['Engine_CC']],
        'Power': [request.form['Power']],
        'Seats': [request.form['Seats']],
        'Mileage Km/L': [request.form['Mileage_Km_L']]
    }

    # Convert to DataFrame
    new_data = pd.DataFrame(data)

    # Preprocess the new data
    new_data_preprocessed = preprocessor.transform(new_data)

    # Make predictions
    prediction = model.predict(new_data_preprocessed)

    # Render result template with prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
