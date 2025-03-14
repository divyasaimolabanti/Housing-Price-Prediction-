from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

# Define feature columns
feature_columns = ["median_income", "total_rooms", "housing_median_age", "ocean_proximity"]

# HTML Template with Styling
home_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f4f4f4;
            font-family: Arial, sans-serif;
        }
        .container {
            max-width: 500px;
            margin-top: 50px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            background: #d4edda;
            border-left: 5px solid #28a745;
            color: #155724;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center">House Price Prediction</h2>
        <form action="/predict" method="post">
            <div class="mb-3">
                <label for="median_income" class="form-label">Median Income:</label>
                <input type="text" class="form-control" id="median_income" name="median_income" required>
            </div>

            <div class="mb-3">
                <label for="total_rooms" class="form-label">Total Rooms:</label>
                <input type="number" class="form-control" id="total_rooms" name="total_rooms" required>
            </div>

            <div class="mb-3">
                <label for="housing_median_age" class="form-label">Housing Median Age:</label>
                <input type="number" class="form-control" id="housing_median_age" name="housing_median_age" required>
            </div>

            <div class="mb-3">
                <label for="ocean_proximity" class="form-label">Ocean Proximity:</label>
                <select class="form-control" id="ocean_proximity" name="ocean_proximity">
                    <option value="1H OCEAN">1H OCEAN</option>
                    <option value="INLAND">INLAND</option>
                    <option value="ISLAND">ISLAND</option>
                    <option value="NEAR BAY">NEAR BAY</option>
                    <option value="NEAR OCEAN">NEAR OCEAN</option>
                </select>
            </div>

            <button type="submit" class="btn btn-primary w-100">Predict Price</button>
        </form>

        {% if predicted_price %}
            <div class="result-box mt-3">
                <h4>Estimated House Price: ${{ predicted_price }}</h4>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(home_page)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get request data
        data = request.form.to_dict()

        # Extract numerical features
        numeric_data = {
            "median_income": float(data["median_income"]),
            "total_rooms": float(data["total_rooms"]),
            "housing_median_age": float(data["housing_median_age"]),
        }

        # Get ocean_proximity
        ocean_proximity = data.get("ocean_proximity", "NEAR BAY")

        # Prepare input DataFrame
        input_data = {**numeric_data, "ocean_proximity": ocean_proximity}
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Predict price
        predicted_price = model.predict(input_df)[0]

        # Render page with prediction
        return render_template_string(home_page, predicted_price=round(predicted_price, 2))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=False)
