# Housing Price Prediction - Machine Learning Pipeline and API

This project provides a machine learning pipeline for predicting house prices using various features. The model is served via a Flask web application, allowing users to input house features and get price predictions.

## Table of Contents

- [1. Install the Required Libraries](#1-install-the-required-libraries)
- [2. Train the Model and Save the Artifacts](#2-train-the-model-and-save-the-artifacts)
- [3. Start the Flask Web Application](#3-start-the-flask-web-application)
- [4. Access the Web Application](#4-access-the-web-application)
- [5. Notes](#5-notes)

---

## 1. Install the Required Libraries

Before running the project, make sure to install the necessary dependencies. Open a terminal or command prompt and run the following command:

```
pip install flask scikit-learn pandas numpy matplotlib seaborn joblib 

```
## 2. Train the Model and Save the Artifacts
Run the Python script to preprocess the data, train the model, and save it. Execute the following command:

```
python Assignment3.py

```

Once executed, the script will generate:

A trained model (house_price_model.pkl).

Any necessary preprocessing files.

## 3. Start the Flask Web Application
Navigate to the directory containing app.py and run the following command:

```
python app.py

```

If everything is set up correctly, the Flask server will start and display the following message:

```
Running on http://127.0.0.1:5002/
```
This means the API is live and ready to accept input.

## 4. Access the Web Application

Open a web browser and go to http://127.0.0.1:5002/.

Enter house features in the form and click the Predict button.

The predicted house price will be displayed on the screen.

Notes:
Ensure all files (Assignment3.py, app.py, and any dependencies) are in the correct directory.

If you encounter any issues, check the terminal logs for error messages.