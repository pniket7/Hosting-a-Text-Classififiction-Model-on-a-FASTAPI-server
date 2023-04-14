` `**README**

# **1. PROJECT NAME:**
- Hosting a Text Classififiction Model with Logistic Regression on a FASTAPI server

**2. PROJECT OVERVIEW:**

- This project is an example of a text classification model using logistic regression.
- It uses the pandas, scikit-learn, fastapi, and joblib libraries for data handling, machine learning, and building a REST API.
- The goal of this project is to train a logistic regression model on text data to predict whether a given text belongs to a certain class (Non\_Acceptance) based on the input text.

**3. FEATURES:**

- Loads data from a CSV file (Normalized\_Data.csv) using pandas.
- Performs text classification using logistic regression from scikit-learn.
- Implements a REST API using fastapi for making predictions on new text inputs.
- Saves and loads the trained model using joblib.

**4. INSTALLATIONS:**

- To install the project dependencies, run the following command in your project environment:

`       `pip install pandas scikit-learn fastapi joblib uvicorn pydantic

**5. USAGE:**

- Load the data: The input data for this project is expected to be in a CSV file format. Make sure to provide the correct file path in the code to load the data.
- Train the model: The code provided trains a logistic regression model on the loaded text data. It performs data preprocessing using “CountVectorizer” for text vectorization, and then fits the logistic regression model using “LogisticRegression”. The best hyperparameters of the model are found using “RandomizedSearchCV”.
- ` `Save the model: The trained model is saved using “joblib” to a file named “model.joblib”.
- Run the API: The REST API is implemented using “fastapi” and can be run using “uvicorn”. The API provides a single endpoint “/predict” for making predictions on new text inputs. The input text is sent as a POST request to the “/predict” endpoint in JSON format, and the API returns the predicted class (0 or 1) as a response.
- Make predictions: To make predictions, send a POST request to the “/predict” endpoint of the API with the input text as a JSON payload. The API will return the predicted class (0 or 1) based on the trained logistic regression model.

**6. API REFERENCES:**

- “POST /predict”: Endpoint for making predictions on new text inputs. Expects input text as a JSON payload with the key "text". Returns the predicted class (0 or 1) as a JSON response.

**7. LICENSE:**

- This project is licensed under the MIT LICENSE.

**8.** **CONTACT INFORMATION:**

- For any questions or feedback, please contact:

Name- Niket Virendra Patil

Email- pniket7@gmail.com



