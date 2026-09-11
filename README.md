# Wine Quality Classification Using Random Forest Machine Learning

## Project Overview

This project presents a machine learning system for classifying wine quality into three categories: **Low**, **Average**, and **High**.

The system uses chemical properties of wine as input and applies a **Balanced Random Forest Classifier** to predict the quality category. A **FastAPI** backend provides the prediction API, while a **Streamlit** frontend provides a user-friendly interface for making predictions.

The project was developed as part of a Machine Learning program at **Gworld Soft Solution Limited**, Mathematics Department.

---

## Project Objectives

The major objectives of this project are to:

1. Develop a machine learning model for wine quality classification.
2. Handle class imbalance in the wine quality dataset.
3. Compare different machine learning approaches.
4. Select a model based on balanced classification performance.
5. Develop an API for real-time wine quality prediction.
6. Develop an interactive web interface for users.
7. Organize the project into a professional and portable structure.

---

## Dataset

The dataset contains **1,359 wine records** and **11 chemical features** used for prediction.

### Input Features

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol

### Target Variable

The target variable is:

`quality_category`

The original wine quality scores were grouped into three categories:

| Quality Score | Category |
| ------------- | -------- |
| 3–4           | Low      |
| 5–6           | Average  |
| 7–8           | High     |

### Class Distribution

| Category  | Number of Records | Percentage |
| --------- | ----------------: | ---------: |
| Average   |             1,112 |     81.82% |
| High      |               184 |     13.54% |
| Low       |                63 |      4.63% |
| **Total** |         **1,359** |   **100%** |

The dataset is imbalanced, with the **Average** category containing substantially more observations than the **Low** and **High** categories.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the wine dataset using Pandas.
2. Examined the structure and variables.
3. Created the `quality_category` target variable.
4. Separated the input features from the target.
5. Excluded the original `quality` column from model training.
6. Investigated class imbalance.
7. Tested different techniques for handling the imbalance.
8. Evaluated multiple machine learning models.

---

## Machine Learning Models Evaluated

Several approaches were investigated during model development, including:

* Random Forest with class weighting
* Random Forest with Random Over-Sampling
* Random Forest with SMOTE
* Random Forest with balanced subsampling
* Gradient Boosting
* Logistic Regression with class weighting
* Extra Trees with Random Over-Sampling
* Balanced Random Forest

The models were evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Confusion Matrix
* Cross-validation

---

## Final Model

The final selected model is:

**Balanced Random Forest Classifier**

Configuration:

```text
n_estimators = 300
random_state = 42
n_jobs = -1
```

The Balanced Random Forest was selected because the dataset contains significant class imbalance and the model provided better performance across the minority classes than the original Random Forest approach.

### Cross-Validation Performance

Using 5-fold cross-validation, the Balanced Random Forest achieved:

**Mean Macro F1-score: 0.527**

**Standard Deviation: 0.032**

The model showed substantially better recognition of the minority **Low** class compared with the original Random Forest.

### Holdout Test Performance

On the 20% holdout test set:

| Category | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| Average  |      0.92 |   0.69 |     0.79 |
| High     |      0.38 |   0.86 |     0.52 |
| Low      |      0.24 |   0.38 |     0.29 |

Overall accuracy on the holdout test set was approximately **70%**, while the macro F1-score was approximately **0.54**.

Because the dataset is imbalanced, macro F1-score and class-level recall were considered important when selecting the final model rather than relying on accuracy alone.

---

## System Architecture

The application consists of three major components:

```text
User
  │
  ▼
Streamlit Frontend
  │
  │ HTTP Request
  ▼
FastAPI Backend
  │
  ▼
Balanced Random Forest Model
  │
  ▼
Prediction
```

---

## Project Structure

```text
Wine_Quality_Classification/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── streamlit_app.py
│
├── model/
│   └── wine_quality_model.pkl
│
├── data/
│   └── winequality_categorized.csv
│
├── train_final_model.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Imbalanced-learn

### Model Storage

* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit

---

## Installation

Make sure Python and Anaconda are installed.

Open Anaconda Prompt and navigate to the project directory:

```bat
cd C:\Users\pc\Wine_Quality_Classification
```

Install the required packages:

```bat
pip install -r requirements.txt
```

---

## Running the FastAPI Backend

Navigate to the backend directory:

```bat
cd backend
```

Start the API:

```bat
uvicorn app:app --reload --port 8003
```

The API will be available at:

```text
http://127.0.0.1:8003
```

### API Documentation

Open the following address in a browser:

```text
http://127.0.0.1:8003/docs
```

The Swagger documentation provides access to the available API endpoints.

---

## API Endpoints

### Home Endpoint

```text
GET /
```

This endpoint confirms that the API is running and displays the available prediction classes.

### Prediction Endpoint

```text
POST /predict
```

This endpoint receives the 11 wine chemical properties and returns the predicted wine quality category.

Possible prediction results are:

```text
Average
High
Low
```

---

## Running the Streamlit Frontend

Open another Anaconda Prompt window.

Navigate to the frontend directory:

```bat
cd C:\Users\pc\Wine_Quality_Classification\frontend
```

Start Streamlit:

```bat
streamlit run streamlit_app.py --server.port 8503
```

The application can then be opened in a web browser using the local address provided by Streamlit.

---

## Example Prediction

The user enters values for:

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol

The application sends the values to the FastAPI backend.

The trained Balanced Random Forest model processes the input and returns a quality category such as:

```text
Predicted Quality: Average
```

---

## Model File

The trained model is stored as:

```text
model/wine_quality_model.pkl
```

The model artifact contains:

* Trained Balanced Random Forest model
* Feature names
* Class labels
* Target variable information

---

## Re-training the Model

To retrain the model using the available dataset, navigate to the project root:

```bat
cd C:\Users\pc\Wine_Quality_Classification
```

Run:

```bat
python train_final_model.py
```

The script trains the Balanced Random Forest and saves the resulting model as:

```text
model/wine_quality_model.pkl
```

---

## Limitations

The model has several limitations:

1. The dataset is imbalanced.
2. The minority Low class contains relatively few observations.
3. Classification performance varies across the three categories.
4. The model should not be interpreted as a professional wine-tasting or safety assessment system.
5. Predictions depend on the quality and range of the input data.

---

## Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the dataset.
* Collecting additional wine-related features.
* Performing more extensive hyperparameter optimization.
* Testing XGBoost, LightGBM, or other advanced classifiers.
* Applying advanced imbalance-handling techniques.
* Adding probability/confidence scores to predictions.
* Deploying the API to a cloud platform.
* Deploying the Streamlit application publicly.
* Adding authentication and monitoring for production deployment.

---

## Project Author

**Joseph Mfoniso Etim**

### Institution

**Gworld Soft Solution Limited**

### Department

**Mathematics Department**

### Program

**Machine Learning**

### Supervisor

**Engr. Gospower**

### Academic Year

**2025–2026**

---

## Conclusion

This project demonstrates how machine learning can be applied to classify wine quality based on measurable chemical properties.

A Balanced Random Forest approach was selected after evaluating several classification strategies, particularly because of the imbalance among the quality categories.

The completed system combines machine learning, FastAPI, and Streamlit into an end-to-end application capable of accepting wine characteristics and returning a predicted quality category.

The project therefore demonstrates practical skills in **data preprocessing, class-imbalance handling, machine learning model development, model evaluation, API development, and interactive application development**.
