AQI_PROJECT — Air Quality Index Prediction

Overview
This project predicts the **Air Quality Index (AQI) category** of a location based on key pollutant concentrations using Machine Learning. It includes data preprocessing, model training with multiple classifiers, model comparison, and a **Streamlit web application** for real-time AQI prediction.

---

 Objectives
- Load and preprocess real-world AQI data from Indian cities
- Train and compare multiple ML classification models
- Select the best-performing model automatically
- Save the trained model, scaler, and label encoder
- Deploy an interactive web app for AQI prediction

---

 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Pandas & NumPy** | Data manipulation and analysis |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **Matplotlib** | Model comparison visualization |
| **Streamlit** | Web application for prediction |
| **Pickle** | Model serialization |

---
 Project Structure

```
AQI_PROJECT/
│
├── data/
│   └── city_day.csv           # Raw AQI dataset (29,531 records, 16 features)
│
├── models/
│   ├── model.pkl              # Best trained ML model
│   ├── scaler.pkl             # StandardScaler for input normalization
│   └── label_encoder.pkl      # LabelEncoder for AQI category decoding
│
├── notebooks/
│   └── AQI_Analysis.ipynb     # Jupyter Notebook (EDA + Model Training)
│
├── app.py                     # Streamlit web application
└── README.md                  # Project documentation
```

---

Dataset

- **File:** `city_day.csv`
- **Records:** 29,531 rows across multiple Indian cities (2015 onwards)
- **Features used for training:**

| Feature | Description |
|---------|-------------|
| PM2.5 | Fine particulate matter |
| PM10 | Coarse particulate matter |
| NO2 | Nitrogen Dioxide |
| SO2 | Sulphur Dioxide |
| CO | Carbon Monoxide |
| O3 | Ozone |
| **AQI_Bucket** | Target variable (AQI Category) |

---

ML Models Compared

The following classifiers were trained and compared:

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Tree-based classifier |
| **Random Forest** | Ensemble of 200 decision trees |
| KNN | K-Nearest Neighbors |
| SVM | Support Vector Machine |

The **best model** is automatically selected based on test accuracy and saved as `model.pkl`.

---
 AQI Categories (Target Classes)

| AQI Category | Health Impact |
|-------------|--------------|
| Good | Minimal impact |
| Satisfactory | Minor breathing discomfort |
| Moderate | Discomfort to sensitive groups |
| Poor | Breathing discomfort to most |
| Very Poor | Respiratory illness risk |
| Severe | Serious health hazard |

---

 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Sachin-307/AQI_PROJECT.git
cd AQI_PROJECT
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

### 3. Run the Jupyter Notebook (Model Training)
```bash
jupyter notebook notebooks/AQI_Analysis.ipynb
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

---

##  Web Application

The **Streamlit app** (`app.py`) allows users to:
- Input pollutant values (PM2.5, PM10, NO2, SO2, CO, O3)
- Click **Predict AQI** to get the AQI category instantly
- View results in a clean, user-friendly interface

---

##  Collaborators

| Name | GitHub Profile |
|------|---------------|
| Collaborator 1 | [rohit-585](https://github.com/rohit-585) |
| Collaborator 2 | [amansaxena708](https://github.com/amansaxena708) |
| Collaborator 3 | [amansingh249](https://github.com/amansingh249) |

---
## Deploy link

  https://aqiproject-3r2sypkf7vwc8i7ycauhfn.streamlit.app/
## License
This project is licensed under the **MIT License**.

---

> *"Breathe clean. Live healthy. "*
