import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initializing session state
if "models" not in st.session_state:
    st.session_state.models = {}
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None
if "accuracies" not in st.session_state:
    st.session_state.accuracies = {}
if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = {}

# Function to load data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None

# Function to encode categorical columns using LabelEncoder
def encode_categorical_features(df, feature_columns):
    label_encoders = {}
    for feature in feature_columns:
        if df[feature].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()
            df[feature] = label_encoder.fit_transform(df[feature])
            label_encoders[feature] = label_encoder
    return df, label_encoders

# Function to preprocess data
def preprocess_data(df, target_column, feature_columns):
    df, label_encoders = encode_categorical_features(df, feature_columns)  # Encoding categorical features
    X = df[feature_columns]
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, label_encoders

# Streamlit app setup
st.title("Machine Learning Regression App")

# Uploading dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dataset Preview:")
        st.write(df.head())

        # Selecting target and feature columns
        target_column = st.selectbox("Select target column", df.columns)
        feature_columns = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])

        if feature_columns:
            X, y, scaler, label_encoders = preprocess_data(df, target_column, feature_columns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Storing data in session state
            st.session_state.scaler = scaler
            st.session_state.feature_columns = feature_columns
            st.session_state.label_encoders = label_encoders
            st.success("Data preprocessed successfully!")

            # Display label encodings to the user
            if label_encoders:
                st.write("### Label Encodings:")
                for feature, encoder in label_encoders.items():
                    st.write(f"{feature}:")
                    encoding_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    st.write(encoding_dict)

            # Training models with default parameters (no user inputs)
            st.write("### Train Models")
            
            # Linear Regression
            if st.button("Train Linear Regression"):
                model_lr = LinearRegression()
                model_lr.fit(X_train, y_train)
                y_pred = model_lr.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Linear Regression R² Score: {r2:.2f}")
                st.write(f"Linear Regression MAE: {mae:.2f}")
                st.session_state.models["Linear Regression"] = model_lr
                st.session_state.accuracies["Linear Regression"] = {"R²": r2, "MAE": mae}

            # KNN Regressor
            if st.button("Train KNN Regressor"):
                model_knn = KNeighborsRegressor()  # Using default K=5
                model_knn.fit(X_train, y_train)
                y_pred = model_knn.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"KNN Regressor R² Score: {r2:.2f}")
                st.write(f"KNN Regressor MAE: {mae:.2f}")
                st.session_state.models["KNN Regressor"] = model_knn
                st.session_state.accuracies["KNN Regressor"] = {"R²": r2, "MAE": mae}

            # SVM Regressor
            if st.button("Train SVM Regressor"):
                model_svm = SVR()  # Using default 'rbf' kernel
                model_svm.fit(X_train, y_train)
                y_pred = model_svm.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"SVM Regressor R² Score: {r2:.2f}")
                st.write(f"SVM Regressor MAE: {mae:.2f}")
                st.session_state.models["SVM Regressor"] = model_svm
                st.session_state.accuracies["SVM Regressor"] = {"R²": r2, "MAE": mae}

            # Decision Tree Regressor
            if st.button("Train Decision Tree Regressor"):
                model_dt = DecisionTreeRegressor(random_state=42)  # Using default parameters
                model_dt.fit(X_train, y_train)
                y_pred = model_dt.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Decision Tree R² Score: {r2:.2f}")
                st.write(f"Decision Tree MAE: {mae:.2f}")
                st.session_state.models["Decision Tree Regressor"] = model_dt
                st.session_state.accuracies["Decision Tree Regressor"] = {"R²": r2, "MAE": mae}

            # Display all model scores
            if st.session_state.accuracies:
                st.write("### Model Performance Scores")
                for model_name, metrics in st.session_state.accuracies.items():
                    st.write(f"{model_name}:")
                    st.write(f"  R² Score: {metrics['R²']:.2f}")
                    st.write(f"  MAE: {metrics['MAE']:.2f}")

            # Predicting new data
            st.write("### Predict with New Data")
            new_data = {feature: st.number_input(f"Enter value for {feature}", value=0.0) for feature in feature_columns}

            if st.button("Predict"):
                if not st.session_state.models:
                    st.warning("Please train at least one model before predicting.")
                else:
                    # Validating input dimensions
                    if len(new_data) != len(feature_columns):
                        st.error(f"Number of features must match the trained model: {len(feature_columns)}.")
                    else:
                        # Scaling input data
                        new_data_scaled = st.session_state.scaler.transform([list(new_data.values())])

                        # Making predictions
                        for model_name, model in st.session_state.models.items():
                            pred = model.predict(new_data_scaled)
                            st.write(f"{model_name} Prediction: {pred[0]:.2f}")
