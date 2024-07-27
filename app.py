import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Title of the app
st.title("Interactive Machine Learning Model Explorer")

# Sidebar for upload and configuration
st.sidebar.title("Upload and Configure")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Cache the data loading function
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Function to check if the target variable is continuous
def is_continuous(series):
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 20

# Function to save the model to a specific path using pickle
def save_model(model, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        st.success(f"Model successfully saved to {path}")
    except Exception as e:
        st.error(f"Error saving model: {e}")

# If file is uploaded, display dataset and options
if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.write("## Dataset")
        st.write(df.head())

        # Sidebar options for data inspection and preprocessing
        if st.sidebar.checkbox("Show Data Summary"):
            st.write("### Data Summary")
            st.write(df.describe())
            st.write("### Data Types")
            st.write(df.dtypes)
            st.write("### Null Values")
            st.write(df.isnull().sum())

        # Data preprocessing options in the sidebar
        st.sidebar.subheader("Data Preprocessing")

        # Handle missing values
        if st.sidebar.checkbox("Handle Missing Values"):
            fill_value = st.sidebar.selectbox("Fill Missing Values With", ["Mean", "Median", "Mode"])
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if fill_value == "Mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif fill_value == "Median":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif fill_value == "Mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)

        # Encode categorical variables
        if st.sidebar.checkbox("Encode Categorical Variables"):
            label_encoders = {}
            for col in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

        # Option to select dataset
        st.sidebar.subheader("Select Dataset")
        dataset_option = st.sidebar.selectbox("Select the dataset", ["Original Dataset", "Cleaned Dataset"])
        if dataset_option == "Cleaned Dataset":
            st.write("### Cleaned Dataset")
            st.write(df.head())
        else:
            st.write("### Original Dataset")
            st.write(load_data(uploaded_file).head())

        # Data Visualization
        st.sidebar.subheader("Data Visualization")
        if st.sidebar.checkbox("Show Heatmap"):
            st.write("### Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.write(fig)

        if st.sidebar.checkbox("Show Pairplot"):
            st.write("### Pairplot")
            pairplot_fig = sns.pairplot(df)
            st.pyplot(pairplot_fig.fig)

        # Feature Selection
        st.sidebar.subheader("Feature Selection")
        target = st.sidebar.selectbox("Select Target Variable", df.columns)
        features = st.sidebar.multiselect("Select Feature Variables", [col for col in df.columns if col != target])

        if features:
            # Data Splitting
            st.sidebar.subheader("Data Splitting")
            test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
            X = df[features]
            y = df[target]

            # Check if the task is regression or classification
            is_regression = is_continuous(df[target])

            if not is_regression:
                bins = st.sidebar.slider("Number of bins for discretizing target", 2, 10, 3)
                y = pd.cut(y, bins=bins, labels=False)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Model Selection and Training
            st.sidebar.subheader("Model Selection and Training")
            task = st.sidebar.selectbox("Choose Task", ["Regression", "Classification"])

            if task == "Regression":
                model_type = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor", "Decision Tree Regressor", "K-Nearest Neighbors Regressor"])
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif model_type == "Support Vector Regressor":
                    model = SVR()
                elif model_type == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                elif model_type == "K-Nearest Neighbors Regressor":
                    model = KNeighborsRegressor()
            else:
                model_type = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest Classifier", "Support Vector Machine", "Decision Tree Classifier", "K-Nearest Neighbors Classifier"])
                if model_type == "Logistic Regression":
                    model = LogisticRegression()
                elif model_type == "Random Forest Classifier":
                    model = RandomForestClassifier()
                elif model_type == "Support Vector Machine":
                    model = SVC(probability=True)
                elif model_type == "Decision Tree Classifier":
                    model = DecisionTreeClassifier()
                elif model_type == "K-Nearest Neighbors Classifier":
                    model = KNeighborsClassifier()

            if st.sidebar.button("Train Model"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("## Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### r2 score")
                    if is_regression:
                        st.write(r2_score(y_test, y_pred))
                with col2:
                    st.write("### Mean Squared Error")
                    if is_regression:
                        st.write(mean_squared_error(y_test, y_pred))

                if task == "Classification":
                    with st.form("classification_report_form"):
                        st.write("### Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                        st.form_submit_button()
                        st.markdown(''' 
            <style>
            .st-emotion-cache-19rxjzo.ef3psqc7
            { 
                visibility: hidden;           
            }
            </style>
            ''', unsafe_allow_html=True)

                        st.write("### Accuracy")
                        st.write(accuracy_score(y_test, y_pred))

                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    st.write(fig)

                    # ROC Curve for multiclass classification
                    st.write("### ROC Curves")
                    y_bin = label_binarize(y_test, classes=np.unique(y_test))
                    y_prob = model.predict_proba(X_test)
                    
                    if y_prob.shape[1] > 1:  # Only if more than one class
                        fig, ax = plt.subplots()
                        for i in range(y_bin.shape[1]):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:0.2f})')
                        
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic')
                        ax.legend(loc="lower right")
                        st.write(fig)

                    # Decision Boundaries for classifiers (excluding Linear Regression)
                    if len(features) == 2:
                        st.write("### Decision Boundaries")
                        x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
                        y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                             np.arange(y_min, y_max, 0.1))
                        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        fig, ax = plt.subplots()
                        ax.contourf(xx, yy, Z, alpha=0.3)
                        ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, s=20, edgecolor='k')
                        st.write(fig)

                # Heatmap of True vs Predicted
                if task == "Classification":
                    st.write("### Heatmap of True vs Predicted")
                    df_results = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
                    fig, ax = plt.subplots()
                    sns.heatmap(pd.crosstab(df_results['True'], df_results['Predicted']), annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title("Heatmap of True vs Predicted")
                    st.write(fig)

                # # Save model to the specified directory
                # st.sidebar.subheader("Save and Download Model")
                
                # # Specify the save path
                # save_path = r"C:\Users\chara\Desktop\ML_Model_explorer\trained_model.pkl"
                
                # if st.sidebar.button("Save Model"):
                #     save_model(model, save_path)

    except Exception as e:
        st.error(f"Error loading file: {e}")
