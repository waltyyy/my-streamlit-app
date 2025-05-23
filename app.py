import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Load trained model pipeline ---
try:
    model_pipeline = joblib.load('best_model_pipeline.pkl')
    st.success("Model pipeline loaded successfully!")
    try:
        model_classes = model_pipeline.named_steps['classifier'].classes_
    except AttributeError:
        st.warning("Could not retrieve class names from the classifier.")
        model_classes = None
except FileNotFoundError:
    st.error("Error: 'best_model_pipeline.pkl' not found. Please upload the model file.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- Define expected feature names ---
try:
    feature_names = ['Condition', 'Condition_TotalPop', 'F_TOTAL', 'RPL_Themes', 'TotalPopulation']
    st.info(f"Model expects {len(feature_names)} features.")
except Exception as e:
    st.error(f"Error initializing feature names: {e}")
    st.stop()

# --- Title and Input Method ---
st.title("Social Vulnerability Classification")
option = st.radio("Choose input method:", ('Manual Input', 'CSV Upload'))

input_df = None  # Initialize input_df

# --- Manual Input ---
if option == 'Manual Input':
    st.subheader("Enter Feature Values")
    manual_inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            manual_inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.2f")

    if st.button("Predict"):
        input_df = pd.DataFrame([manual_inputs], columns=feature_names)
        st.write("Input Data:")
        st.write(input_df)

# --- CSV Upload ---
elif option == 'CSV Upload':
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df_uploaded.head())

        # Check for missing required features
        missing_cols = [col for col in feature_names if col not in df_uploaded.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Keep only required features, ignore extras
        input_df = df_uploaded[feature_names]
        st.success("CSV validated. Using only required features.")
        st.write("Filtered Input Data:")
        st.write(input_df)

# --- Make Prediction ---
if input_df is not None:
    try:
        prediction = model_pipeline.predict(input_df)
        st.subheader("Prediction")
        st.write(prediction)

        # Show prediction probabilities if available
        if hasattr(model_pipeline.named_steps['classifier'], 'predict_proba'):
            prediction_proba = model_pipeline.predict_proba(input_df)
            if model_classes is not None:
                proba_df = pd.DataFrame(prediction_proba, columns=model_classes)
            else:
                proba_df = pd.DataFrame(prediction_proba)
            st.subheader("Prediction Probability")
            st.write(proba_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

    # --- Show Performance Metrics ---
    st.subheader("Performance Metrics (Using Test Data)")
    try:
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")
        y_test_pred = model_pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)

        st.write(f"**Accuracy on Test Set:** {acc:.2f}")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=model_classes, yticklabels=model_classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
        plt.close(fig)

    except FileNotFoundError:
        st.warning("Test data files (X_test.pkl and y_test.pkl) not found.")
    except Exception as e:
        st.error(f"An error occurred while calculating performance metrics: {e}")
