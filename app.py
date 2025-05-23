import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Load trained model pipeline ---
try:
    model_pipeline = joblib.load('best_model.pkl')
    try:
        model_classes = model_pipeline.named_steps['classifier'].classes_
    except AttributeError:
        st.warning("Could not retrieve class names from the classifier.")
        model_classes = None
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Please upload the model file.")
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

# --- CSV Upload ---
elif option == 'CSV Upload':
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df_uploaded.head())

        missing_cols = [col for col in feature_names if col not in df_uploaded.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        input_df = df_uploaded.copy()
        st.success("CSV validated. Using required features for prediction.")

# --- Show input summary and visualize ---
if input_df is not None:
    st.subheader("📊 Input Data Used for Prediction")
    st.write(input_df[feature_names])

    # Visualization
    st.subheader("📈 Input Data Visualization")
    try:
        if input_df.shape[0] > 1:
            st.markdown("Showing distribution of features (for CSV Upload):")
            sns.set(style="whitegrid")
            fig = sns.pairplot(input_df[feature_names])
            st.pyplot(fig)
            plt.close()
        else:
            st.markdown("Bar chart of individual input (for Manual Input):")
            fig, ax = plt.subplots()
            input_df[feature_names].T.plot(kind='bar', legend=False, ax=ax)
            plt.xticks(rotation=45)
            plt.ylabel("Value")
            st.pyplot(fig)
            plt.close()
    except Exception as e:
        st.warning(f"Could not generate visualization: {e}")

    # --- Make Prediction ---
    try:
        prediction = model_pipeline.predict(input_df[feature_names])
        st.subheader("🧠 Prediction")
        st.write(prediction)

        # Show prediction probabilities if available
        if hasattr(model_pipeline.named_steps['classifier'], 'predict_proba'):
            prediction_proba = model_pipeline.predict_proba(input_df[feature_names])
            if model_classes is not None:
                proba_df = pd.DataFrame(prediction_proba, columns=model_classes)
            else:
                proba_df = pd.DataFrame(prediction_proba)
            st.subheader("📊 Prediction Probability")
            st.write(proba_df)

        # --- Evaluate if true labels exist ---
        if 'Quintile_Category' in input_df.columns:
            st.subheader("📈 Model Performance on Input Data")
            true_labels = input_df['Quintile_Category']
            predicted_labels = prediction

            acc = accuracy_score(true_labels, predicted_labels)
            st.write(f"**Accuracy on Input Data:** {acc:.2f}")

            cm = confusion_matrix(true_labels, predicted_labels)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=model_classes, yticklabels=model_classes)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No true labels (`Quintile_Category`) found in the input. Only predictions shown.")

    except Exception as e:
        st.error(f"An error occurred during prediction or evaluation: {e}")
