
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler # Ensure this is imported

# --- Start: Fix loading and prediction ---
# Load the trained pipeline instead of just the classifier
try:
    model_pipeline = joblib.load('best_model_pipeline.pkl')
    st.success("Model pipeline loaded successfully!")
    # Get the classes from the loaded pipeline's classifier
    try:
        model_classes = model_pipeline.named_steps['classifier'].classes_
    except AttributeError:
         st.warning("Could not retrieve class names from the loaded model.")
         model_classes = None

except FileNotFoundError:
    st.error("Error: 'best_model_pipeline.pkl' not found. Please upload the model file.")
    st.stop() # Stop execution if model not found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# Title
st.title("Social Vulnerability Classification")

# Get the list of feature names the model was trained on
# In a real deployment, you should ideally save these feature names during training
# and load them here. For this Colab context, we'll assume X was available
# when the script was generated and use its columns.
# A more robust way would be to save X.columns.tolist() during training.
# Assuming feature_names is available from the original Colab session or hardcoded if known
# If not available, you would need a way to get them from the model/pipeline if possible,
# or hardcode them based on your training data.
try:
    # This is a placeholder. In a real app, you should load saved feature names.
    # For this exercise, we'll manually define the expected features based on the
    # original data preprocessing code.
    # Referencing X from the Colab environment won't work in a standalone app.
    # You need to know or load the expected feature names.
    # Based on ipython-input-1-71b33754a05c, the dropped columns were:
    # drop_cols = ['FeatureID', 'Geography', 'Name_Geography',
    #              'NotSociallyVulnerable', 'OBJECTID', 'WeightedAvgQuintile',
    #              'Year', 'the_geom', 'Quintile', 'Shape__Area', 'Shape__Length']
    # The target was 'Quintile_Category'.
    # So, feature_names are all columns in the original df except those in drop_cols and 'Quintile_Category'.
    # Let's manually list the columns based on a likely scenario after dropping.
    # THIS ASSUMES YOU KNOW YOUR FEATURE NAMES.
    # Replace this with loading saved feature names in a real deployment.
    # A safer approach is to save a list of feature names alongside the model.
    # Example: feature_names = ['EP_NOVEH', 'EP_AGE65', 'EP_PCI', ...]
    # For this fix, we'll assume X was available and its columns are the feature names needed.
    # This block will only work if 'X' is still defined when this cell runs.
    # In a standalone Streamlit app, you would need to load this list.
    # Let's simulate loading from a file if feature_names isn't in globals().
    try:
         # Attempt to get from global Colab environment if available
         feature_names = globals().get('X', pd.DataFrame()).columns.tolist()
         if not feature_names:
             # If X wasn't available, try loading from a saved list if you had one
             # Example: feature_names = joblib.load('feature_names.pkl')
             # For this demo, we'll hardcode a subset as a fallback
             st.warning("Could not get feature names from X. Using a hardcoded placeholder list.")
             # Replace with actual feature names from your data preprocessing
             feature_names = ['Condition', 'Condition_TotalPop', 'F_TOTAL', 'RPL_Themes', 'TotalPopulation']
             try:
                 st.write(f"Model expects {len(feature_names)} features.") # Show the count
             except NameError:
                 st.error("Error: Could not determine expected feature names.")
                 st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred getting feature names: {e}")
        st.stop()


except Exception as e:
    st.error(f"An error occurred initializing feature names: {e}")
    st.stop()


# Upload or manual input
option = st.radio("Choose input method:", ('Manual Input', 'CSV Upload'))

input_df = None # Initialize input_df

if option == 'Manual Input':
    st.subheader("Enter Feature Values")
    manual_inputs = {}
    # Use the determined feature_names to create inputs
    cols_per_row = 3 # Arrange inputs in columns
    cols = st.columns(cols_per_row)
    col_idx = 0

    for feature in feature_names:
        with cols[col_idx]:
             # You might need to set appropriate default values and steps based on your data's range
            manual_inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.2f") # Use format for better display
        col_idx = (col_idx + 1) % cols_per_row


    if st.button("Predict"):
        # Create a DataFrame from the manual inputs, ensuring column order
        input_df = pd.DataFrame([manual_inputs], columns=feature_names)
        st.write("Input Data:")
        st.write(input_df)

elif option == 'CSV Upload':
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df_uploaded.head()) # Show head instead of whole df

        # Validate uploaded CSV columns against expected features
        uploaded_cols = df_uploaded.columns.tolist()
        if sorted(uploaded_cols) != sorted(feature_names): # Sort columns for comparison
            st.error(f"Error: Uploaded CSV columns do not match expected model features (ignoring order).")
            st.write(f"Expected: {', '.join(sorted(feature_names))}")
            st.write(f"Uploaded: {', '.join(sorted(uploaded_cols))}")
            st.stop() # Stop if columns don't match

        # Reorder uploaded DataFrame columns to match the model's expected order
        try:
            input_df = df_uploaded[feature_names]
            st.success("CSV columns validated and ordered.")
        except KeyError as e:
             st.error(f"Error reordering columns. Missing column: {e}")
             st.stop()


# If input_df was created (either manually or via upload)
if input_df is not None:
    try:
        # Use the loaded pipeline to make predictions
        # The pipeline will automatically apply the scaler and then the classifier
        prediction = model_pipeline.predict(input_df)
        st.subheader("Prediction")
        st.write(prediction)

        # Display prediction probability if the model supports it and is loaded in pipeline
        try:
            # Try predicting probabilities if the final estimator has 'predict_proba'
            if hasattr(model_pipeline.named_steps['classifier'], 'predict_proba'):
                 prediction_proba = model_pipeline.predict_proba(input_df)
                 st.subheader("Prediction Probability")
                 # Create a DataFrame for better display
                 if model_classes is not None:
                     proba_df = pd.DataFrame(prediction_proba, columns=model_classes)
                 else:
                      proba_df = pd.DataFrame(prediction_proba) # Fallback if classes not found
                 st.write(proba_df)
        except Exception as e:
             st.warning(f"Could not display prediction probabilities: {e}")


    except Exception as e:
         st.error(f"An error occurred during prediction: {e}")


    # --- Start: Fix performance metrics display ---
    # Display performance metrics
    st.subheader("Performance Metrics (Using Test Data)")
    try:
    # Load test data from files
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    st.success("Test data loaded successfully.")

    # Make predictions on the test set
    y_test_pred = model_pipeline.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    # Display accuracy
    st.write(f"**Accuracy on Test Set:** {acc:.2f}")

    # Display confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=model_classes, yticklabels=model_classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(fig)
    plt.close(fig)

except FileNotFoundError:
    st.warning("Test data files (X_test.pkl and y_test.pkl) not found. Please upload them to use this feature.")
except Exception as e:
    st.error(f"An error occurred while calculating performance metrics: {e}")
# --- End: Fix performance metrics display ---
