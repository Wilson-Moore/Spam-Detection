import streamlit as st
import matplotlib
matplotlib.use('Agg')

import import_ipynb
import os
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Spam Detection Dashboard")

st.title("Spam Detection Dashboard")

# ---- Sidebar: Dataset Selection ----
st.sidebar.header("Dataset")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset:",
    ("SMS", "Email", "Both")
)

# ---- Sidebar: Model Selection ----
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ("TextCNN", "XGBoost", "Decision Tree", "KNN", "SVM", "FastText")
)

# ---- Import DataPreProcessing and load chosen dataset ----
from notebooks import DataPreProcessing as dp

# Reload the dataset based on user choice
X_train, X_test, y_train, y_test, df = dp.load_dataset(dataset_choice)
st.sidebar.info(f"Dataset: **{dataset_choice}** ({len(df)} samples)")

# ---- Import ONLY the chosen model notebook (lazy) ----
model_module = None
import_error = None
classic_ml_type = None

if model_choice == "TextCNN":
    from notebooks import TextCNN as model_module
elif model_choice == "XGBoost":
    from notebooks import XGBoost as model_module
elif model_choice in ("Decision Tree", "KNN", "SVM"):
    from notebooks import Classic_ML as model_module
    classic_ml_type = {"Decision Tree": "DecisionTree", "KNN": "KNN", "SVM": "SVM"}[model_choice]
elif model_choice == "FastText":
    try:
        from notebooks import FastText as model_module
    except Exception as e:
        import_error = str(e)

if import_error:
    st.error(f"Could not load {model_choice}: `{import_error}`")
    st.info("For FastText, you need to install it first: `pip install fasttext`")
    st.stop()

# ---- Sidebar: Train or Load ----
if classic_ml_type:
    saved_exists = model_module.has_saved_model(classic_ml_type)
else:
    saved_exists = model_module.has_saved_model()

st.sidebar.markdown("---")
if saved_exists:
    st.sidebar.success("Saved model found!")
    action = st.sidebar.radio("Action:", ("Load saved model", "Retrain from scratch"))
else:
    st.sidebar.warning("No saved model found. You must train first.")
    action = "Retrain from scratch"

# ---- Execute the chosen action ----
model_obj = None
extra_obj = None

if action == "Load saved model":
    with st.spinner("Loading saved model..."):
        if model_choice == "TextCNN":
            model_obj, extra_obj = model_module.load_model()
        elif model_choice == "XGBoost":
            model_obj, extra_obj = model_module.load_model()
        elif classic_ml_type:
            model_obj, extra_obj = model_module.load_model(classic_ml_type)
        elif model_choice == "FastText":
            model_obj = model_module.load_model()
    st.success("Model loaded from disk!")

elif action == "Retrain from scratch":
    if st.sidebar.button("Start Training"):
        with st.spinner(f"Training {model_choice} on {dataset_choice} dataset..."):
            if model_choice == "TextCNN":
                model_obj, extra_obj, train_losses, test_losses, train_accs, test_accs = model_module.train(
                    X_train, y_train, X_test, y_test
                )
                st.success("Training complete! Model saved automatically.")
                st.subheader("Training Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Train Loss", f"{train_losses[-1]:.2f}")
                col2.metric("Final Train Accuracy", f"{train_accs[-1]:.2f}%")
                col3.metric("Final Test Loss", f"{test_losses[-1]:.2f}")
                col4.metric("Final Test Accuracy", f"{test_accs[-1]:.2f}%")

            elif model_choice == "XGBoost":
                model_obj, extra_obj, train_losses, test_losses, train_acc, test_acc = model_module.train(
                    X_train, y_train, X_test, y_test
                )
                st.success("Training complete! Model saved automatically.")
                st.subheader("Training Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Train Loss", f"{train_losses[-1]:.2f}")
                col2.metric("Train Accuracy", f"{train_acc*100:.2f}%")
                col3.metric("Test Loss", f"{test_losses[-1]:.2f}")
                col4.metric("Test Accuracy", f"{test_acc*100:.2f}%")

            elif classic_ml_type:
                model_obj, extra_obj, acc = model_module.train(
                    X_train, y_train, X_test, y_test,
                    model_type=classic_ml_type
                )
                st.success("Training complete! Model saved automatically.")
                st.subheader("Training Results")
                st.metric("Test Accuracy", f"{acc*100:.2f}%")

            elif model_choice == "FastText":
                model_obj, acc = model_module.train(
                    X_train, y_train, X_test, y_test
                )
                st.success("Training complete! Model saved automatically.")
                st.subheader("Training Results")
                st.metric("Test Accuracy", f"{acc*100:.2f}%")

# ---- Prediction ----
st.markdown("---")
st.subheader("Predict a Message")
message = st.text_area(
    "Enter a message to analyze:",
    "Congratulations! You've won a free iPhone. Click here to claim now!"
)

if st.button("Detect Spam"):
    if model_obj is None and not saved_exists:
        st.error("Please train the model first!")
    else:
        with st.spinner("Analyzing..."):
            if model_choice == "TextCNN":
                label, confidence = model_module.predict_message(message, model_obj, extra_obj)
            elif model_choice == "XGBoost":
                label, confidence = model_module.predict_message(message, model_obj, extra_obj)
            elif classic_ml_type:
                label, confidence = model_module.predict_message(
                    message, model_obj, extra_obj, model_type=classic_ml_type
                )
            elif model_choice == "FastText":
                label, confidence = model_module.predict_message(message, model_obj)

        if label.strip().upper() == "SPAM":
            st.error(f"**SPAM** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"**HAM** (Confidence: {confidence:.2f}%)")