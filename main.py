import streamlit as st
import matplotlib
matplotlib.use('Agg')

import import_ipynb
import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Spam Detection Dashboard", layout="wide")

st.title("Spam Detection Dashboard")

# ---- Sidebar: Dataset Selection ----
st.sidebar.header("Dataset")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset:",
    ("SMS", "Email", "Both")
)

# ---- Sidebar: Model Selection ----
st.sidebar.header("Model")

# Split models by type for better UX
st.sidebar.markdown("**Deep Learning Models**")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ("TextCNN", "FastText")
)

st.sidebar.markdown("**Traditional ML Models**")
ml_model_choice = st.sidebar.selectbox(
    "Choose an ML Model:",
    ("XGBoost", "Decision Tree", "KNN", "SVM")
)

# Combine selection - actually use the one that was selected
# For simplicity, we'll use model_choice as the primary, but you could add a radio button
use_ml = st.sidebar.radio("Model Type:", ("Deep Learning", "Traditional ML"))

if use_ml == "Traditional ML":
    model_choice = ml_model_choice

# ---- Import DataPreProcessing and load chosen dataset ----
from notebooks import DataPreProcessing as dp

# Reload the dataset based on user choice
X_train, X_test, y_train, y_test, df = dp.load_dataset(dataset_choice)
st.sidebar.info(f"Dataset: **{dataset_choice}** ({len(df)} samples)")

# Display class distribution
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Distribution")
label_counts = df['label'].value_counts()
st.sidebar.write(f"📧 Ham: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
st.sidebar.write(f"⚠️ Spam: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")

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

# ---- Dataset-aware model paths ----
def get_model_key():
    """Generate unique model identifier based on dataset and model type"""
    dataset_key = dataset_choice.lower()
    model_key = model_choice.lower().replace(" ", "_")
    if classic_ml_type:
        model_key = classic_ml_type.lower()
    return f"{dataset_key}_{model_key}"

# ---- Sidebar: Train or Load with dataset awareness ----
if classic_ml_type:
    saved_exists = model_module.has_saved_model(classic_ml_type, dataset_choice=dataset_choice) if hasattr(model_module, 'has_saved_model') else False
else:
    saved_exists = model_module.has_saved_model(dataset_choice=dataset_choice) if hasattr(model_module, 'has_saved_model') else False

st.sidebar.markdown("---")
if saved_exists:
    st.sidebar.success(f"✅ Saved model found for {dataset_choice} dataset!")
    action = st.sidebar.radio("Action:", ("Load saved model", "Retrain from scratch"))
else:
    st.sidebar.warning(f"⚠️ No saved model found for {dataset_choice} dataset. You must train first.")
    action = "Retrain from scratch"

# ---- Execute the chosen action ----
model_obj = None
extra_obj = None
metrics = None  # Store metrics for display

if action == "Load saved model":
    with st.spinner("Loading saved model..."):
        if model_choice == "TextCNN":
            model_obj, extra_obj = model_module.load_model(dataset_choice=dataset_choice)
        elif model_choice == "XGBoost":
            model_obj, extra_obj = model_module.load_model(dataset_choice=dataset_choice)
        elif classic_ml_type:
            model_obj, extra_obj = model_module.load_model(classic_ml_type, dataset_choice=dataset_choice)
        elif model_choice == "FastText":
            model_obj = model_module.load_model(dataset_choice=dataset_choice)
    st.success(f"✅ Model loaded from disk for {dataset_choice} dataset!")

elif action == "Retrain from scratch":
    if st.sidebar.button("🚀 Start Training", type="primary"):
        with st.spinner(f"Training {model_choice} on {dataset_choice} dataset..."):
            if model_choice == "TextCNN":
                model_obj, extra_obj, train_losses, test_losses, train_accs, test_accs = model_module.train(
                    X_train, y_train, X_test, y_test, dataset_choice=dataset_choice
                )
                st.success("✅ Training complete! Model saved automatically.")
                
                # Get predictions for metrics
                y_pred = []
                for text in X_test:
                    label, _ = model_module.predict_message(text, model_obj, extra_obj)
                    y_pred.append(1 if label.upper() == "SPAM" else 0)
                
                # Calculate metrics
                metrics = {
                    'train_loss': train_losses[-1],
                    'test_loss': test_losses[-1],
                    'train_acc': train_accs[-1],
                    'test_acc': test_accs[-1],
                    'f1': f1_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

            elif model_choice == "XGBoost":
                model_obj, extra_obj, train_losses, test_losses, train_acc, test_acc = model_module.train(
                    X_train, y_train, X_test, y_test, dataset_choice=dataset_choice
                )
                st.success("✅ Training complete! Model saved automatically.")
                
                y_pred = model_obj.predict(extra_obj.transform(X_test))
                metrics = {
                    'train_loss': train_losses[-1],
                    'test_loss': test_losses[-1],
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'f1': f1_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

            elif classic_ml_type:
                model_obj, extra_obj, acc = model_module.train(
                    X_train, y_train, X_test, y_test,
                    model_type=classic_ml_type,
                    dataset_choice=dataset_choice
                )
                st.success("✅ Training complete! Model saved automatically.")
                
                y_pred = model_obj.predict(extra_obj.transform(X_test))
                metrics = {
                    'test_acc': acc,
                    'f1': f1_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

            elif model_choice == "FastText":
                model_obj, acc = model_module.train(
                    X_train, y_train, X_test, y_test, dataset_choice=dataset_choice
                )
                st.success("✅ Training complete! Model saved automatically.")
                
                # Get predictions for metrics
                y_pred = []
                for text in X_test:
                    label, _ = model_module.predict_message(text, model_obj)
                    y_pred.append(1 if label.upper() == "SPAM" else 0)
                
                metrics = {
                    'test_acc': acc,
                    'f1': f1_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

# ---- Display Metrics if available ----
if metrics:
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'train_acc' in metrics:
        col1.metric("🎯 Train Accuracy", f"{metrics['train_acc']*100:.2f}%" if metrics['train_acc'] <= 1 else f"{metrics['train_acc']:.2f}%")
    col2.metric("✅ Test Accuracy", f"{metrics['test_acc']*100:.2f}%" if metrics['test_acc'] <= 1 else f"{metrics['test_acc']:.2f}%")
    col3.metric("📈 F1 Score", f"{metrics['f1']*100:.2f}%")
    col4.metric("🔍 Recall (Spam Detection)", f"{metrics['recall']*100:.2f}%")
    
    if 'train_loss' in metrics:
        st.caption(f"Final Train Loss: {metrics['train_loss']:.4f} | Final Test Loss: {metrics['test_loss']:.4f}")
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = metrics['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_choice} on {dataset_choice}')
    st.pyplot(fig)
    plt.close()
    
    # Additional metrics in expandable section
    with st.expander("📋 Detailed Classification Report"):
        tn, fp, fn, tp = cm.ravel()
        st.write(f"""
        - **True Negatives (Ham correctly classified):** {tn}
        - **False Positives (Ham misclassified as Spam):** {fp}
        - **False Negatives (Spam misclassified as Ham):** {fn}
        - **True Positives (Spam correctly classified):** {tp}
        - **Precision:** {tp/(tp+fp)*100:.2f}% (if tp+fp > 0 else 0)
        - **Specificity:** {tn/(tn+fp)*100:.2f}% (if tn+fp > 0 else 0)
        """)

# ---- Prediction Section ----
st.markdown("---")
st.subheader("🔮 Predict a Message")
message = st.text_area(
    "Enter a message to analyze:",
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    height=100
)

if st.button("🔍 Detect Spam", type="primary"):
    if model_obj is None and not saved_exists:
        st.error("❌ Please train the model first!")
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

        # Display result with animation effect
        if label.strip().upper() == "SPAM":
            st.error(f"🚨 **SPAM DETECTED** 🚨\n\nConfidence: {confidence:.2f}%")
            st.warning("⚠️ This message appears to be spam. Do not click on suspicious links!")
        else:
            st.success(f"✅ **SAFE MESSAGE** ✅\n\nConfidence: {confidence:.2f}%")
            st.info("📧 This message appears to be legitimate (HAM).")

# ---- Info about dataset-model separation ----
with st.sidebar.expander("ℹ️ About Dataset-Model Separation"):
    st.write("""
    Models are saved **separately per dataset**:
    - `sms_textcnn.pth` for SMS + TextCNN
    - `email_xgboost.pkl` for Email + XGBoost
    - etc.
    
    This ensures optimal performance since SMS and email spam have different patterns!
    """)