import streamlit as st
import matplotlib
matplotlib.use('Agg')

import import_ipynb
import os
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Spam Detection Dashboard", layout="wide")

st.title("📧 Spam Detection Dashboard")
st.markdown("*A Comprehensive System for SMS and Email Spam Classification*")

# ---- Sidebar: Navigation ----
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🎯 Prediction", "📊 Model Comparison", "🔍 Error Analysis", "📈 Dataset Explorer"]
)

# ---- Sidebar: Dataset Selection ----
st.sidebar.markdown("---")
st.sidebar.header("Dataset Configuration")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset:",
    ("SMS", "Email", "Both"),
    help="SMS: Mobile text messages | Email: Email messages | Both: Combined dataset"
)

# ---- Sidebar: Model Selection ----
st.sidebar.header("Model Selection")

use_ml = st.sidebar.radio(
    "Model Type:", 
    ("Deep Learning", "Traditional ML"),
    help="Deep Learning: TextCNN, FastText | Traditional ML: XGBoost, Decision Tree, KNN, SVM"
)

if use_ml == "Traditional ML":
    model_choice = st.sidebar.selectbox(
        "Choose an ML Model:",
        ("XGBoost", "Decision Tree", "KNN", "SVM")
    )
else:
    model_choice = st.sidebar.selectbox(
        "Choose a Deep Learning Model:",
        ("TextCNN", "FastText")
    )

# ---- Import DataPreProcessing and load chosen dataset ----
from notebooks import DataPreProcessing as dp

# Load dataset
X_train, X_test, y_train, y_test, df = dp.load_dataset(dataset_choice)
st.sidebar.success(f"✅ Loaded {dataset_choice} dataset ({len(df)} samples)")

# Display class distribution in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Distribution")
label_counts = df['label'].value_counts()
total = len(df)
st.sidebar.write(f"📧 **Ham (0):** {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")
st.sidebar.write(f"⚠️ **Spam (1):** {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")

# Color-coded warning for imbalanced data
if label_counts.get(1, 0) / total < 0.2:
    st.sidebar.warning("⚠️ Imbalanced dataset! Consider using class weights or SMOTE.")

# ---- Import model module ----
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
    st.error(f"❌ Could not load {model_choice}: `{import_error}`")
    st.info("For FastText, install it: `pip install fasttext`")
    st.stop()

# ---- Dataset-aware model checking ----
def get_model_key():
    dataset_key = dataset_choice.lower()
    model_key = model_choice.lower().replace(" ", "_")
    if classic_ml_type:
        model_key = classic_ml_type.lower()
    return f"{dataset_key}_{model_key}"

# Check if saved model exists
if classic_ml_type:
    saved_exists = hasattr(model_module, 'has_saved_model') and model_module.has_saved_model(classic_ml_type, dataset_choice=dataset_choice)
else:
    saved_exists = hasattr(model_module, 'has_saved_model') and model_module.has_saved_model(dataset_choice=dataset_choice)

st.sidebar.markdown("---")
if saved_exists:
    st.sidebar.success(f"✅ Saved model found for {dataset_choice}")
    action = st.sidebar.radio("Action:", ("Load saved model", "Retrain from scratch"))
else:
    st.sidebar.warning(f"⚠️ No saved model for {dataset_choice}. Train first.")
    action = "Retrain from scratch"

# ---- Train or Load Model ----
model_obj = None
extra_obj = None
metrics_history = None

if action == "Load saved model":
    with st.spinner(f"Loading {model_choice} model for {dataset_choice}..."):
        try:
            if model_choice == "TextCNN":
                model_obj, extra_obj = model_module.load_model(dataset_choice=dataset_choice)
            elif model_choice == "XGBoost":
                model_obj, extra_obj = model_module.load_model(dataset_choice=dataset_choice)
            elif classic_ml_type:
                model_obj, extra_obj = model_module.load_model(classic_ml_type, dataset_choice=dataset_choice)
            elif model_choice == "FastText":
                model_obj = model_module.load_model(dataset_choice=dataset_choice)
            st.success(f"✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("Please retrain the model.")
            action = "Retrain from scratch"

elif action == "Retrain from scratch":
    if st.sidebar.button("🚀 Start Training", type="primary"):
        with st.spinner(f"Training {model_choice} on {dataset_choice} dataset..."):
            try:
                if model_choice == "TextCNN":
                    model_obj, extra_obj, train_losses, test_losses, train_accs, test_accs = model_module.train(
                        X_train, y_train, X_test, y_test, dataset_choice=dataset_choice
                    )
                    # Get predictions for metrics
                    y_pred = []
                    for text in X_test:
                        label, _ = model_module.predict_message(text, model_obj, extra_obj)
                        y_pred.append(1 if label.upper() == "SPAM" else 0)
                    
                    metrics_history = {
                        'train_loss': train_losses[-1],
                        'test_loss': test_losses[-1],
                        'train_acc': train_accs[-1],
                        'test_acc': test_accs[-1],
                        'f1': f1_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }

                elif model_choice == "XGBoost":
                    model_obj, extra_obj, train_losses, test_losses, train_acc, test_acc = model_module.train(
                        X_train, y_train, X_test, y_test, dataset_choice=dataset_choice
                    )
                    y_pred = model_obj.predict(extra_obj.transform(X_test))
                    metrics_history = {
                        'train_loss': train_losses[-1],
                        'test_loss': test_losses[-1],
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'f1': f1_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }

                elif classic_ml_type:
                    model_obj, extra_obj, acc = model_module.train(
                        X_train, y_train, X_test, y_test,
                        model_type=classic_ml_type,
                        dataset_choice=dataset_choice
                    )
                    y_pred = model_obj.predict(extra_obj.transform(X_test))
                    metrics_history = {
                        'test_acc': acc,
                        'f1': f1_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }

                elif model_choice == "FastText":
                    model_obj, acc = model_module.train(
                        X_train, y_train, X_test, y_test, dataset_choice=dataset_choice
                    )
                    y_pred = []
                    for text in X_test:
                        label, _ = model_module.predict_message(text, model_obj)
                        y_pred.append(1 if label.upper() == "SPAM" else 0)
                    
                    metrics_history = {
                        'test_acc': acc,
                        'f1': f1_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }
                
                st.success(f"✅ Training complete! Model saved for {dataset_choice} dataset.")
                
            except Exception as e:
                st.error(f"Training failed: {e}")

# ---- PAGE 1: PREDICTION ----
if page == "🎯 Prediction":
    st.header("🔮 Spam Detection")
    
    # Show current model info
    st.info(f"**Active Model:** {model_choice} | **Dataset:** {dataset_choice}")
    
    # Display metrics if available
    if metrics_history:
        st.markdown("---")
        st.subheader("📊 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        if 'test_acc' in metrics_history:
            test_acc = metrics_history['test_acc']
            test_acc_pct = test_acc * 100 if test_acc <= 1 else test_acc
            col1.metric("✅ Accuracy", f"{test_acc_pct:.2f}%")
        col2.metric("📈 F1-Score", f"{metrics_history['f1']*100:.2f}%")
        col3.metric("🎯 Precision", f"{metrics_history['precision']*100:.2f}%")
        col4.metric("🔍 Recall", f"{metrics_history['recall']*100:.2f}%")
        
        # Confusion Matrix
        with st.expander("📊 View Confusion Matrix"):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(metrics_history['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            plt.close()
    
    # Prediction input
    st.markdown("---")
    st.subheader("✏️ Test Your Message")
    
    example_messages = {
        "Spam Example": "CONGRATULATIONS! You've won $1,000,000. Click here to claim your prize now!",
        "Ham Example": "Hey, are we still meeting for lunch at 2pm? Let me know.",
        "Custom": ""
    }
    
    example_choice = st.selectbox("Try an example:", list(example_messages.keys()))
    if example_choice != "Custom":
        message = st.text_area("Message:", example_messages[example_choice], height=100)
    else:
        message = st.text_area("Enter your message:", height=100)
    
    if st.button("🔍 Detect Spam", type="primary"):
        if model_obj is None and not saved_exists:
            st.error("❌ Please train or load a model first!")
        else:
            with st.spinner("Analyzing message..."):
                try:
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
                    
                    # Display result
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if label.strip().upper() == "SPAM":
                            st.error(f"🚨 **SPAM DETECTED** 🚨\n\nConfidence: {confidence:.2f}%")
                            st.warning("⚠️ This appears to be spam. Do not click suspicious links!")
                        else:
                            st.success(f"✅ **SAFE MESSAGE** ✅\n\nConfidence: {confidence:.2f}%")
                            st.info("📧 This appears to be legitimate (HAM).")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ---- PAGE 2: MODEL COMPARISON ----
elif page == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison")
    st.markdown("*Compare all models on the current dataset*")
    
    if st.button("🔄 Run Comparison on All Models"):
        with st.spinner("Evaluating all models (this may take a few minutes)..."):
            results = []
            models_to_test = ["TextCNN", "XGBoost", "Decision Tree", "SVM", "KNN", "FastText"]
            
            progress_bar = st.progress(0)
            for idx, model_name in enumerate(models_to_test):
                try:
                    # Load or evaluate each model
                    if model_name == "TextCNN":
                        from notebooks import TextCNN as mod
                        if mod.has_saved_model(dataset_choice=dataset_choice):
                            m, v = mod.load_model(dataset_choice=dataset_choice)
                            y_pred = []
                            for text in X_test:
                                label, _ = mod.predict_message(text, m, v)
                                y_pred.append(1 if label.upper() == "SPAM" else 0)
                            # Calculate metrics
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            prec = precision_score(y_test, y_pred, zero_division=0)
                            rec = recall_score(y_test, y_pred, zero_division=0)
                        else:
                            acc = f1 = prec = rec = None  # Mark as not available
                    
                    elif model_name == "XGBoost":
                        from notebooks import XGBoost as mod
                        if mod.has_saved_model(dataset_choice=dataset_choice):
                            m, v = mod.load_model(dataset_choice=dataset_choice)
                            y_pred = m.predict(v.transform(X_test))
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            prec = precision_score(y_test, y_pred, zero_division=0)
                            rec = recall_score(y_test, y_pred, zero_division=0)
                        else:
                            acc = f1 = prec = rec = None
                    
                    elif model_name in ["Decision Tree", "SVM", "KNN"]:
                        from notebooks import Classic_ML as mod
    
                        if model_name == "Decision Tree":
                            ml_type = "DecisionTree"
                        elif model_name == "SVM":
                            ml_type = "SVM"
                        elif model_name == "KNN":
                            ml_type = "KNN"
                        
                        if mod.has_saved_model(ml_type, dataset_choice=dataset_choice):
                            m, v = mod.load_model(ml_type, dataset_choice=dataset_choice)
                            y_pred = m.predict(v.transform(X_test))
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            prec = precision_score(y_test, y_pred, zero_division=0)
                            rec = recall_score(y_test, y_pred, zero_division=0)
                        else:
                            acc = f1 = prec = rec = None
                    
                    elif model_name == "FastText":
                        from notebooks import FastText as mod
                        if mod.has_saved_model(dataset_choice=dataset_choice):
                            m = mod.load_model(dataset_choice=dataset_choice)
                            y_pred = []
                            for text in X_test:
                                label, _ = mod.predict_message(text, m)
                                y_pred.append(1 if label.upper() == "SPAM" else 0)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            prec = precision_score(y_test, y_pred, zero_division=0)
                            rec = recall_score(y_test, y_pred, zero_division=0)
                        else:
                            acc = f1 = prec = rec = None
                    
                    # Add to results (handle None values)
                    results.append({
                        "Model": model_name,
                        "Accuracy (%)": f"{acc*100:.2f}" if acc is not None else "Not Trained",
                        "F1-Score (%)": f"{f1*100:.2f}" if f1 is not None else "Not Trained",
                        "Precision (%)": f"{prec*100:.2f}" if prec is not None else "Not Trained",
                        "Recall (%)": f"{rec*100:.2f}" if rec is not None else "Not Trained",
                        "_acc_num": acc if acc is not None else -1  # Hidden column for sorting
                    })
                    
                except Exception as e:
                    results.append({
                        "Model": model_name,
                        "Accuracy (%)": "Error",
                        "F1-Score (%)": "Error",
                        "Precision (%)": "Error",
                        "Recall (%)": "Error",
                        "_acc_num": -1
                    })
                
                progress_bar.progress((idx + 1) / len(models_to_test))
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Display results
            st.subheader("📈 Comparison Results")
            
            # Show only the display columns (exclude _acc_num)
            display_df = results_df.drop(columns=['_acc_num'])
            st.dataframe(display_df, use_container_width=True)
            
            # Find best model (only among those with valid accuracy)
            valid_models = results_df[results_df['_acc_num'] > 0]
            if len(valid_models) > 0:
                best_model_row = valid_models.loc[valid_models['_acc_num'].idxmax()]
                best_model = best_model_row['Model']
                best_acc = best_model_row['Accuracy (%)']
                st.success(f"🏆 **Best Model:** {best_model} with {best_acc} accuracy")
            else:
                st.warning("⚠️ No trained models found. Please train at least one model first.")
            
            # Bar chart (only for models with numeric values)
            chart_data = []
            for _, row in results_df.iterrows():
                if row['_acc_num'] > 0:  # Only include trained models
                    chart_data.append({
                        "Model": row['Model'],
                        "Accuracy (%)": float(row['Accuracy (%)']),
                        "F1-Score (%)": float(row['F1-Score (%)']),
                        "Precision (%)": float(row['Precision (%)']),
                        "Recall (%)": float(row['Recall (%)'])
                    })
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                fig, ax = plt.subplots(figsize=(12, 6))
                chart_df_melted = chart_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
                sns.barplot(data=chart_df_melted, x="Model", y="Score", hue="Metric", ax=ax)
                ax.set_title(f"Model Performance Comparison on {dataset_choice} Dataset")
                ax.set_ylabel("Score (%)")
                ax.set_ylim(0, 100)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()
            else:
                st.info("📊 Train some models first to see the comparison chart.")

# ---- PAGE 3: ERROR ANALYSIS ----
elif page == "🔍 Error Analysis":
    st.header("🔍 Error Analysis")
    st.markdown("*Understanding where and why the model makes mistakes*")
    
    if model_obj is None and not saved_exists:
        st.warning("⚠️ Please train or load a model first to perform error analysis.")
    else:
        # Get predictions
        with st.spinner("Analyzing predictions..."):
            y_pred = []
            for text in X_test:
                try:
                    if model_choice == "TextCNN":
                        label, _ = model_module.predict_message(text, model_obj, extra_obj)
                    elif model_choice == "XGBoost":
                        label, _ = model_module.predict_message(text, model_obj, extra_obj)
                    elif classic_ml_type:
                        label, _ = model_module.predict_message(text, model_obj, extra_obj, model_type=classic_ml_type)
                    elif model_choice == "FastText":
                        label, _ = model_module.predict_message(text, model_obj)
                    y_pred.append(1 if label.upper() == "SPAM" else 0)
                except:
                    y_pred.append(0)
        
        # Find misclassified examples
        misclassified = []
        for i, (text, true_label) in enumerate(zip(X_test, y_test)):
            if y_pred[i] != true_label:
                misclassified.append({
                    'text': text,
                    'true': 'Spam' if true_label == 1 else 'Ham',
                    'pred': 'Spam' if y_pred[i] == 1 else 'Ham'
                })
        
        # Error statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Test Samples", len(X_test))
        col2.metric("Misclassified", len(misclassified))
        col3.metric("Error Rate", f"{len(misclassified)/len(X_test)*100:.2f}%")
        
        # Confusion Matrix breakdown
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        st.subheader("📊 Error Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**False Positives (Ham → Spam)**")
            st.write(f"Count: {fp}")
            st.write("*Legitimate messages wrongly flagged as spam*")
        with col2:
            st.write("**False Negatives (Spam → Ham)**")
            st.write(f"Count: {fn}")
            st.write("*Spam messages that slipped through*")
        
        # Show examples
        st.subheader("📝 Misclassified Examples")
        
        if len(misclassified) > 0:
            fp_examples = [m for m in misclassified if m['true'] == 'Ham' and m['pred'] == 'Spam']
            fn_examples = [m for m in misclassified if m['true'] == 'Spam' and m['pred'] == 'Ham']
            
            tab1, tab2 = st.tabs(["False Positives (Ham→Spam)", "False Negatives (Spam→Ham)"])
            
            with tab1:
                if fp_examples:
                    for ex in fp_examples[:10]:
                        st.warning(f"**True: Ham | Predicted: Spam**")
                        st.write(f"📝 {ex['text'][:200]}...")
                        st.markdown("---")
                else:
                    st.success("No false positives found!")
            
            with tab2:
                if fn_examples:
                    for ex in fn_examples[:10]:
                        st.error(f"**True: Spam | Predicted: Ham**")
                        st.write(f"📝 {ex['text'][:200]}...")
                        st.markdown("---")
                else:
                    st.success("No false negatives found!")
        else:
            st.success("🎉 Perfect classification! No errors found on test set.")

# ---- PAGE 4: DATASET EXPLORER ----
elif page == "📈 Dataset Explorer":
    st.header("📈 Dataset Explorer")
    st.markdown("*Explore and visualize your data*")
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Ham Messages", label_counts.get(0, 0))
    col3.metric("Spam Messages", label_counts.get(1, 0))
    
    # Class distribution chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    sns.countplot(data=df, x='label', ax=axes[0])
    axes[0].set_title('Class Distribution')
    axes[0].set_xlabel('Label (0=Ham, 1=Spam)')
    axes[0].set_ylabel('Count')
    
    # Pie chart
    df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title('Class Proportions')
    axes[1].set_ylabel('')
    
    st.pyplot(fig)
    plt.close()
    
    # Message length analysis
    st.subheader("📏 Message Length Analysis")
    df['message_length'] = df['message'].str.len()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in [0, 1]:
        subset = df[df['label'] == label]
        ax.hist(subset['message_length'], bins=50, alpha=0.7, label=f"{'Ham' if label==0 else 'Spam'}")
    ax.set_xlabel('Message Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Message Length Distribution by Class')
    ax.legend()
    st.pyplot(fig)
    plt.close()
    
    # Sample messages
    st.subheader("📝 Sample Messages")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ham Examples**")
        ham_samples = df[df['label'] == 0]['message'].head(5)
        for i, msg in enumerate(ham_samples):
            st.write(f"{i+1}. {msg[:100]}..." if len(msg) > 100 else f"{i+1}. {msg}")
            st.markdown("---")
    
    with col2:
        st.write("**Spam Examples**")
        spam_samples = df[df['label'] == 1]['message'].head(5)
        for i, msg in enumerate(spam_samples):
            st.write(f"{i+1}. {msg[:100]}..." if len(msg) > 100 else f"{i+1}. {msg}")
            st.markdown("---")

# ---- Footer ----
st.markdown("---")
st.markdown("*Spam Detection System")