import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Renewable Energy Adoption – Decision Tree",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Renewable Energy Adoption – Decision Tree Classifier")
st.markdown("Upload your dataset, explore the data, train a Decision Tree, and tune hyperparameters — all in one place.")

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 1, 20, 4)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameter Tuning")
run_grid_search = st.sidebar.checkbox("Run GridSearchCV", value=False)
cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5, disabled=not run_grid_search)

# ─── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is None:
    st.info("👈 Upload `Renewable_Energy_Adoption.csv` from the sidebar to get started.")
    st.stop()

data = load_data(uploaded_file)

# Detect target column
if "adoption" not in data.columns:
    st.error("Expected a column named `adoption` as the target. Please check your CSV.")
    st.stop()

feature_cols = [c for c in data.columns if c != "adoption"]
target_col = "adoption"

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🌳 Model Training", "🔧 Hyperparameter Tuning"])

# ══════════════════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", data.shape[0])
    col2.metric("Features", len(feature_cols))
    col3.metric("Target classes", data[target_col].nunique())

    st.subheader("Missing Values")
    missing = data.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(data) * 100).round(2)
    st.dataframe(missing, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(data.describe(), use_container_width=True)

    st.subheader("Target Distribution")
    vc = data[target_col].value_counts().reset_index()
    vc.columns = ["Class", "Count"]
    st.bar_chart(vc.set_index("Class"))

# ══════════════════════════════════════════════════════════
# TAB 2 – VISUALIZATIONS
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Feature Distributions")

    num_cols = data[feature_cols].select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        selected_feat = st.selectbox("Select feature to visualize", num_cols)
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.histplot(data[selected_feat], kde=True, ax=ax, color="#2196F3", bins=30)
        ax.set_title(f"Distribution of {selected_feat}")
        ax.set_xlabel(selected_feat)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Heatmap")
    corr_cols = data[num_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_cols, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Pairplot (sampled)")
    sample_size = min(500, len(data))
    pair_data = data[num_cols + [target_col]].sample(sample_size, random_state=42)
    fig3 = sns.pairplot(pair_data, hue=target_col, diag_kind="kde", plot_kws={"alpha": 0.5})
    st.pyplot(fig3)
    plt.close()

# ══════════════════════════════════════════════════════════
# TAB 3 – MODEL TRAINING
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Train Decision Tree Classifier")

    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_cols])
    X = pd.DataFrame(X_scaled, columns=feature_cols)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=int(random_state)
    )

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=int(random_state))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=["Non-Adoption", "Adoption"], output_dict=True)

        st.success(f"✅ Training complete! Accuracy: **{acc:.4f}**")

        # ── Metrics ──────────────────────────────────────────
        st.subheader("Model Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision (Adoption)", f"{report['Adoption']['precision']:.4f}")
        col3.metric("Recall (Adoption)", f"{report['Adoption']['recall']:.4f}")

        # ── Confusion Matrix ──────────────────────────────────
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Adoption", "Adoption"],
                    yticklabels=["Non-Adoption", "Adoption"], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # ── Classification Report ─────────────────────────────
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        # ── Decision Tree Visualization ───────────────────────
        st.subheader("Decision Tree Visualization")
        fig_tree, ax_tree = plt.subplots(figsize=(18, 8))
        plot_tree(model, feature_names=feature_cols,
                  class_names=["Non-Adoption", "Adoption"],
                  filled=True, rounded=True, ax=ax_tree, fontsize=8)
        ax_tree.set_title("Decision Tree")
        st.pyplot(fig_tree)
        plt.close(fig_tree)

        # ── Feature Importance ────────────────────────────────
        st.subheader("Feature Importance")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig_fi, ax_fi = plt.subplots(figsize=(7, 3))
        importances.plot(kind="bar", ax=ax_fi, color="#4CAF50")
        ax_fi.set_title("Feature Importances")
        ax_fi.set_ylabel("Importance")
        st.pyplot(fig_fi)
        plt.close(fig_fi)

        # ── Download Model ────────────────────────────────────
        st.subheader("Save Model")
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)
        st.download_button(
            label="⬇️ Download Trained Model (.pkl)",
            data=model_buffer,
            file_name="decision_tree_model.pkl",
            mime="application/octet-stream",
        )

# ══════════════════════════════════════════════════════════
# TAB 4 – HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Hyperparameter Tuning with GridSearchCV")

    if not run_grid_search:
        st.info("Enable **Run GridSearchCV** in the sidebar to activate this tab.")
    else:
        param_grid = {
            "max_depth": [2, 3, 4, 5, 10, None],
            "min_samples_split": [2, 3, 5, 10],
            "min_samples_leaf": [1, 2, 3, 5],
            "criterion": ["gini", "entropy"],
        }

        st.json({k: str(v) for k, v in param_grid.items()})

        if st.button("🔍 Run Grid Search", type="primary"):
            scaler2 = StandardScaler()
            X_scaled2 = scaler2.fit_transform(data[feature_cols])
            X2 = pd.DataFrame(X_scaled2, columns=feature_cols)
            y2 = data[target_col]
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X2, y2, test_size=test_size / 100, random_state=int(random_state)
            )

            with st.spinner(f"Running GridSearchCV with {cv_folds}-fold CV... This may take a moment."):
                dtree = DecisionTreeClassifier(random_state=int(random_state))
                grid_search = GridSearchCV(
                    estimator=dtree, param_grid=param_grid,
                    cv=cv_folds, scoring="accuracy", n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train2, y_train2)

            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            y_pred_best = best_model.predict(X_test2)
            acc_best = accuracy_score(y_test2, y_pred_best)
            cm_best = confusion_matrix(y_test2, y_pred_best)
            report_best = classification_report(y_test2, y_pred_best, target_names=["Non-Adoption", "Adoption"], output_dict=True)

            st.success(f"✅ Best accuracy: **{acc_best:.4f}**")

            st.subheader("Best Hyperparameters")
            st.json(best_params)

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc_best:.4f}")
            col2.metric("Precision (Adoption)", f"{report_best['Adoption']['precision']:.4f}")
            col3.metric("Recall (Adoption)", f"{report_best['Adoption']['recall']:.4f}")

            st.subheader("Confusion Matrix (Best Model)")
            fig_cm2, ax_cm2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Adoption", "Adoption"],
                        yticklabels=["Non-Adoption", "Adoption"], ax=ax_cm2)
            ax_cm2.set_xlabel("Predicted")
            ax_cm2.set_ylabel("Actual")
            ax_cm2.set_title("Confusion Matrix (Best Model)")
            st.pyplot(fig_cm2)
            plt.close(fig_cm2)

            st.subheader("Classification Report (Best Model)")
            report_df2 = pd.DataFrame(report_best).transpose()
            st.dataframe(report_df2.style.format("{:.4f}"), use_container_width=True)

            st.subheader("Best Decision Tree Visualization")
            fig_tree2, ax_tree2 = plt.subplots(figsize=(18, 8))
            plot_tree(best_model, feature_names=feature_cols,
                      class_names=["Non-Adoption", "Adoption"],
                      filled=True, rounded=True, ax=ax_tree2, fontsize=8)
            ax_tree2.set_title("Best Decision Tree")
            st.pyplot(fig_tree2)
            plt.close(fig_tree2)

            # CV Results table
            st.subheader("GridSearchCV Results (Top 10)")
            cv_results = pd.DataFrame(grid_search.cv_results_)
            top_results = cv_results[["param_max_depth", "param_criterion",
                                       "param_min_samples_split", "param_min_samples_leaf",
                                       "mean_test_score", "std_test_score", "rank_test_score"]]
            top_results = top_results.sort_values("rank_test_score").head(10)
            st.dataframe(top_results.reset_index(drop=True), use_container_width=True)

            # Download best model
            model_buffer2 = io.BytesIO()
            joblib.dump(best_model, model_buffer2)
            model_buffer2.seek(0)
            st.download_button(
                label="⬇️ Download Best Model (.pkl)",
                data=model_buffer2,
                file_name="best_decision_tree_model.pkl",
                mime="application/octet-stream",
            )
