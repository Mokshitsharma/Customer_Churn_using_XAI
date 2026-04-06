# 🔍 Customer Churn Prediction with Explainable & Actionable AI (XAI)

> An end-to-end ML pipeline that doesn't just predict customer churn — it **explains why** each customer is at risk using SHAP and **recommends what to do** about it, all wrapped in an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6E25?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-FF6B6B?style=for-the-badge)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 1. 🎯 Project Name

**Customer Churn Prediction with Explainable & Actionable AI (XAI)**

---

## 2. 🔍 Problem Statement

Telecom companies lose billions annually to customer churn — but traditional ML approaches only answer **"will this customer leave?"**. They fail to address the two business-critical questions that actually drive retention:

- **Why** is this specific customer likely to churn? (individual-level reasoning, not just global feature importance)
- **What specific actions** can the business take to prevent it?

A model that outputs a churn probability with no explanation is not actionable for a retention team. Account managers, CRM teams, and business analysts need customer-level narratives — not just probabilities — to intervene effectively.

---

## 3. 💡 My Solution

Built a 3-layer **Explainable + Actionable AI system**:

**Layer 1 — Prediction:** XGBoost classifier trained on 7,043 telecom customers, producing a churn probability per customer.

**Layer 2 — Explanation (XAI):** SHAP `TreeExplainer` computes per-customer Shapley values. These are translated from numerical SHAP scores into **human-readable natural language reasons** (e.g., *"Month-to-month contract significantly increases churn risk"*), with both:
- **SHAP Waterfall plot** — shows how each feature pushes the prediction above/below the base rate
- **SHAP Summary Bar** — global feature importance across the dataset

**Layer 3 — Actionable Recommendations:** The system maps SHAP-identified risk factors to **specific business interventions** (e.g., *"Offer a one-year contract upgrade with a loyalty discount"*), delivered per customer in the UI.

**Deployment:** Interactive **Streamlit dashboard** with 19 input fields matching the real Telco dataset schema, real-time inference, and instant SHAP visualization generation.

---

## 4. 📐 System Architecture

```
┌─────────────────────────────────────────────────────┐
│              DATA PIPELINE (main.py)                 │
│                                                      │
│  load_data() ──► preprocess_data()                  │
│    │                    │                           │
│    ▼                    ▼                           │
│  Raw CSV          Encode + Split                    │
│  (7,043 rows)     (stratified)                      │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│              ML LAYER                                │
│                                                      │
│  train_model() → XGBoost Classifier                 │
│  evaluate_model() → Accuracy, ROC-AUC, Report       │
│  save_model() → models/churn_model.pkl              │
│  save feature_columns.pkl                           │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│              XAI LAYER                               │
│                                                      │
│  shap_explainer.py → TreeExplainer                  │
│  shap_values[customer] → top contributing features  │
│  predict_with_explanation() →                       │
│    {                                                 │
│      prediction: 0 or 1,                            │
│      probability: float,                            │
│      reasons: [human-readable strings],             │
│      suggestions: [business recommendations]        │
│    }                                                 │
│  Outputs: SHAP Waterfall + Summary Bar plots        │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│              STREAMLIT DASHBOARD (app.py)            │
│                                                      │
│  19-field customer input form                       │
│  → Churn prediction + probability                   │
│  → "Why this prediction?" (NL reasons)              │
│  → "How to reduce churn?" (suggestions)             │
│  → SHAP Waterfall visual                            │
│  → SHAP Feature Importance Bar visual               │
└─────────────────────────────────────────────────────┘
```

---

## 5. 🛠️ Skills Used

| Category | Technologies |
|---|---|
| **Machine Learning** | XGBoost (gradient boosting), Scikit-learn (preprocessing, train/test split, evaluation) |
| **Explainable AI (XAI)** | SHAP (TreeExplainer, waterfall plots, summary plots, Shapley values) |
| **NLP / Reasoning Layer** | Custom SHAP-to-natural-language translation engine (`predict_with_explanation`) |
| **Data Processing** | Pandas, NumPy, label encoding, stratified splits |
| **Visualization** | Matplotlib, Seaborn, SHAP waterfall + bar plots |
| **Deployment / UI** | Streamlit (interactive dashboard, `@st.cache_resource`) |
| **Model Persistence** | Joblib (`.pkl` serialization for model + feature columns) |
| **Software Engineering** | Modular `src/` architecture (data, models, explainability, utils packages) |

---

## 6. 📊 Project Metrics

| Metric | Value |
|---|---|
| **Model** | XGBoost Classifier |
| **Accuracy** | ~80–85% |
| **ROC-AUC Score** | ~0.85+ |
| **Dataset Size** | 7,043 customers |
| **Features** | 19 input features (demographics, services, billing) |
| **Target Variable** | Churn (Yes/No) → binary classification |
| **Class Imbalance Handling** | Stratified train/test split |
| **Explanation Types** | Per-customer NL reasons + business suggestions + 2 SHAP plots |
| **Inference Speed** | Real-time (single customer) |
| **Model Artifacts** | `churn_model.pkl` + `feature_columns.pkl` |

---

## 7. 📦 Dataset Details

| Property | Detail |
|---|---|
| **Source** | Kaggle — [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Provider** | IBM Sample Dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) |
| **Rows** | 7,043 customers |
| **Columns** | 21 (19 features + customerID + target) |
| **Target** | `Churn` — Yes (26.5%) / No (73.5%) → imbalanced |
| **Feature Categories** | Demographics (gender, SeniorCitizen, Partner, Dependents), Account (tenure, Contract, PaperlessBilling, PaymentMethod), Services (PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies), Billing (MonthlyCharges, TotalCharges) |
| **Preprocessing** | Label encoding for categoricals, TotalCharges coerced to numeric (has whitespace entries), stratified split |

---

## 8. 📁 Folder Structure

```
Customer_Churn_using_XAI/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│
├── src/
│   ├── data/
│   │   ├── load_data.py              # CSV ingestion
│   │   └── preprocess.py            # Encoding, splitting, cleaning
│   ├── models/
│   │   ├── train.py                 # XGBoost training
│   │   ├── evaluate.py              # Accuracy, ROC-AUC, classification report
│   │   ├── save_model.py            # Joblib serialization
│   │   └── predict.py               # predict_with_explanation(), align_features()
│   ├── explainability/
│   │   └── shap_explainer.py        # TreeExplainer + SHAP plot generation
│   └── utils/
│       └── (helper utilities)
│
├── models/
│   ├── churn_model.pkl              # Trained XGBoost model
│   └── feature_columns.pkl          # Saved feature column order
│
├── notebooks/
│   └── (EDA + experimentation notebooks)
│
├── reports/
│   └── shap_outputs/                # Saved SHAP visualization files
│
├── app.py                           # Streamlit dashboard (19-field input + XAI output)
├── main.py                          # Training + prediction pipeline orchestrator
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 9. ⚙️ Why This Tech Stack?

| Choice | Reason |
|---|---|
| **XGBoost** | State-of-the-art for tabular/structured data; handles mixed types, missing values, and class imbalance well; compatible with SHAP `TreeExplainer` for exact Shapley value computation |
| **SHAP (TreeExplainer)** | Industry-standard XAI library; TreeExplainer is the only method that computes *exact* Shapley values for tree-based models in polynomial time; produces locally faithful per-customer explanations |
| **Stratified Split** | Preserves the natural 26.5/73.5 class distribution in both train/test sets — critical for calibrated probability outputs |
| **Joblib** | Efficient binary serialization for large sklearn/XGBoost model objects, faster than pickle for numpy arrays |
| **Streamlit** | Zero-frontend code needed; `@st.cache_resource` caches model on first load; ideal for rapid ML dashboard deployment |
| **Modular `src/` architecture** | Separates data, model, explainability, and utils concerns — mirrors production ML engineering patterns used in enterprise MLOps pipelines |
| **Matplotlib** | Direct SHAP plot integration; SHAP's waterfall_legacy + summary_plot require matplotlib figure objects for Streamlit rendering |

---

## 10. 🚀 Future Improvements

- **FastAPI backend** — wrap `predict_with_explanation()` as a REST endpoint for CRM integration (Salesforce, HubSpot)
- **Batch prediction mode** — CSV upload in Streamlit with per-row churn scores and bulk SHAP summaries
- **What-If Simulator** — allow users to change specific features (e.g., switch Contract from Month-to-month to Two year) and see predicted churn drop in real time
- **LIME comparison** — add LIME explanations alongside SHAP for cross-method validation
- **Class imbalance handling** — implement SMOTE oversampling or `scale_pos_weight` tuning in XGBoost for improved recall on the minority churn class
- **Model comparison** — benchmark XGBoost vs LightGBM vs Random Forest with SHAP on the same dataset
- **Deployment** — containerize with Docker and deploy to Streamlit Cloud / AWS SageMaker
- **Database integration** — persist per-customer predictions + SHAP explanations to PostgreSQL for historical tracking
- **Email alert system** — trigger CRM alerts when a customer's churn probability crosses a configurable threshold

---

## 11. ⭐ Rating

```
╔══════════════════════════════════════╗
║   Project Rating:  8.9 / 10         ║
║                                      ║
║  ML Depth             █████████░  9  ║
║  XAI Implementation   ██████████  10 ║
║  Business Relevance   ██████████  10 ║
║  Code Architecture    █████████░  9  ║
║  Actionability Layer  ██████████  10 ║
║  Deployment Quality   ████████░░  8  ║
║  Documentation        ████████░░  8  ║
╚══════════════════════════════════════╝
```

**Why 8.9?** This is your most technically differentiated ML project. The three-layer design — predict → explain (SHAP) → recommend (NL business actions) — is exactly the pattern used in production churn management systems at Jio, Airtel, and Netflix. Most ML students stop at the model; you went three steps further. The modular `src/` architecture (`data/`, `models/`, `explainability/`, `utils/`) mirrors real MLOps pipeline structure. Reaches 9.5+ with a FastAPI deployment and What-If simulator.

---

## 12. 🎯 Preferred Role for This Project

**Data Scientist / ML Engineer** — Primary  
**Data Analyst (Telecom/Fintech/SaaS)** — Strong Secondary  
**Business Analyst (with Analytics)** — Supporting

This project is your **strongest technical ML showcase** for DS/ML roles. The SHAP layer shows you understand model interpretability — a skill explicitly demanded in regulated industries (BFSI, Telecom, Healthcare) where black-box predictions are insufficient.

---

## ▶️ Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python main.py
```
> Updates `models/churn_model.pkl`, `models/feature_columns.pkl`, and outputs SHAP plots to `reports/shap_outputs/`

### Launch the Dashboard
```bash
streamlit run app.py
```
> Opens at `http://localhost:8501`

---

## 📬 Contact

**Mokshit Sharma** — AI & Data Science Engineer  
📧 sharman48520@gmail.com  
🌐 [mokshitsharma27.vercel.app](https://mokshitsharma27.vercel.app)

---

*Built with Python, XGBoost, SHAP, Scikit-learn, Streamlit & Matplotlib*
