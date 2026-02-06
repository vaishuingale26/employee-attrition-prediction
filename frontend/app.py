import streamlit as st
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
header, footer {visibility: hidden;}
body { background-color: #f8fafc; }

.card {
    background: white;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-card {
    background: #f1f5f9;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

.high-risk {
    background: #fee2e2;
    border-left: 6px solid #dc2626;
    padding: 20px;
    border-radius: 10px;
}

.low-risk {
    background: #dcfce7;
    border-left: 6px solid #16a34a;
    padding: 20px;
    border-radius: 10px;
}

.title {
    font-size: 34px;
    font-weight: 800;
    color: #0f172a;
}

.subtitle {
    color: #475569;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("""
<div class="card">
    <div class="title">HR Attrition Prediction Dashboard</div>
    <div class="subtitle">
        Predict employee attrition risk using machine learning and HR analytics
    </div>
</div>
""", unsafe_allow_html=True)

# ================= MAPPINGS =================
edu = {"Below College":1,"College":2,"Bachelor":3,"Master":4,"Doctor":5}
job_sat = {"Very Dissatisfied":1,"Dissatisfied":2,"Satisfied":3,"Very Satisfied":4}
env_sat = {"Poor":1,"Average":2,"Good":3,"Excellent":4}
job_inv = {"Low":1,"Medium":2,"High":3,"Very High":4}
wlb = {"Poor":1,"Fair":2,"Good":3,"Excellent":4}
perf = {"Low":1,"Good":2,"Excellent":3,"Outstanding":4}
job_lvl = {"Entry Level":1,"Junior":2,"Mid Level":3,"Senior":4,"Manager":5}

# ================= SIDEBAR INPUT =================
st.sidebar.title("🧑 Employee Details")

Age = st.sidebar.slider("Age", 18, 60, 30)
MonthlyIncome = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
DailyRate = st.sidebar.number_input("Daily Rate", 100, 1500, 800)
DistanceFromHome = st.sidebar.slider("Distance From Home (km)", 0, 50, 5)

st.sidebar.subheader("📚 Job & Education")

Education = edu[st.sidebar.selectbox("Education Level", edu.keys())]
JobLevel = job_lvl[st.sidebar.selectbox("Job Level", job_lvl.keys())]
JobSatisfaction = job_sat[st.sidebar.selectbox("Job Satisfaction", job_sat.keys())]
EnvironmentSatisfaction = env_sat[st.sidebar.selectbox("Environment Satisfaction", env_sat.keys())]
JobInvolvement = job_inv[st.sidebar.selectbox("Job Involvement", job_inv.keys())]

st.sidebar.subheader("📈 Experience")

TotalWorkingYears = st.sidebar.slider("Total Working Years", 0, 40, 8)
YearsAtCompany = st.sidebar.slider("Years At Company", 0, 40, 5)
NumCompaniesWorked = st.sidebar.slider("Companies Worked", 0, 10, 1)
WorkLifeBalance = wlb[st.sidebar.selectbox("Work Life Balance", wlb.keys())]
PercentSalaryHike = st.sidebar.slider("Salary Hike (%)", 5, 30, 13)
PerformanceRating = perf[st.sidebar.selectbox("Performance Rating", perf.keys())]

predict = st.sidebar.button("🔍 Predict Attrition")

# ================= MAIN CONTENT =================
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Overview")

    if predict:
        payload = {
            "Age": Age,
            "DailyRate": DailyRate,
            "DistanceFromHome": DistanceFromHome,
            "Education": Education,
            "EnvironmentSatisfaction": EnvironmentSatisfaction,
            "JobInvolvement": JobInvolvement,
            "JobLevel": JobLevel,
            "JobSatisfaction": JobSatisfaction,
            "MonthlyIncome": MonthlyIncome,
            "NumCompaniesWorked": NumCompaniesWorked,
            "PercentSalaryHike": PercentSalaryHike,
            "PerformanceRating": PerformanceRating,
            "TotalWorkingYears": TotalWorkingYears,
            "WorkLifeBalance": WorkLifeBalance,
            "YearsAtCompany": YearsAtCompany
        }

        result = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        ).json()["Attrition"]

        if result == "Yes":
            st.markdown("""
            <div class="high-risk">
                <h3>⚠ High Attrition Risk</h3>
                Employee may leave the organization soon.
                Immediate HR action recommended.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="low-risk">
                <h3>✅ Low Attrition Risk</h3>
                Employee is likely to stay with the organization.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Fill details from sidebar and click Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= BULK PREDICTION =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## 📂 Bulk Prediction & Evaluation")

uploaded_file = st.file_uploader("Upload HR CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if "Attrition" not in df.columns:
        st.error("Dataset must contain 'Attrition' column")
    else:
        X = df.drop("Attrition", axis=1)
        y_true = df["Attrition"].map({"Yes":1,"No":0})

        preds = []
        for _, row in X.iterrows():
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json=row.to_dict()
            ).json()["Attrition"]
            preds.append(1 if res == "Yes" else 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_true, preds):.2f}")
        c2.metric("Precision", f"{precision_score(y_true, preds):.2f}")
        c3.metric("Recall", f"{recall_score(y_true, preds):.2f}")
        c4.metric("F1 Score", f"{f1_score(y_true, preds):.2f}")

st.markdown("</div>", unsafe_allow_html=True)
