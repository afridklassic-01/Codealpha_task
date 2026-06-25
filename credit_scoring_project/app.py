# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 1. Page Configuration & Professional Styling
st.set_page_config(
    page_title="Enterprise Risk Engine", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. Main Dashboard Header Layout
st.title("🛡️ Institutional Credit Underwriting & Risk Engine")
st.caption("Advanced automated risk-tiering and capital exposure calculator driven by machine learning optimization.")
st.markdown("---")

model_path = 'models/kaggle_rf_model.pkl'
scaler_path = 'models/kaggle_scaler.pkl'

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.error("🚨 Deployment dependencies missing. Run 'python train.py' first to build the training assets.")
else:
    @st.cache_resource
    def load_production_assets():
        with open(model_path, 'rb') as m_f:
            model = pickle.load(m_f)
        with open(scaler_path, 'rb') as s_f:
            scaler = pickle.load(s_f)
        return model, scaler

    model, scaler = load_production_assets()

    # 3. Form Presentation Layout
    st.subheader("📋 Applicant Credit Parameter Metrics")
    
    with st.form("underwriting_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age of Primary Borrower", min_value=18, max_value=100, value=40)
            income = st.number_input("Verified Monthly Income ($)", min_value=0, value=6500, step=500)
            
        with col2:
            utilization = st.slider("Revolving Credit Utilization Rate", 0.0, 2.0, 0.30, 
                                    help="Total outstanding balance on unsecured lines divided by total credit limits.")
            debt_ratio = st.number_input("Calculated Debt-to-Income Ratio", min_value=0.0, max_value=15.0, value=0.35, step=0.05)
            
        with col3:
            past_due = st.selectbox("Historical 30-59 Days Delinquency (Last 2 Years)", [0, 1, 2, 3, 4, "5+"])
            past_due_num = 5 if past_due == "5+" else past_due

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Run Deep Underwriting Risk Evaluation", type="primary")

    # 4. Analysis & Advanced Business Logic Pipeline
    if submit_button:
        # Construct inference vector
        input_data = pd.DataFrame([{
            'age': age,
            'MonthlyIncome': income,
            'DebtRatio': debt_ratio,
            'RevolvingUtilizationOfUnsecuredLines': utilization,
            'NumberOfTime30-59DaysPastDueNotWorse': past_due_num
        }])
        
        # Scale and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        safety_score = 1.0 - probability

        # 🔥 NEW FEATURE 1: Credit Tier Segmentation Matrix
        if probability < 0.15:
            credit_tier = "🥇 PRIME (Excellent Credit)"
            tier_color = "green"
        elif probability < 0.40:
            credit_tier = "🥈 NEAR-PRIME (Good Credit)"
            tier_color = "orange"
        else:
            credit_tier = "🥉 SUBPRIME (High Risk Exposure)"
            tier_color = "red"

        # 🔥 NEW FEATURE 2: Maximum Eligible Loan Recommendation
        # Banks limit total monthly debt obligations to roughly 45% of gross income
        max_allowed_monthly_debt = income * 0.45
        current_monthly_debt = income * debt_ratio
        available_monthly_buffer = max_allowed_monthly_debt - current_monthly_debt
        
        if available_monthly_buffer > 0 and prediction == 0:
            # Estimate a safe loan amount assuming a 5-year tenure (60 months) at standard interest rates
            estimated_loan_cap = available_monthly_buffer * 45  
            estimated_loan_cap = round(estimated_loan_cap, -3) # Round to nearest thousand
        else:
            estimated_loan_cap = 0.0

        # 🔥 NEW FEATURE 3: Automated Adverse Action Reason Codes
        reason_codes = []
        if utilization > 0.50:
            reason_codes.append("⚠️ Excessive credit limit utilization balances.")
        if debt_ratio > 0.45:
            reason_codes.append("⚠️ Debt-to-income bounds exceed safe parameters.")
        if past_due_num > 0:
            reason_codes.append("⚠️ Recent critical payment delinquency history detected.")
        if age < 25:
            reason_codes.append("⚠️ Limited historical credit portfolio duration.")

        st.markdown("---")
        st.subheader("🎯 Automated Underwriting Risk Analysis")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown("#### 📊 Machine Learning Risk Output")
            if prediction == 1 or probability >= 0.40:
                st.error("❌ **Application Result: DECLINED**")
                st.metric(label="Default Probability Factor", value=f"{probability * 100:.1f}%")
            else:
                st.success("✅ **Application Result: APPROVED**")
                st.metric(label="Credit Safety Score", value=f"{safety_score * 100:.1f}%")

        with res_col2:
            st.markdown("#### 🏅 Institutional Risk Tiering")
            st.markdown(f"<h3 style='color:{tier_color};'>{credit_tier}</h3>", unsafe_allow_html=True)
            if estimated_loan_cap > 0:
                st.metric(label="Maximum Recommended Capital Exposure", value=f"${estimated_loan_cap:,.2f}")
            else:
                st.metric(label="Maximum Recommended Capital Exposure", value="$0.00 (Risk Limit Exceeded)")

        with res_col3:
            st.markdown("#### 📈 Underwriting Risk Distribution")
            chart_data = pd.DataFrame({
                'Risk Parameter': ['Safety Multiplier', 'Default Vector'],
                'Percentage (%)': [safety_score * 100, probability * 100]
            }).set_index('Risk Parameter')
            st.bar_chart(chart_data, color="#1f77b4" if prediction == 0 else "#d62728")

        # Dynamic Adverse Actions Display
        if prediction == 1 or len(reason_codes) > 0:
            st.markdown("---")
            st.subheader("🔍 Adverse Underwriting Flags & Reason Codes")
            for reason in reason_codes:
                st.warning(reason)
            if not reason_codes:
                st.info("💡 General risk classification override based on collective baseline pattern trends.")

        # Formal Executive Data Audit Table
        st.markdown("---")
        st.markdown("### 📑 Applicant Financial Factsheet")
        
        report_df = pd.DataFrame({
            "Financial Metric Indicator": [
                "Primary Account Holder Age",
                "Verified Gross Monthly Income",
                "Calculated Debt-to-Income Margin",
                "Revolving Credit Limit Utilization Rate",
                "Historical Delinquencies (30-59 Days Past Due)"
            ],
            "Recorded Value": [
                f"{age} Years Old",
                f"${income:,.2f}",
                f"{debt_ratio * 100:.1f}% of Income",
                f"{utilization * 100:.1f}% Utilized",
                f"{past_due} Incidents Recorded"
            ],
            "Risk Evaluation Status": [
                "🟢 Standard Bracket" if 25 <= age <= 65 else "🟡 Non-Standard Bracket",
                "🟢 Solid Income Flow" if income >= 5000 else "🟡 Moderate Income Flow",
                "🟢 Low Debt Burden" if debt_ratio <= 0.40 else "🔴 Elevated Debt Exposure",
                "🟢 Optimized Credit Usage" if utilization <= 0.35 else "🔴 Over-Extended Credit Activity",
                "🟢 Pristine Payment History" if past_due_num == 0 else "🔴 Delinquency Flag Raised"
            ]
        })
        st.table(report_df)

