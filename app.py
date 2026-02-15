import streamlit as st
import pandas as pd
import joblib
import json
import numpy_financial as npf
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime
import uuid

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Loan Bot", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("loan_model.pkl")

# -----------------------------
# CUSTOM DARK STYLE
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }

.red-box {
    background-color: #7f1d1d;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}

.green-box {
    background-color: #14532d;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}

.blue-box {
    background-color: #1e3a8a;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}

.section-title {
    margin-top: 30px;
    font-size: 24px;
    font-weight: bold;
}

.info-box {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🏦 AI Loan Eligibility & Risk Assessment Bot")
st.caption("Dataset-Driven • Explainable • Compliance-First Decision Engine")

# -----------------------------
# LOAN TYPE
# -----------------------------
st.markdown("### 🏷 Loan Type")
loan_type = st.selectbox(
    "Select Type of Loan",
    ["Personal Loan", "Home Loan", "Car Loan", "Education Loan"]
)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown("## 📋 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    loan_amount = st.number_input("Loan Amount (Rs.)", min_value=1000.0, value=50000.0)
    tenure = st.number_input("Tenure (Months)", min_value=6, value=12)
    employment_years = st.number_input("Employment Years", min_value=0, value=2)
    existing_loans = st.number_input("Existing Loans Count", min_value=0, value=0)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    monthly_income = st.number_input("Monthly Income (Rs.)", min_value=1000.0, value=30000.0)
    existing_emi = st.number_input("Existing EMI (Rs.)", min_value=0.0, value=2000.0)
    age = st.number_input("Age", min_value=18, value=30)

# -----------------------------
# RECEIPT FUNCTION
# -----------------------------
def generate_receipt():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Loan Decision Receipt", styles['Title']))
    elements.append(Spacer(1, 0.3 * inch))

    reference_id = str(uuid.uuid4())[:8]
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    data = [
        ["Reference ID", reference_id],
        ["Date & Time", current_time],
        ["Loan Type", loan_type],
        ["Decision", decision],
        ["Risk Level", risk_level],
        ["Loan Amount", f"Rs. {loan_amount}"],
        ["Tenure", tenure],
        ["Monthly EMI", f"Rs. {round(emi,2)}"],
        ["EMI Ratio", f"{round(emi_ratio,2)} %"],
        ["Approval Probability", f"{round(probability,2)} %"]
    ]

    table = Table(data, colWidths=[2.5 * inch, 3 * inch])
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# -----------------------------
# EVALUATION
# -----------------------------
if st.button("🚀 Evaluate Loan Application"):

    input_data = pd.DataFrame([{
        "age": age,
        "employment_years": employment_years,
        "monthly_income": monthly_income,
        "credit_score": credit_score,
        "existing_emi": existing_emi,
        "existing_loans_count": existing_loans,
        "requested_loan_amount": loan_amount,
        "requested_tenure_months": tenure
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    # Interest rates
    if loan_type == "Personal Loan":
        annual_interest = 0.14
    elif loan_type == "Home Loan":
        annual_interest = 0.085
    elif loan_type == "Car Loan":
        annual_interest = 0.10
    else:
        annual_interest = 0.09

    monthly_interest = annual_interest / 12
    emi = abs(npf.pmt(monthly_interest, tenure, -loan_amount))
    emi_ratio = (emi + existing_emi) / monthly_income * 100

    # Risk Logic
    risk_level = "LOW"
    if credit_score < 650 or emi_ratio > 50:
        risk_level = "HIGH"
    if loan_type == "Personal Loan" and credit_score < 700:
        risk_level = "HIGH"

    decision = "APPROVED" if prediction == 1 and risk_level == "LOW" else "REJECTED"

    # -----------------------------
    # DECISION SUMMARY (BOX STYLE)
    # -----------------------------
    st.markdown("## 📊 Decision Summary")

    colA, colB, colC = st.columns(3)

    with colA:
        if decision == "APPROVED":
            st.markdown(f"<div class='green-box'>✔ {decision}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='red-box'>✖ {decision}</div>", unsafe_allow_html=True)

    with colB:
        st.markdown(f"<div class='blue-box'>Loan Type: {loan_type}</div>", unsafe_allow_html=True)

    with colC:
        if risk_level == "HIGH":
            st.markdown("<div class='red-box'>⚠ HIGH RISK</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='green-box'>LOW RISK</div>", unsafe_allow_html=True)

    # -----------------------------
    # AFFORDABILITY
    # -----------------------------
    st.markdown("## 💰 Affordability Analysis")

    col1, col2 = st.columns(2)
    col1.metric("Monthly EMI", f"Rs. {round(emi,2)}")
    col2.metric("EMI / Income Ratio", f"{round(emi_ratio,2)} %")

    # -----------------------------
    # APPROVAL PROBABILITY
    # -----------------------------
    st.markdown("## 📈 Approval Probability")
    st.progress(int(probability))
    st.write(f"Estimated Approval Probability: {round(probability,2)}%")

    # -----------------------------
    # EXPLAINABLE AI
    # -----------------------------
    st.markdown("## 🧠 Explainable AI Decision")
    st.write(f"• Decision: {decision}")
    st.write(f"• Loan Type: {loan_type}")
    st.write(f"• Risk Level: {risk_level}")
    st.write("• Eligibility: Based on trained ML model + policy rules")

    # -----------------------------
    # RULE IMPACT
    # -----------------------------
    st.markdown("## ❌ Rule Impact Summary")

    if credit_score < 650:
        st.write("• Credit score below required threshold (650)")
    if emi_ratio > 50:
        st.write("• EMI exceeds affordability policy")
    if loan_type == "Personal Loan" and credit_score < 700:
        st.write("• Personal loan requires higher credit score (700+)")

    # -----------------------------
    # WHAT IF
    # -----------------------------
    st.markdown("## 🔄 What-If Suggestion")
    st.markdown(
        "<div class='info-box'>Try increasing tenure or reducing loan amount to improve approval chances.</div>",
        unsafe_allow_html=True
    )

    # -----------------------------
    # DOWNLOAD RECEIPT
    # -----------------------------
    st.markdown("## 📄 Download Receipt")
    pdf_buffer = generate_receipt()

    st.download_button(
        label="⬇ Download Loan Decision Receipt (PDF)",
        data=pdf_buffer,
        file_name="Loan_Decision_Receipt.pdf",
        mime="application/pdf"
    )
