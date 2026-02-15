import pandas as pd
import json
import joblib
import numpy as np

# -----------------------------
# LOAD DATASETS (NO HARDCODING)
# -----------------------------

with open("Lending_Rules_Dataset (1).json") as f:
    lending_rules = json.load(f)

with open("Approval_Threshold_Dataset (1).json") as f:
    approval_thresholds = json.load(f)

model = joblib.load("loan_model (4).pkl")

# -----------------------------
# STATE MACHINE
# -----------------------------

class LoanApplication:

    def __init__(self, applicant_data):
        self.data = applicant_data
        self.state = "SUBMITTED"
        self.explanations = []
        self.probability = 0
        self.final_decision = None

    # -----------------------------
    # ELIGIBILITY CHECK
    # -----------------------------

    def check_eligibility(self):
        self.state = "ELIGIBILITY_CHECK"

        age_min = lending_rules["age_limits"]["min"]
        age_max = lending_rules["age_limits"]["max"]

        if self.data["age"] < age_min or self.data["age"] > age_max:
            self.explanations.append(
                f"Age outside allowed range ({age_min}-{age_max})."
            )
            return "reject"

        for band in lending_rules["credit_score_bands"]:
            min_score = band["min_score"]
            max_score = band["max_score"]

            if min_score is None and self.data["credit_score"] <= max_score:
                return band["decision"]

            if max_score is None and self.data["credit_score"] >= min_score:
                return band["decision"]

            if min_score and max_score:
                if min_score <= self.data["credit_score"] <= max_score:
                    return band["decision"]

        return "review"

    # -----------------------------
    # RISK & AFFORDABILITY
    # -----------------------------

    def risk_assessment(self):
        self.state = "RISK_ASSESSMENT"

        ratio = self.data["existing_emi"] / self.data["monthly_income"]

        for rule in lending_rules["emi_income_ratio"]:
            if ratio <= rule["threshold"]:
                return rule["decision"]

        self.explanations.append(
            f"EMI to income ratio too high: {round(ratio,2)}"
        )
        return "reject"

    # -----------------------------
    # ML PROBABILITY
    # -----------------------------

    def calculate_probability(self):
        self.state = "PROBABILITY_ESTIMATION"

        features = [
            self.data["age"],
            self.data["employment_years"],
            self.data["monthly_income"],
            self.data["credit_score"],
            self.data["existing_emi"],
            self.data["existing_loans_count"],
            self.data["requested_loan_amount"],
            self.data["requested_tenure_months"]
        ]

        prob = model.predict_proba([features])[0].max()
        self.probability = round(prob * 100, 2)

    # -----------------------------
    # FINAL DECISION USING DATASET
    # -----------------------------

    def apply_threshold(self):
        self.state = "FINAL_DECISION"

        thresholds = approval_thresholds["thresholds"]

        approve_limit = thresholds["approve_if_probability_above"]
        review_low, review_high = thresholds["review_if_probability_between"]
        reject_limit = thresholds["reject_if_probability_below"]

        if self.probability >= approve_limit:
            self.final_decision = "Approved"
        elif review_low <= self.probability < review_high:
            self.final_decision = "Review"
        elif self.probability < reject_limit:
            self.final_decision = "Rejected"
        else:
            self.final_decision = "Escalated"

    # -----------------------------
    # FULL PROCESS
    # -----------------------------

    def process(self):

        eligibility = self.check_eligibility()
        if eligibility == "reject":
            self.final_decision = "Rejected"
            return

        risk = self.risk_assessment()
        if risk == "reject":
            self.final_decision = "Rejected"
            return

        self.calculate_probability()
        self.apply_threshold()

    # -----------------------------
    # EXPLAINABLE OUTPUT
    # -----------------------------

    def summary(self):

        print("\n----- LOAN ASSESSMENT REPORT -----")
        print("Current State:", self.state)
        print("Approval Probability:", self.probability, "%")
        print("Final Decision:", self.final_decision)
        print("Explanation:")
        for e in self.explanations:
            print("-", e)
        print("-----------------------------------")


# -----------------------------
# SAMPLE APPLICANT
# -----------------------------

sample_applicant = {
    "age": 30,
    "employment_years": 5,
    "monthly_income": 50000,
    "credit_score": 700,
    "existing_emi": 10000,
    "existing_loans_count": 1,
    "requested_loan_amount": 400000,
    "requested_tenure_months": 48
}

application = LoanApplication(sample_applicant)
application.process()
application.summary()
