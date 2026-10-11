<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=220&text=KESTREL%20HOME&fontSize=48&fontAlignY=38&desc=WARRANTY%20CLAIM%20REVIEW%20%7C%20MACHINE%20LEARNING%20%7C%20FRAUD%20DETECTION&descAlignY=60&animation=fadeIn&fontColor=ffffff&color=0:0F766E,50:0D9488,100:2563EB"/>

<br>

### **AI-Assisted Fraud Detection • Warranty Analytics • Business Intelligence**

<p>
Analysing warranty claims to identify potential fraud risks
<b>using machine learning, data analysis and interactive dashboards.</b>
</p>

<br>

<img src="https://img.shields.io/badge/Warranty%20Analytics-0F766E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Machine%20Learning-0D9488?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Fraud%20Detection-2563EB?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>

<br><br>

<a href="https://xugptpjrimtbus572dwu7m.streamlit.app/">
<img src="https://img.shields.io/badge/Live%20Dashboard-Open-0D9488?style=for-the-badge"/>
</a>

&nbsp;

<a href="https://github.com/rskarthika2005-ai/Kestrel-Home-Warranty-Claim-Review-Variant-C-">
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>

</div>

---

<div align="center">

## ✦ **ABOUT THE PROJECT**

</div>

**Kestrel Home — Warranty Claim Review (Variant C)** is a machine-learning project designed to identify potentially fraudulent warranty claims before payment.

The project analyses historical warranty data, identifies patterns associated with fraud, and generates risk scores to support human review.

The project provides three main capabilities:

- **Claim Risk Scoring** — estimates the risk associated with individual warranty claims.
- **Fraud Pattern Analysis** — identifies trends across claim values, partners and policy periods.
- **Interactive Dashboard** — presents claim-scoring functionality and data-driven findings.

The solution combines **Python, Pandas, Scikit-learn, FastAPI and Streamlit** to build an end-to-end warranty claim review workflow.

---

<div align="center">

## ✦ **PROJECT OBJECTIVE**

</div>

### Identify Potential Fraud Within Limited Review Capacity

The business challenge is to identify potentially fraudulent claims while the team can review only **40 claims per month**.

Only **141 out of 11,146 decided claims** were labelled fraudulent, approximately **1.27%** of the dataset.

This makes accuracy alone misleading: a system that labels every claim as non-fraudulent could achieve approximately **98.7% accuracy while detecting no fraud**.

The project therefore focuses on fraud detection and prioritising claims for human review rather than relying on accuracy alone.

---

<div align="center">

## ✦ **PROJECT OVERVIEW**

</div>

<table align="center">

<tr>
<td align="center" width="25%">

### **12,029**

Raw Rows

</td>
<td align="center" width="25%">

### **11,348**

Unique Claims

</td>
<td align="center" width="25%">

### **681**

Duplicates

</td>
<td align="center" width="25%">

### **2,252**

Test Claims

</td>
</tr>

<tr>
<td align="center">

### **720**

June Claims

</td>
<td align="center">

### **22**

Fraud Cases

</td>
<td align="center">

### **14**

Fraud Cases Detected

</td>
<td align="center">

### **7/7**

Format Checks Passed

</td>
</tr>

</table>

---

<div align="center">

## ✦ **KEY FINDINGS**

</div>

### Fraud Trends

The observed fraud rate increased from **1.08% before May 1, 2026, to 2.53% afterward**. This is an observed association and does not by itself establish the cause.

### Partner Risk Patterns

Newer partners onboarded from October 2025 showed a higher observed fraud rate after the policy change than older partners. These patterns can help prioritise investigation but do not prove individual fraud.

### High-Risk Outlets

The analysis identified outlets including **SP3160, SP3232, SP3318, SP3129 and SP3319** for further review.

### Validation Results

On 720 June 2026 claims, the model selected 38 claims for review. It identified 14 of the 22 labelled fraud cases and incorrectly flagged 24 genuine claims.

---

<div align="center">

## ✦ **DATA PREPROCESSING**

</div>

Before model training, the warranty data is cleaned and checked for quality issues.

| Processing Step | Result |
|---|---:|
| Raw claim rows | 12,029 |
| Unique claims after deduplication | 11,348 |
| Duplicate rows removed | 681 |
| Duplicate rate | 5.7% |
| Claims without a final outcome | 202 |
| Serial numbers with inconsistent formatting | Approximately 30% |

### Data Quality Considerations

- Duplicate claims are removed during preprocessing.
- Claims without a final outcome are excluded from model training.
- Older records from the legacy system may contain labels that do not reliably distinguish undecided claims from genuine claims.
- Serial-number formatting inconsistencies may affect data matching and should be handled carefully.
- The raw input data is kept separate from generated outputs.

**Privacy note:** Original or confidential claim data is not included in the public GitHub repository.

---

<div align="center">

## ✦ **MACHINE LEARNING & VALIDATION**

</div>

The pipeline combines **Logistic Regression and Gradient Boosting** to generate claim risk predictions.

A time-based validation approach is used to test performance on later claims rather than relying only on a random train-test split.

### June 2026 Validation

| Metric | Result |
|---|---:|
| Claims evaluated | 720 |
| Labelled fraud cases | 22 |
| Claims selected for review | 38 |
| Fraud cases detected | 14 |
| Genuine claims incorrectly flagged | 24 |
| Fraud detection rate | 64% |
| Estimated net saving per reviewed claim | ₹284 |

These results come from a small validation sample, so performance and estimated savings are uncertain. High-risk scores should be treated as **review priorities, not proof of fraud**.

---

<div align="center">

## ✦ **APPLICATION ARCHITECTURE**

</div>

The project is organised into separate components for data processing, model training, claim scoring and dashboard presentation.

| Component | Purpose |
|---|---|
| `run_pipeline.py` | Runs the end-to-end processing and prediction pipeline |
| `modules/preprocess.py` | Cleans and prepares warranty claim data |
| `modules/train_model.py` | Trains the machine-learning models |
| `modules/predict_claim.py` | Generates risk predictions |
| `modules/validation.py` | Evaluates model performance |
| `api.py` | Provides the claim-scoring API |
| `dashboard.py` | Displays the interactive dashboard |
| `app.py` | Application entry point |

### Run the Pipeline

```bash
python run_pipeline.py
