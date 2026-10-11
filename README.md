# Kestrel Home — Warranty Claim Review (Variant C)

**Developed by:** Karthika S  
**Email:** rskarthika08@gmail.com

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=220&text=KESTREL%20HOME&fontSize=48&fontAlignY=38&desc=WARRANTY%20CLAIM%20REVIEW%20%7C%20MACHINE%20LEARNING%20%7C%20FRAUD%20DETECTION&descAlignY=60&animation=fadeIn&fontColor=ffffff&color=0:0F766E,50:0D9488,100:2563EB" alt="Kestrel Home Warranty Claim Review"/>

### AI-Assisted Fraud Detection • Warranty Analytics • Business Intelligence

Analysing warranty claims to identify potential fraud risks using machine learning, data analysis and interactive dashboards.

<img src="https://img.shields.io/badge/Warranty%20Analytics-0F766E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Machine%20Learning-0D9488?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Fraud%20Detection-2563EB?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>

<br/><br/>

<a href="https://xugptpjrimtbus572dwu7m.streamlit.app/">
<img src="https://img.shields.io/badge/Live%20Dashboard-Open-0D9488?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live Dashboard"/>
</a>

<a href="https://github.com/rskarthika2005-ai/Kestrel-Home-Warranty-Claim-Review-Variant-C-">
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repository"/>
</a>

</div>

---

<div align="center">

## ✦ ABOUT THE PROJECT

</div>

**Kestrel Home — Warranty Claim Review (Variant C)** is a machine-learning project designed to identify potentially fraudulent warranty claims before payment.

The project analyses historical warranty data, identifies patterns associated with fraud, and generates risk scores to support human investigation.

The solution provides three main capabilities:

- **Claim Risk Scoring:** Estimates the risk associated with individual warranty claims.
- **Fraud Pattern Analysis:** Identifies trends across claim values, partners, outlets and policy periods.
- **Interactive Dashboard:** Presents claim-scoring functionality, fraud trends and validation findings.

The project combines **Python, Pandas, Scikit-learn, FastAPI and Streamlit** to create an end-to-end warranty claim review workflow.

---

<div align="center">

## ✦ PROJECT OBJECTIVE

</div>

### Identify Potential Fraud Within Limited Review Capacity

The business challenge is to identify potentially fraudulent claims while the investigation team can review only **40 claims per month**.

Only **141 out of 11,146 decided claims** were labelled fraudulent, representing approximately **1.27%** of the dataset.

A system that classifies every claim as non-fraudulent could achieve approximately **98.7% accuracy while detecting no fraud**.

Therefore, overall accuracy alone is not an appropriate measure of success.

The primary objective is to prioritise claims for investigation while balancing fraud detection against the cost of incorrectly flagging genuine customers. Holding a genuine claim costs approximately ₹380 in customer goodwill.

---

<div align="center">

## ✦ PROJECT OVERVIEW

</div>

<table align="center">
<tr>
<td align="center" width="25%">

### 12,029

Raw Rows

</td>
<td align="center" width="25%">

### 11,348

Unique Claims

</td>
<td align="center" width="25%">

### 681

Duplicates

</td>
<td align="center" width="25%">

### 2,252

Test Claims

</td>
</tr>
<tr>
<td align="center">

### 720

June Claims

</td>
<td align="center">

### 22

Fraud Cases

</td>
<td align="center">

### 14

Fraud Cases Detected

</td>
<td align="center">

### 7/7

Format Checks Passed

</td>
</tr>
</table>

---

<div align="center">

## ✦ KEY FINDINGS

</div>

### Fraud Trends

The observed fraud rate increased from **1.08% before 1 May 2026 to 2.53% afterward**, following the introduction of automatic approval for smaller claims.

This is an observed association and does not independently establish that the policy change caused the increase.

### Partner Risk Patterns

Newer partners onboarded from October 2025 had an observed fraud rate of **16.1% after the policy change**, compared with **0.17% for older partners**.

Smaller claims from newer partners included 34 fraud cases among 161 claims, representing **21.1%**.

### High-Risk Outlets

The analysis identified the following outlets for further investigation:

- SP3160
- SP3232
- SP3318
- SP3129
- SP3319

The findings suggest that risk was concentrated in certain outlets rather than being uniform across all newer partners. These are risk indicators, not proof of wrongdoing.

### Policy Compliance

The investigation identified **27 claims worth ₹2,000 or more that were paid without inspection after 1 May 2026**. These claims warrant a separate policy-compliance review.

### Model Validation

On 720 June 2026 claims, the model selected 38 claims for review. It detected 14 of the 22 labelled fraud cases and incorrectly flagged 24 genuine claims.

---

<div align="center">

## ✦ DATA PREPROCESSING

</div>

Before model training, the warranty data was cleaned and checked for quality issues.

| Processing Step | Result |
|---|---:|
| Raw claim rows | 12,029 |
| Unique claims after deduplication | 11,348 |
| Duplicate rows removed | 681 |
| Duplicate rate | 5.7% |
| Claims without final outcomes | 202 |
| Serial numbers with inconsistent formatting | Approximately 30% |
| Legacy Zoho claims labelled non-fraud | 4,560 |

### Data Quality Considerations

- Duplicate submissions are removed during preprocessing.
- Claims without final investigation outcomes are excluded from model training.
- Older Zoho records may contain unreliable labels because some undecided claims were labelled non-fraud.
- Serial-number inconsistencies may affect record matching and analysis.
- Policy-compliance findings are analysed separately from model predictions.
- Original input data is kept separate from generated reports.

**Privacy note:** Original or confidential claim datasets should not be committed to the public GitHub repository.

---

<div align="center">

## ✦ MACHINE LEARNING & VALIDATION

</div>

The pipeline uses **Logistic Regression and Gradient Boosting** to generate claim-risk predictions. Their predictions are combined in the implemented workflow.

A time-based validation approach evaluates the model on later claims, helping reduce the risk of using future information when evaluating past claims.

### June 2026 Validation Results

| Metric | Result |
|---|---:|
| Claims evaluated | 720 |
| Labelled fraud cases | 22 |
| Claims selected for review | 38 |
| Fraud cases detected | 14 |
| Genuine claims incorrectly flagged | 24 |
| Fraud detection rate | 64% |
| Estimated net saving per reviewed claim | ₹284 |

The fraud detection rate is calculated as 14 detected fraud cases divided by 22 labelled fraud cases, or approximately 64%.

The validation sample is small, so detection rates and financial estimates are uncertain. The model performed better than the simple comparison approaches evaluated, including reviewing the highest-value claims or prioritising all new partners.

The main evaluation principle is that **fraud detected and net money saved per claim reviewed are more useful business measures than overall accuracy alone**.

---

<div align="center">

## ✦ APPLICATION ARCHITECTURE

</div>

The application separates data processing, model training, claim scoring, API functionality and dashboard presentation.

| Component | Purpose |
|---|---|
| `run_pipeline.py` | Runs the complete processing and prediction pipeline |
| `modules/preprocess.py` | Cleans and prepares claim data |
| `modules/train_model.py` | Trains and validates the models |
| `modules/predict_claim.py` | Generates individual claim-risk predictions |
| `modules/validation.py` | Performs validation and output checks |
| `api.py` | Provides the FastAPI scoring service |
| `dashboard.py` | Displays the interactive dashboard |
| `app.py` | Main application entry point |

### Run the Pipeline

```bash
python run_pipeline.py
```

The pipeline performs the following operations:

1. Loads and preprocesses the input data.
2. Removes duplicate submissions and prepares model features.
3. Analyses fraud patterns across claims and partners.
4. Trains the machine-learning models.
5. Evaluates model performance using time-based validation.
6. Generates claim-level risk predictions.
7. Saves the resulting reports and prediction files.

### Generated Outputs

| Output File | Purpose |
|---|---|
| `predictions.csv` | Final predictions for 2,252 test claims |
| `scored_claims.csv` | Risk scores, risk levels and available explanations |
| `data_quality_report.csv` | Data quality and preprocessing findings |
| `fraud_pattern_analysis.csv` | Fraud trends and partner or outlet patterns |
| `validation_report.csv` | Model validation results |

All **seven submission-format checks passed** in the recorded run.

---

<div align="center">

## ✦ PROJECT STRUCTURE

</div>

The project is divided into separate modules so that data cleaning, machine learning, scoring, validation and presentation can be maintained independently.

### Directory Structure

```text
Kestrel-Home-Warranty-Claim-Review/
│
├── data/                          # Private input data; not published
│
├── source/                        # Configuration and optional AI setup
│   ├── config.py                  # File paths, business rules and settings
│   ├── llm_client.py              # Optional GPT explanation integration
│   └── prompts.py                 # Instructions for claim explanations
│
├── modules/                       # Core processing and ML logic
│   ├── preprocess.py              # Data cleaning and feature engineering
│   ├── train_model.py             # Model training and validation
│   ├── analyze_claim.py            # Fraud pattern analysis and explanations
│   ├── predict_claim.py            # Individual claim risk scoring
│   ├── generate_predictions.py     # Batch predictions
│   └── validation.py               # Validation and output checks
│
├── output/                        # Generated CSV reports
├── model/                         # Saved model and settings
├── evidence/                      # Charts and validation evidence
├── tests/                         # Tests, where included
│
├── app.py                         # Main application entry point
├── api.py                         # FastAPI scoring service
├── dashboard.py                   # Streamlit dashboard
├── run_pipeline.py                # End-to-end pipeline
├── requirements.txt               # Python dependencies
└── README.md                       # Project documentation
```

*This is the intended logical structure. Actual filenames should match the repository.*

### Component Responsibilities

**1. `data/` — Input Data**

Contains the original claim, partner, product and test datasets used by the pipeline. The datasets are kept separate from generated reports and are not included in the public repository.

**2. `source/` — Configuration and AI Integration**

- `config.py` centralises file paths, business rules and model settings.
- `llm_client.py` handles the optional GPT-4o-mini connection and usage tracking.
- `prompts.py` contains the instructions used to generate concise claim explanations.

**3. `modules/` — Processing and Machine Learning**

- `preprocess.py` cleans data, removes duplicates and prepares features.
- `train_model.py` trains Logistic Regression and Gradient Boosting models.
- `analyze_claim.py` supports fraud-pattern analysis and explanations.
- `predict_claim.py` scores individual claims.
- `generate_predictions.py` creates predictions for the test dataset.
- `validation.py` checks data quality and prediction formats.

**4. `output/` — Generated Results**

Stores predictions, scored claims, data-quality reports, fraud-pattern analysis and validation results.

**5. `model/` and `evidence/` — Supporting Artefacts**

The `model/` directory stores the trained model and associated settings. The `evidence/` directory stores supporting charts, confusion matrices and validation analysis.

**6. Main Application Files**

- `run_pipeline.py` runs the complete pipeline.
- `api.py` exposes claim-scoring functionality through FastAPI.
- `dashboard.py` presents the interactive Streamlit interface.
- `app.py` acts as the main application entry point according to the configured launch process.
- `requirements.txt` lists the Python dependencies.

### End-to-End Workflow

```text
Input Data
    ↓
Data Cleaning and Deduplication
    ↓
Feature Engineering
    ↓
Fraud Pattern Analysis
    ↓
Model Training
    ↓
Time-Based Validation
    ↓
Claim Risk Scoring
    ↓
CSV Reports and Predictions
    ↓
FastAPI Scoring Service
    ↓
Streamlit Dashboard
    ↓
Human Investigation
```

### Why I Chose This Architecture

Python and Scikit-learn support the data-processing and machine-learning workflow. FastAPI exposes scoring functionality through an API, while Streamlit presents the findings in a dashboard designed for business users.

This structure allows batch predictions and individual claim scoring to use the same underlying model workflow. It also keeps the final investigation decision with a human reviewer.

---

<div align="center">

## ✦ STREAMLIT DASHBOARD

</div>

The Streamlit dashboard provides a visual interface for interacting with claim-scoring functionality and exploring fraud-analysis results.

### 1. Score a Claim

This section allows employees to enter claim details and inspect the available scoring results.

Depending on the configured implementation, the results include:

- Fraud-risk score or estimated probability.
- HIGH, MEDIUM or LOW risk classification.
- Recommended review action.
- Supporting reasons or an optional AI-generated explanation.

### 2. What the Data Says

This section presents the main findings from the historical analysis:

- Fraud trends before and after the May 2026 policy change.
- Partner and outlet risk patterns.
- Claims that may warrant further investigation.
- Financial findings and review priorities.
- Model validation results and limitations.

### Why Streamlit?

Instead of asking Ritu Deshpande to examine terminal output or several CSV files, Streamlit provides one interface for reviewing suspicious claims and understanding the evidence behind the analysis.

FastAPI also makes the scoring functionality accessible to other applications through an API endpoint.

**Deployment note:** The hosted dashboard must have access to the model and required application dependencies. Because private data is excluded from GitHub, the deployed application should use a safe configuration that does not depend on publishing confidential datasets.

---

<div align="center">

## ✦ AI TOOLS & COST

</div>

The project uses local machine learning for core fraud detection and an optional generative AI feature for natural-language claim explanations.

This design keeps the main pipeline independent of paid generative AI API calls.

### AI Tools Used

| Tool | Purpose | Cost Consideration |
|---|---|---|
| Python | Pipeline and application development | No separate language usage fee |
| Pandas | Data cleaning and analysis | Local processing |
| Scikit-learn | Model training, scoring and evaluation | Local processing |
| FastAPI | Claim-scoring API | Hosting may incur costs |
| Streamlit | Interactive dashboard | Hosting depends on deployment |
| GPT-4o-mini | Optional claim explanations | API charges may apply when enabled |
| Claude | Planning, coding assistance, debugging and validation support | Depends on the account and service used |

### 1. Core Machine Learning

Logistic Regression and Gradient Boosting perform the core fraud-risk prediction work using Scikit-learn.

Training and scoring run locally and do not require a generative AI API request for every claim.

### 2. Optional GPT-4o-mini Explanations

GPT-4o-mini is used only for optional natural-language explanations. When enabled and configured with an API key, it can generate concise text to help reviewers understand why a claim may require investigation.

The feature is disabled by default. An explanation supplements the model output and does not establish that fraud has occurred.

### Estimated API Cost

The project's estimates assume approximately 300 input tokens and 60 output tokens per explanation.

| Metric | Estimated Value |
|---|---:|
| Input tokens per explanation | 300 |
| Output tokens per explanation | 60 |
| Estimated cost per explanation | $0.00008 |
| Cost for 40 explanations | $0.0032 |
| Estimated monthly claim volume | Approximately 750 |
| Cost for 750 explanations | $0.06 |

The monthly volume estimate is based on 2,252 claims over three months:

**2,252 ÷ 3 ≈ 751 claims per month.**

These are estimates based on the project's stated assumptions, not guaranteed current provider prices. Actual charges depend on applicable model pricing and real token usage.

### Recorded Test Usage

| Metric | Recorded Result |
|---|---:|
| Tokens recorded during testing | 134 |
| Recorded API spend | $0.00 |

The recorded test reported no spend. This does not guarantee that future API calls will be free.

### How I Reduced AI Usage

- Used local machine-learning models for core fraud predictions.
- Kept GPT-4o-mini explanations optional and disabled by default.
- Sent only essential claim details to the language model.
- Kept generated explanations concise.
- Reused cached explanations when a matching cached result was available.
- Included usage tracking for tokens, requests and estimated costs, where implemented.

### Overall Cost Assessment

The core data-processing, training and prediction pipeline can run without a paid generative AI API key.

Under the stated estimate, generating explanations for 750 claims would cost approximately **$0.06**. This is an estimated API cost only and excludes hosting, infrastructure and development expenses.

---

<div align="center">

## ✦ LIMITATIONS & RESPONSIBLE USE

</div>

### 1. Limited Validation Data

The June 2026 validation set contained 720 claims and only 22 labelled fraud cases. The observed detection rate and estimated financial savings may change as more investigated claims become available.

### 2. Changing Fraud Patterns

A model trained on earlier claims may not recognise new fraud patterns equally well. Regular monitoring and retraining are important.

### 3. New Partners

Newly onboarded partners may have insufficient historical data for reliable predictions. Risk estimates should be interpreted cautiously when historical evidence is limited.

### 4. Legacy Labels

Some older Zoho records may contain claims labelled non-fraud even when the investigation outcome was uncertain. This can introduce noise into training data and affect model performance.

### 5. False Positives

The model incorrectly flagged 24 genuine claims in the June validation sample. Incorrectly holding genuine claims can affect customer goodwill and should be considered when selecting a review threshold.

### 6. Human Verification

A HIGH-risk score is a signal for further investigation, not proof of fraud. Claims should not be rejected automatically based only on a model prediction.

### 7. Modelling Assumption

The model assumes a 45-day delay before investigation outcomes become available. This is a modelling assumption, not a confirmed Kestrel Home policy.

### 8. Deployment and Data Availability

The dashboard requires the model, dependencies and configured application resources to be available in the deployment environment. Private claim data and API keys must remain protected.

### 9. Cost Estimates

The optional AI cost figures are estimates. Actual API charges can change with provider pricing, token consumption and configuration.

### Production Recommendations

- Retrain and validate the model regularly using newly investigated claims.
- Monitor fraud detection, false positives and net financial savings.
- Review unusual outlet patterns using independent evidence.
- Protect private claim records and API credentials.
- Keep human investigators responsible for final claim decisions.

---

<div align="center">

## ✦ CONCLUSION

</div>

The project delivers an end-to-end warranty claim review workflow combining data preprocessing, machine learning, validation, API scoring and an interactive dashboard.

### Key Achievements

- Processed 12,029 raw rows into 11,348 unique claims after cleaning and deduplication.
- Identified an observed fraud-rate increase from 1.08% to 2.53% after 1 May 2026.
- Analysed partner and outlet patterns to identify areas for further investigation.
- Trained a combined Logistic Regression and Gradient Boosting solution.
- Detected 14 out of 22 labelled fraud cases in the June validation sample.
- Selected 38 claims for review during validation.
- Generated predictions for 2,252 test claims.
- Passed all seven recorded submission-format checks.
- Developed a FastAPI scoring service and Streamlit dashboard.
- Kept the core machine-learning pipeline independent of paid generative AI API calls.

### Final Recommendation

Kestrel Home should use the model to support investigation prioritisation rather than as an automatic fraud-decision system.

Recommended actions:

1. Prioritise the 40 highest-risk claims for monthly investigation, subject to review capacity and operational checks.
2. Investigate the five outlets showing unusual fraud patterns without treating those patterns as proof of wrongdoing.
3. Review the 27 higher-value claims reportedly paid without inspection after 1 May 2026.
4. Improve historical investigation labels and partner data quality.
5. Track fraud detected, false positives, customer goodwill costs and net savings per claim reviewed.
6. Retrain and validate the model regularly as new investigation outcomes become available.

The most useful success measures are **fraud detected, investigation effectiveness and net money saved per claim reviewed**, rather than overall accuracy alone.

---

<div align="center">

## ✦ PROJECT LINKS

<br/>

<a href="https://xugptpjrimtbus572dwu7m.streamlit.app/">
<img src="https://img.shields.io/badge/Launch%20Dashboard-0D9488?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch Dashboard"/>
</a>

<a href="https://github.com/rskarthika2005-ai/Kestrel-Home-Warranty-Claim-Review-Variant-C-">
<img src="https://img.shields.io/badge/View%20GitHub%20Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="View GitHub Repository"/>
</a>

<br/><br/>

**Three-Minute Screen Recording:** Add your actual recording URL here.

<br/>

**Developed by Karthika S**  
Email: rskarthika08@gmail.com

</div>
```
