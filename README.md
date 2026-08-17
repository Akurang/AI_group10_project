# Group 10: Customer Churn E-commerce

Ewurakua Amoah 74492028
Eyram Fenu 31122028
Gabriel Akurang 35882028
Nana Asantewaa Osei Bonsu 11782028


A Streamlit app that predicts customer churn and generates AI-powered retention strategies.

```
Customer Data -> Random Forest -> Churn Probability -> Explanation -> LLM -> Retention Strategy
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What you need in place

| File | Purpose |
|---|---|
| `models/randomforest_fair.pkl` | Trained Random Forest |
| `models/scaler.pkl` | Fitted StandardScaler (must match training) |
| `data/Test Data.xlsx` | Customer dataset (sheet name: `E Comm`) |



To do: set `GROQ_API_KEY` (env var or `.env` file) to enable real AI-generated retention strategies. 

## Pages

- **Dashboard** — overall churn stats and highest-risk customers
- **Customer Analysis** — pick an existing customer, or enter one manually, to see their churn risk and what's driving it
- **AI Retention Strategy** — generates recommended retention actions for the analyzed customer
- **About** — explains the model, threshold, and known limitations

## Project structure

```
app.py
models/
  randomforest_fair.pkl
  randomforest.pkl
  scaler.pkl
data/
  E Commerce Dataset.xlsx
  Test Data.xlsx
src/
  preprocessing.py   # feature engineering + encoding
  prediction.py       # model loading + prediction (0.39 threshold)
  explanation.py      # why a customer is flagged as at-risk
  llm.py               # retention strategy generation via Groq/Llama

requirements.txt
final_project.ipynb #Analysis the data and training models
driver.ipynb  #conduct test in the terminal before app build
requirements.txt

```

