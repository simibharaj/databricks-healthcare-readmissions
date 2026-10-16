# Hospital Readmission Risk Pipeline on Databricks

An end-to-end data and ML project built on **Databricks Free Edition**, using a healthcare scenario.

**All data is synthetic (randomly generated). No real patient data is used.**

## What it does

1. **Bronze**: generates 20,000 synthetic hospital encounters (age, diagnosis, payer, length of stay, prior admissions, medications, follow-up at discharge, 30-day readmission) and saves them as a Delta table.
2. **Silver**: cleans and de-duplicates the data, adds age bands and a polypharmacy flag.
3. **Gold**: readmission rate by diagnosis and age band, plus a SQL check on whether a follow-up appointment at discharge lowers readmissions.
4. **Model**: logistic regression (scikit-learn) that predicts 30-day readmission, evaluated with ROC AUC.
5. **Scoring**: scores every encounter and saves a `gold_patient_risk` table with low, medium and high risk tiers.

## Stack

PySpark, Delta Lake, Databricks SQL, Unity Catalog, serverless compute, pandas, scikit-learn.

## Run it

1. Sign up for [Databricks Free Edition](https://www.databricks.com/learn/free-edition).
2. In your workspace: Workspace, three-dot menu, Import, upload `readmission_risk_pipeline.py`.
3. Attach to serverless compute and click **Run all**.

Tables are created in `workspace.readmissions_demo`.

## Ideas for next steps

- Schedule the notebook as a Databricks Job
- Add a Genie space or AI/BI dashboard on the gold tables
- Track experiments with MLflow
