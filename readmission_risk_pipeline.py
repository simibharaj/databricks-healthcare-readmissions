# Databricks notebook source
# MAGIC %md
# MAGIC # Hospital Readmission Risk Pipeline (synthetic data)
# MAGIC Built on Databricks Free Edition. Medallion pipeline (bronze, silver, gold) in Delta tables,
# MAGIC SQL analytics, and a simple readmission risk model.
# MAGIC
# MAGIC **All data is randomly generated. No real patient data.**

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "readmissions_demo"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze: generate raw synthetic encounters

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 20000

diagnoses = ["Heart failure", "COPD", "Pneumonia", "Diabetes", "Hip fracture", "Sepsis"]
payers = ["Medicare", "Medicaid", "Commercial", "Self-pay"]

pdf = pd.DataFrame({
    "encounter_id": np.arange(1, n + 1),
    "age": rng.integers(18, 95, n),
    "primary_dx": rng.choice(diagnoses, n),
    "payer": rng.choice(payers, n, p=[0.45, 0.2, 0.3, 0.05]),
    "length_of_stay": rng.integers(1, 15, n),
    "num_prior_admissions": rng.poisson(1.2, n),
    "num_medications": rng.integers(0, 25, n),
    "discharged_with_followup": rng.choice([0, 1], n, p=[0.35, 0.65]),
})

# Hidden rule that creates the signal for the label
logit = (
    -3.0
    + 0.02 * pdf["age"]
    + 0.25 * pdf["num_prior_admissions"]
    + 0.04 * pdf["num_medications"]
    + 0.05 * pdf["length_of_stay"]
    - 0.8 * pdf["discharged_with_followup"]
    + np.where(pdf["primary_dx"].isin(["Heart failure", "COPD", "Sepsis"]), 0.5, 0)
)
prob = 1 / (1 + np.exp(-logit))
pdf["readmitted_30d"] = (rng.random(n) < prob).astype(int)

bronze = spark.createDataFrame(pdf)
bronze.write.mode("overwrite").saveAsTable("bronze_encounters")
print("bronze rows:", bronze.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver: clean and add features

# COMMAND ----------

from pyspark.sql import functions as F

silver = (
    spark.table("bronze_encounters")
    .dropDuplicates(["encounter_id"])
    .filter((F.col("age") >= 18) & (F.col("length_of_stay") > 0))
    .withColumn("age_band", F.when(F.col("age") < 45, "18-44")
                              .when(F.col("age") < 65, "45-64")
                              .when(F.col("age") < 80, "65-79")
                              .otherwise("80+"))
    .withColumn("polypharmacy", (F.col("num_medications") >= 10).cast("int"))
)
silver.write.mode("overwrite").saveAsTable("silver_encounters")
print("silver rows:", silver.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gold: readmission rate by diagnosis and age band

# COMMAND ----------

gold = (
    spark.table("silver_encounters")
    .groupBy("primary_dx", "age_band")
    .agg(
        F.count("*").alias("encounters"),
        F.round(F.avg("readmitted_30d") * 100, 1).alias("readmit_rate_pct"),
        F.round(F.avg("length_of_stay"), 1).alias("avg_los"),
    )
)
gold.write.mode("overwrite").saveAsTable("gold_readmission_summary")
display(gold.orderBy(F.desc("readmit_rate_pct")))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Does a follow-up appointment at discharge lower readmissions?
# MAGIC SELECT discharged_with_followup,
# MAGIC        COUNT(*) AS encounters,
# MAGIC        ROUND(AVG(readmitted_30d) * 100, 1) AS readmit_rate_pct
# MAGIC FROM silver_encounters
# MAGIC GROUP BY discharged_with_followup
# MAGIC ORDER BY discharged_with_followup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model: predict 30-day readmission

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = spark.table("silver_encounters").toPandas()
features = ["age", "length_of_stay", "num_prior_admissions", "num_medications",
            "discharged_with_followup", "primary_dx", "payer"]
X = pd.get_dummies(data[features], columns=["primary_dx", "payer"], drop_first=True)
y = data["readmitted_30d"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Test AUC: {auc:.3f}")

# COMMAND ----------

coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
display(coefs.reset_index().rename(columns={"index": "feature", 0: "coefficient"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Score patients and save a risk table

# COMMAND ----------

data["risk_score"] = model.predict_proba(X)[:, 1]
data["risk_tier"] = pd.cut(data["risk_score"], [0, 0.1, 0.25, 1.0], labels=["low", "medium", "high"])
data["risk_tier"] = data["risk_tier"].astype(str)
scored = spark.createDataFrame(data[["encounter_id", "primary_dx", "risk_score", "risk_tier"]])
scored.write.mode("overwrite").saveAsTable("gold_patient_risk")
display(scored.groupBy("risk_tier").count())
