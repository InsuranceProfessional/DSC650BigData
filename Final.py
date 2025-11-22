# %% Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# %% Spark session
spark = SparkSession.builder \
    .appName("HBase Random Forest") \
    .getOrCreate()

# %% HBase configuration
catalog = {
    "table": {"namespace": "default", "name": "final"},
    "rowkey": "ID",
    "columns": {
        "ID": {"cf": "rowkey", "col": "ID", "type": "string"},
        "Customer_ID": {"cf": "cf", "col": "Customer_ID", "type": "string"},
        "Month": {"cf": "cf", "col": "Month", "type": "string"},
        "Name": {"cf": "cf", "col": "Name", "type": "double"},
        "Age": {"cf": "cf", "col": "Age", "type": "int"},
        "SSN": {"cf": "cf", "col": "SSN", "type": "int"},
        "Occupation": {"cf": "cf", "col": "Occupation", "type": "string"},
        "Annual_Income": {"cf": "cf", "col": "Annual_Income", "type": "double"},
        "Monthly_In_hand_Salary": {"cf": "cf", "col": "Monthly_In_hand_Salary", "type": "double"},
        "Num_Bank_Accounts": {"cf": "cf", "col": "Num_Bank_Accounts", "type": "int"},
        "Num_Credit_Card": {"cf": "cf", "col": "Num_Credit_Card", "type": "int"},
        "Interest_Rate": {"cf": "cf", "col": "Interest_Rate", "type": "int"},
        "Num_of_Loan": {"cf": "cf", "col": "Num_of_Loan", "type": "int"},
        "Type_of_Loan": {"cf": "cf", "col": "Type_of_Loan", "type": "string"},
        "Delay_from_due_date": {"cf": "cf", "col": "Delay_from_due_date", "type": "int"},
        "Num_of_Delayed_Payment": {"cf": "cf", "col": "Num_of_Delayed_Payment", "type": "int"},
        "Changed_Credit_Limit": {"cf": "cf", "col": "Changed_Credit_Limit", "type": "double"},
        "Num_Credit_Inquiries": {"cf": "cf", "col": "Num_Credit_Inquiries", "type": "int"},
        "Credit_Mix": {"cf": "cf", "col": "Credit_Mix", "type": "string"},
        "Outstanding_Debt": {"cf": "cf", "col": "Outstanding_Debt", "type": "double"},
        "Credit_Utilization_Ratio": {"cf": "cf", "col": "Credit_Utilization_Ratio", "type": "double"},
        "Credit_History_Age": {"cf": "cf", "col": "Credit_History_Age", "type": "string"},
        "Payment_of_Min_Amount": {"cf": "cf", "col": "Payment_of_Min_Amount", "type": "string"},
        "Total_EMI_per_month": {"cf": "cf", "col": "Total_EMI_per_month", "type": "double"},
        "Amount_invested_monthly": {"cf": "cf", "col": "Amount_invested_monthly", "type": "double"},
        "Payment_Behaviour": {"cf": "cf", "col": "Payment_Behaviour", "type": "string"},
        "Monthly_Balance": {"cf": "cf", "col": "Monthly_Balance", "type": "double"},
        "Credit_Score": {"cf": "cf", "col": "Credit_Score", "type": "string"}
    }
}

# %% Load data from HBase
df = spark.read \
    .options(catalog=str(catalog)) \
    .format("org.apache.spark.sql.execution.datasources.hbase") \
    .load()

# %% Drop rows with missing values
df = df.dropna()

# %% Columns to encode
categorical_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score'
]

# %% Create StringIndexers
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep") for col in categorical_cols]

# %% Features
feature_cols = [c for c in df.columns if c != "Credit_Score"]
# Replace categorical columns with their index columns
feature_cols = [c + "_idx" if c in categorical_cols else c for c in feature_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# %% Random Forest
rf = RandomForestClassifier(
    labelCol="Credit_Score_idx",
    featuresCol="features",
    numTrees=310,
    maxDepth=22,
    minInstancesPerNode=3,
    seed=42
)

# %% Build pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# %% Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# %% Train model
model = pipeline.fit(train_df)

# %% Predictions
predictions = model.transform(test_df)

# %% Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol="Credit_Score_idx",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")

# %% Write predictions back to HBase (add a column 'Prediction')
from pyspark.sql.functions import col, expr

predictions_to_write = predictions.select(
    col("ID"),
    col("prediction").alias("Predicted_Credit_Score")
)

# Save back to HBase (adjust cf if needed)
hbase_write_catalog = {
    "table": {"namespace": "default", "name": "final"},
    "rowkey": "ID",
    "columns": {
        "ID": {"cf": "rowkey", "col": "ID", "type": "string"},
        "Predicted_Credit_Score": {"cf": "cf", "col": "Predicted_Credit_Score", "type": "double"}
    }
}

predictions_to_write.write \
    .options(catalog=str(hbase_write_catalog)) \
    .format("org.apache.spark.sql.execution.datasources.hbase") \
    .save()

spark.stop()
