# ============================================
# Champlin Smith - Credit Score Prediction in Spark + HBase (CSV from HDFS)
# ============================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, monotonically_increasing_id
import happybase

# -----------------------------
# Step 1: Create Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("CreditScorePrediction") \
    .getOrCreate()

# -----------------------------
# Step 2: Load data from HDFS CSV
# -----------------------------
df = spark.read.csv("/user/champlin/nifi-data/dataset.csv", header=True, inferSchema=True)

# -----------------------------
# Step 3: Preprocessing
# -----------------------------
# Drop unnecessary / PII columns
drop_cols = ["Name", "SSN", "Customer_ID", "ID", "Month"]
df = df.drop(*drop_cols)

# Convert numeric columns
numeric_cols = ["Age", "Annual_Income", "Num_of_Loan", "Num_Credit_Card",
                "Num_Bank_Accounts", "Num_Credit_Inquiries", "Amount_invested_monthly",
                "Monthly_In_hand_Salary", "Credit_History_Age"]

for col_name in numeric_cols:
    df = df.withColumn(col_name, col(col_name).cast("double"))

# Drop rows with nulls
df = df.na.drop()

# Derived features
df = df.withColumn("Investment_to_Salary_Ratio", col("Amount_invested_monthly") / col("Monthly_In_hand_Salary"))

# -----------------------------
# Step 4: Encode categorical columns
# -----------------------------
categorical_cols = ["Occupation", "Type_of_Loan", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

for cat_col in categorical_cols:
    indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_index").setHandleInvalid("keep")
    df = indexer.fit(df).transform(df)

# -----------------------------
# Step 5: Assemble features
# -----------------------------
feature_cols = ["Age", "Annual_Income", "Num_of_Loan", "Num_Credit_Card",
                "Num_Bank_Accounts", "Num_Credit_Inquiries", "Credit_History_Age",
                "Amount_invested_monthly", "Monthly_In_hand_Salary",
                "Investment_to_Salary_Ratio"] + \
                [f"{cat}_index" for cat in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
df = assembler.transform(df)

# -----------------------------
# Step 6: Index label column
# -----------------------------
label_indexer = StringIndexer(inputCol="Credit_Score", outputCol="label").fit(df)
df = label_indexer.transform(df)

# -----------------------------
# Step 7: Split into train/test
# -----------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# -----------------------------
# Step 8: Train Random Forest
# -----------------------------
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10)
rf_model = rf.fit(train_df)

# -----------------------------
# Step 9: Predict on test data
# -----------------------------
predictions = rf_model.transform(test_df)

# -----------------------------
# Step 10: Evaluate metrics
# -----------------------------
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Save metrics to HDFS
metrics_df = spark.createDataFrame([(accuracy,)], ["accuracy"])
metrics_df.write.mode("overwrite").csv("/user/champlin/credit_score_metrics")

print(f"Test Accuracy: {accuracy:.4f}")

# Optional: confusion matrix
predictions.groupBy("label", "prediction").count().show()

# -----------------------------
# Step 11: Write predictions to HBase
# -----------------------------
def write_to_hbase_partition(rows):
    connection = happybase.Connection('172.28.1.1')  # Replace with HBase master IP
    connection.open()
    table = connection.table('final')  # HBase table with CF 'cf'
    for row in rows:
        row_key = str(row.id) if 'id' in row.asDict() else str(row.label)
        table.put(row_key, {b'cf:predicted_credit_score': str(int(row.prediction)).encode('utf-8')})
    connection.close()

# Add unique ID if missing
if 'id' not in predictions.columns:
    predictions = predictions.withColumn("id", monotonically_increasing_id())

hbase_df = predictions.select("id", "prediction")
hbase_df.rdd.foreachPartition(write_to_hbase_partition)

# -----------------------------
# Step 12: Stop Spark session
# -----------------------------
spark.stop()
