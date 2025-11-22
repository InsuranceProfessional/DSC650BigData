# ============================================
# Champlin Smith - Credit Score Prediction in Spark + HBase
# ============================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import happybase

# -----------------------------
# Step 1: Create Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("CreditScorePrediction") \
    .enableHiveSupport() \
    .getOrCreate()

# -----------------------------
# Step 2: Load the data from Hive table
# -----------------------------
df = spark.sql("SELECT * FROM final")  # Replace 'final' with your Hive table name

# -----------------------------
# Step 3: Preprocessing
# -----------------------------
# Drop unnecessary columns (ID, PII)
df = df.drop("Name", "SSN", "Customer_ID", "ID", "Month")

# Convert numeric columns to double
numeric_cols = ["Age", "Annual_Income", "Num_of_Loan", "Num_Credit_Card",
                "Num_Bank_Accounts", "Num_Credit_Inquiries", "Amount_invested_monthly",
                "Monthly_In_hand_Salary", "Credit_History_Age", "Median_Occupation_Income"]

for col_name in numeric_cols:
    df = df.withColumn(col_name, df[col_name].cast("double"))

# Drop rows with nulls
df = df.na.drop()

# Derived features
df = df.withColumn("Investment_to_Salary_Ratio", col("Amount_invested_monthly") / col("Monthly_In_hand_Salary"))
df = df.withColumn("Income_vs_Occ_Median", col("Annual_Income") - col("Median_Occupation_Income"))

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
                "Investment_to_Salary_Ratio", "Income_vs_Occ_Median"] + \
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
# Step 10: Evaluate model
# -----------------------------
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1_score:.4f}")

# Optional: confusion matrix
predictions.groupBy("label", "prediction").count().show()

# -----------------------------
# Step 11: Write predictions per row to HBase robustly
# -----------------------------
def write_to_hbase_partition(rows):
    import happybase
    try:
        connection = happybase.Connection('172.28.1.1')  # Master IP
        connection.open()
        table = connection.table('final')  # HBase table
        for row in rows:
            try:
                # Fallback row key if id is null
                row_key = str(row.id) if row.id else str(row.prediction) + "_" + str(int(row.prediction))
                table.put(row_key, {
                    b'cf:predicted_credit_score': str(int(row.prediction)).encode('utf-8')
                })
            except Exception as e_row:
                print(f"Row write failed: {row}, error: {e_row}")
        connection.close()
    except Exception as e_conn:
        print(f"HBase connection failed: {e_conn}")

# Select only id and prediction
hbase_df = predictions.select("id", "prediction")

# Convert to RDD and write
hbase_df.rdd.foreachPartition(write_to_hbase_partition)

# -----------------------------
# Step 12: Write overall metrics to HBase
# -----------------------------
metrics_data = [
    ('metrics', 'cf:accuracy', str(accuracy)),
    ('metrics', 'cf:f1', str(f1_score))
]

def write_metrics_partition(rows):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('final')
    for row_key, col, value in rows:
        table.put(row_key, {col.encode('utf-8'): value.encode('utf-8')})
    connection.close()

spark.sparkContext.parallelize(metrics_data).foreachPartition(write_metrics_partition)


# -----------------------------
# Step 14: Stop Spark session
# -----------------------------
spark.stop()
