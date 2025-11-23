from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("CreditScorePrediction") \
    .getOrCreate()

# 2. Load your data
df = spark.read.csv("your_data.csv", header=True, inferSchema=True)

# 3. Convert Boolean columns to string (needed for StringIndexer)
for c, t in df.dtypes:
    if t == 'boolean':
        df = df.withColumn(c, when(col(c) == True, "True").otherwise("False"))

# 4. Convert Credit_Score numeric to categorical
df = df.withColumn(
    "Credit_Score",
    when(col("Credit_Score").isNull(), "Unknown")
    .when(col("Credit_Score") < 300, "Bad")
    .when((col("Credit_Score") >= 300) & (col("Credit_Score") < 700), "Good")
    .otherwise("Great")
)

# 5. Identify categorical columns for indexing
categorical_cols = [c for c, t in df.dtypes if t == 'string' and c != "Credit_Score"]

# Apply StringIndexer to all categorical columns
indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols + ["Credit_Score"]]

for indexer in indexers:
    df = indexer.fit(df).transform(df)

# 6. Prepare feature columns
feature_cols = [c for c in df.columns if c.endswith("_idx") and c != "Credit_Score_idx"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# 7. Prepare final dataset
final_df = df.select(col("features"), col("Credit_Score_idx").alias("label"))

# 8. Split data
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=42)

# 9. Train RandomForestClassifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10)

model = rf.fit(train_df)

# 10. Predictions
predictions = model.transform(test_df)

# 11. Evaluate metrics
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("=== Model Performance ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: show top 10 predictions
predictions.select("label", "prediction").show(10)

# Stop Spark session
spark.stop()
