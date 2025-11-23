# %% Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import happybase
import pandas as pd
import numpy as np

# %% Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("HBase Credit Score Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# %% Step 2: Connect to HBase and read table
connection = happybase.Connection('master')  # replace with your HBase master host
table = connection.table('final')

rows = []
for key, data in table.scan():
    row = {k.decode().split(':')[-1]: v.decode() for k, v in data.items()}
    row['ID'] = key.decode()
    rows.append(row)

df = pd.DataFrame(rows)

# %% Step 3: Data cleaning

# Replace blank Credit_Score with "unknown"
df['Credit_Score'] = df['Credit_Score'].astype(str).replace('', 'unknown')

# Convert numeric columns safely
for col_name in df.columns:
    if col_name != 'Credit_Score' and col_name != 'ID':
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)

# Ensure non-numeric columns are strings
for col_name in df.columns:
    if col_name not in ['ID'] + df.select_dtypes(include=[np.number]).columns.tolist():
        df[col_name] = df[col_name].fillna('unknown').astype(str)

# %% Step 4: Identify categorical columns
categorical_cols = [c for c in df.columns if c not in ['ID'] + df.select_dtypes(include=[np.number]).columns.tolist()]

# %% Step 5: Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# %% Step 6: Encode categorical columns
indexers = []
for cat_col in categorical_cols:
    indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
    spark_df = indexer.fit(spark_df).transform(spark_df)
    indexers.append(cat_col + "_idx")

# %% Step 7: Assemble features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = numeric_cols + indexers
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
spark_df = assembler.transform(spark_df)

# %% Step 8: Prepare label
label_indexer = StringIndexer(inputCol='Credit_Score', outputCol='label', handleInvalid='keep')
label_model = label_indexer.fit(spark_df)
spark_df = label_model.transform(spark_df)

# Print label mapping
print("Label classes mapping:", list(label_model.labels))

# %% Step 9: Split train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Step 10: Train Random Forest
rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                            numTrees=50, maxDepth=5, seed=42)
model = rf.fit(train_df)

# %% Step 11: Make predictions
predictions = model.transform(test_df)
predictions.select('ID', 'Credit_Score', 'prediction').show(10)

# %% Step 12: Evaluate model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
weighted_precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
weighted_recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")

# %% Step 13: Write predictions back to HBase in batches
predictions_pd = predictions.select('ID', 'prediction').toPandas()
batch_size = 100  # adjust batch size if needed
for i in range(0, len(predictions_pd), batch_size):
    batch_rows = predictions_pd.iloc[i:i+batch_size]
    with table.batch() as b:
        for _, row in batch_rows.iterrows():
            b.put(str(row['ID']), {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()})

print("Predictions written back to HBase successfully.")

# %% Step 14: Stop Spark and HBase connection
spark.stop()
connection.close()
