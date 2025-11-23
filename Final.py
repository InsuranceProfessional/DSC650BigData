# %% Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
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

# %% Step 3: Clean data
for col_name in df.columns:
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)
object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
for col_name in object_cols:
    df[col_name] = df[col_name].fillna('').astype(str)

# Fix Credit_Score: blanks -> "unknown"
df['Credit_Score'] = df['Credit_Score'].replace('', 'unknown')

# %% Step 4: Identify categorical columns
categorical_cols = [c for c in object_cols if c != 'ID' and c != 'Credit_Score']

# %% Step 5: Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# %% Step 6: Encode categorical columns
indexers = []
for cat_col in categorical_cols:
    indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
    spark_df = indexer.fit(spark_df).transform(spark_df)
    indexers.append(cat_col + "_idx")

# %% Step 7: Encode label (Credit_Score)
label_indexer = StringIndexer(inputCol='Credit_Score', outputCol='label', handleInvalid='keep')
label_model = label_indexer.fit(spark_df)
spark_df = label_model.transform(spark_df)

# Save mapping to convert back later
label_mapping = dict(enumerate(label_model.labels))
print("Label mapping:", label_mapping)

# %% Step 8: Assemble features
feature_cols = [c for c in numeric_cols if c != 'ID'] + indexers
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
spark_df = assembler.transform(spark_df)

# %% Step 9: Split train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Step 10: Train Random Forest Classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                            numTrees=5, maxDepth=3, seed=42)
model = rf.fit(train_df)

# %% Step 11: Make predictions
predictions = model.transform(test_df)
predictions.select('ID', 'Credit_Score', 'prediction').show(10)

# %% Step 12: Evaluate performance
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print(f"\n=== Model Performance ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: Confusion matrix
pred_pd = predictions.select('label', 'prediction').toPandas()
conf_matrix = pd.crosstab(pred_pd['label'], pred_pd['prediction'], rownames=['Actual'], colnames=['Predicted'], dropna=False)
print("\nConfusion Matrix:")
print(conf_matrix)

# %% Step 13: Write predictions back to HBase in batches
predictions_pd = predictions.select('ID', 'prediction').toPandas()
batch_size = 500
with table.batch(batch_size=batch_size) as b:
    for i, row in enumerate(predictions_pd.itertuples()):
        # Convert integer prediction back to original string label
        pred_label = label_mapping[int(row.prediction)]
        b.put(str(row.ID), {b'cf:Predicted_Credit_Score': pred_label.encode()})

print("Predictions written back to HBase successfully.")

# %% Step 14: Stop Spark and HBase connection
spark.stop()
connection.close()
