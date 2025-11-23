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

# %% Step 3: Clean and convert types
for col_name in df.columns:
    # Try numeric conversion
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)

object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
for col_name in object_cols:
    df[col_name] = df[col_name].fillna('unknown').astype(str)

# %% Step 4: Fix Credit_Score column
if 'Credit_Score' in df.columns:
    df['Credit_Score'] = df['Credit_Score'].replace({'': 'unknown', ' ': 'unknown', None: 'unknown'})
else:
    raise ValueError("Credit_Score column not found!")

# %% Step 5: Identify categorical columns
# Exclude numeric and ID-like columns
categorical_cols = [c for c in object_cols if c != 'ID' and c != 'Credit_Score']

# %% Step 6: Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# %% Step 7: Encode categorical columns
indexers = []
for cat_col in categorical_cols + ['Credit_Score']:
    indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
    spark_df = indexer.fit(spark_df).transform(spark_df)
    indexers.append(cat_col + "_idx")

# %% Step 8: Assemble features
# Exclude ID from features
high_card_cols = ['ID']
feature_cols = [c for c in numeric_cols if c not in high_card_cols] + \
               [c for c in indexers if not any(h in c for h in high_card_cols)]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
spark_df = assembler.transform(spark_df)

# %% Step 9: Prepare label
label_col_idx = 'Credit_Score_idx'

# %% Step 10: Split train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Step 11: Train Random Forest
rf = RandomForestClassifier(featuresCol='features', labelCol=label_col_idx,
                            numTrees=50, maxDepth=5, seed=42)
model = rf.fit(train_df)

# %% Step 12: Make predictions
predictions = model.transform(test_df)
predictions.select('ID', 'Credit_Score', 'prediction').show(10)

# %% Step 13: Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol=label_col_idx, predictionCol='prediction')
accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
f1 = evaluator.setMetricName("f1").evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# %% Step 14: Write predictions back to HBase in batches
predictions_pd = predictions.select('ID', 'prediction').toPandas()
batch_size = 100
for start in range(0, len(predictions_pd), batch_size):
    batch = predictions_pd.iloc[start:start+batch_size]
    with table.batch() as b:
        for _, row in batch.iterrows():
            b.put(str(row['ID']), {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()})

print("Predictions written back to HBase successfully.")

# %% Step 15: Stop Spark and HBase connection
spark.stop()
connection.close()
