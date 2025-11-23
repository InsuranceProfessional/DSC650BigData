# %% Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
import happybase
import pandas as pd
import numpy as np

# %% Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("HBase Credit Score Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# %% Step 2: Connect to HBase and read table
connection = happybase.Connection('master')
table = connection.table('final')

rows = []
for key, data in table.scan():
    row = {k.decode().split(':')[-1]: v.decode() for k, v in data.items()}
    row['ID'] = key.decode()
    rows.append(row)

df = pd.DataFrame(rows)
df = df.head(100)  # for testing; remove or adjust for full dataset

# %% Step 3: Clean data
# Convert numeric columns safely (exclude ID and Credit_Score)
for col_name in df.columns:
    if col_name not in ['ID', 'Credit_Score']:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

# Fill numeric NaNs with 0
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)

# Fill object columns and convert to string
object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
for col_name in object_cols:
    df[col_name] = df[col_name].fillna('').astype(str)

# Fix Credit_Score: blanks or NaN -> "unknown"
df['Credit_Score'] = df['Credit_Score'].replace(['', np.nan], 'unknown')

# %% Step 4: Identify categorical columns
categorical_cols = [c for c in object_cols if c not in ['ID', 'Credit_Score']]

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
labels = label_model.labels  # Original string labels

# %% Step 8: Assemble features (exclude ID and label)
feature_cols = [c for c in numeric_cols if c != 'ID'] + indexers
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
spark_df = assembler.transform(spark_df)

# %% Step 9: Split train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Step 10: Train Random Forest Classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                            numTrees=10, maxDepth=5, seed=42)
model = rf.fit(train_df)

# %% Step 11: Make predictions
predictions = model.transform(test_df)

# %% Step 12: Map numeric prediction back to original credit score strings
def map_prediction(pred):
    return labels[int(pred)]

# Drop old column if exists
if 'Predicted_Credit_Score' in predictions.columns:
    predictions = predictions.drop('Predicted_Credit_Score')

# Map numeric prediction to string
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

map_udf = udf(lambda x: labels[int(x)], StringType())
predictions = predictions.withColumn('Predicted_Credit_Score', map_udf(col('prediction')))

# %% Step 13: Write predictions back to HBase using foreachPartition
def write_partition_to_hbase(partition):
    import happybase  # must import inside function for workers
    conn = happybase.Connection('master')
    tbl = conn.table('final')
    with tbl.batch(batch_size=500) as b:
        for row in partition:
            b.put(str(row.ID), {b'cf:Predicted_Credit_Score': map_prediction(row.prediction).encode()})
    conn.close()

predictions.select('ID', 'prediction').rdd.foreachPartition(write_partition_to_hbase)

print("Predictions written back to HBase successfully.")

# %% Step 14: Stop Spark and HBase connection
spark.stop()
connection.close()
