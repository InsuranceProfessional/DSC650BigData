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
connection = happybase.Connection('master')  # replace with your HBase master host
table = connection.table('final')

rows = []
for key, data in table.scan():
    row = {k.decode().split(':')[-1]: v.decode() for k, v in data.items()}
    row['Customer_ID'] = key.decode()
    rows.append(row)

df = pd.DataFrame(rows)

# %% Step 3: Automatic type detection and cleaning

# Step 3a: Convert numeric columns safely
for col_name in df.columns:
    # Try converting to numeric
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

# Step 3b: Fill NaNs for numeric columns with 0
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)

# Step 3c: Convert non-numeric/object columns to string and fill NaNs
object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
for col_name in object_cols:
    df[col_name] = df[col_name].fillna('0').astype(str)

# %% Step 4: Identify categorical columns automatically
# Any object column except Customer_ID and numeric columns
categorical_cols = [c for c in object_cols if c != 'Customer_ID']

# %% Step 5: Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# %% Step 6: Encode categorical columns
indexers = []
for cat_col in categorical_cols:
    indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
    spark_df = indexer.fit(spark_df).transform(spark_df)
    indexers.append(cat_col + "_idx")

# %% Step 7: Assemble features
feature_cols = numeric_cols + indexers
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
spark_df = assembler.transform(spark_df)

# %% Step 8: Prepare label
label_col = 'Credit_Score'
if label_col in spark_df.columns:
    spark_df = spark_df.fillna({label_col: 0})
    label_indexer = StringIndexer(inputCol=label_col, outputCol='label', handleInvalid='keep')
    spark_df = label_indexer.fit(spark_df).transform(spark_df)
else:
    raise ValueError("Credit_Score column not found!")

# %% Step 9: Split train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Step 10: Train Random Forest
rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                            numTrees=310, maxDepth=22, seed=42)
model = rf.fit(train_df)

# %% Step 11: Make predictions
predictions = model.transform(test_df)
predictions.select('Customer_ID', 'Credit_Score', 'prediction').show(10)

# %% Step 12: Write predictions back to HBase
predictions_pd = predictions.select('Customer_ID', 'prediction').toPandas()
for _, row in predictions_pd.iterrows():
    table.put(str(row['Customer_ID']), {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()})

print("Predictions written back to HBase successfully.")

# %% Step 13: Stop Spark and HBase connection
spark.stop()
connection.close()
