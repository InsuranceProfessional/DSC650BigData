# %% Imports
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import happybase
import pandas as pd
import numpy as np
from pyspark.sql.functions import col

# %% Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("HBase Simple Linear Regression") \
    .getOrCreate()

# %% Step 2: Connect to HBase and read table
connection = happybase.Connection('master')
table = connection.table('final')

rows = []
for key, data in table.scan():
    row = {}
    for k, v in data.items():
        col_name = k.decode().split(':')[-1]
        try:
            # Convert to float if possible
            row[col_name] = float(v)
        except:
            # If it fails (e.g., string), store as string
            row[col_name] = v.decode()
    row['ID'] = key.decode()
    rows.append(row)


df = pd.DataFrame(rows)
df = df.head(500)  # Only 100 rows for testing

# %% Step 3: Clean numeric data
# Convert to numeric and handle missing
for col_name in ['Annual_Income', 'Monthly_In_hand_Salary']:
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)

# %% Step 4: Convert to Spark DataFrame
spark_df = spark.createDataFrame(df[['ID', 'Annual_Income', 'Monthly_In_hand_Salary']])

# %% Step 5: Assemble features
assembler = VectorAssembler(inputCols=['Annual_Income'], outputCol='features')
spark_df = assembler.transform(spark_df)

# %% Step 6: Split train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Step 7: Train Linear Regression
lr = LinearRegression(featuresCol='features', labelCol='Monthly_In_hand_Salary')
model = lr.fit(train_df)

# %% Step 8: Make predictions
predictions = model.transform(test_df)

predictions.select('ID', 'Annual_Income', 'Monthly_In_hand_Salary', 'prediction').show(10)

# %% Step 9: Show performance metrics
training_summary = model.summary
print("RMSE:", training_summary.rootMeanSquaredError)
print("R2:", training_summary.r2)

# %% Step 10: Write predictions back to HBase
predictions_pd = predictions.select('ID', 'prediction').toPandas()
with table.batch(batch_size=500) as b:
    for i, row in predictions_pd.iterrows():
        b.put(str(row.ID), {b'cf:Predicted_Monthly_Salary': str(row.prediction).encode()})

print("Predictions written back to HBase successfully.")

# %% Step 11: Stop Spark and HBase connection
spark.stop()
connection.close()
