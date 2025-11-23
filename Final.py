# %% Imports
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import happybase
import pandas as pd
import numpy as np
from pyspark.sql.functions import col

#Create Spark session
spark = SparkSession.builder \
    .appName("HBase Simple Linear Regression") \
    .getOrCreate()

#Connect to HBase and read table
connection = happybase.Connection('master')
table = connection.table('final')

rows = []
for key, data in table.scan():
    row = {k.decode().split(':')[-1]: v.decode() for k, v in data.items()}
    row['ID'] = key.decode()
    rows.append(row)

df = pd.DataFrame(rows)

#Clean numeric data
df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce').fillna(0)
df['Monthly_In_hand_Salary'] = pd.to_numeric(df['Monthly_In_hand_Salary'], errors='coerce').fillna(0)

# Filter rows: only keep Monthly_In_hand_Salary > 100
df = df[df['Monthly_In_hand_Salary'] > 100]

#Convert to Spark DataFrame
spark_df = spark.createDataFrame(df[['ID', 'Annual_Income', 'Monthly_In_hand_Salary']])

#Assemble features
assembler = VectorAssembler(inputCols=['Annual_Income'], outputCol='features')
spark_df = assembler.transform(spark_df)

#Split train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

#Train Linear Regression
lr = LinearRegression(featuresCol='features', labelCol='Monthly_In_hand_Salary')
model = lr.fit(train_df)

#Make predictions
predictions = model.transform(test_df)

predictions.select('ID', 'Annual_Income', 'Monthly_In_hand_Salary', 'prediction').show(10)

#Show performance metrics
training_summary = model.summary
print("RMSE:", training_summary.rootMeanSquaredError)
print("R2:", training_summary.r2)

#Write predictions back to HBase
predictions_pd = predictions.select('ID', 'prediction').toPandas()
with table.batch(batch_size=500) as b:
    for i, row in predictions_pd.iterrows():
        b.put(str(row.ID), {b'cf:Predicted_Monthly_Salary': str(row.prediction).encode()})

print("Predictions written back to HBase successfully.")

#Stop Spark and HBase connection
spark.stop()
connection.close()
