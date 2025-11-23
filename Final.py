from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase
import pandas as pd

# -----------------------------
# Step 1: Spark session
# -----------------------------
spark = SparkSession.builder.appName("MLlib HBase GradesML Prediction").getOrCreate()

# -----------------------------
# Step 2: Connect to HBase
# -----------------------------
connection = happybase.Connection('master')  # replace with your HBase master host
table = connection.table('final')  # your HBase table name

# -----------------------------
# Step 3: Scan HBase table
# -----------------------------
rows = table.scan()
data = []

for key, value in rows:
    row = {k.decode().split(":")[1]: v.decode() for k, v in value.items()}  # parse column family
    row['rowkey'] = key.decode()
    data.append(row)

if not data:
    raise ValueError("No data found in HBase table!")

# -----------------------------
# Step 4: Convert to Spark DataFrame
# -----------------------------
df = pd.DataFrame(data)

# Convert numeric columns to proper types
numeric_cols = [
    'test1', 'test2', 'test3', 'test4', 'final_score'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing numeric values
df.dropna(subset=numeric_cols, inplace=True)

spark_df = spark.createDataFrame(df)

# -----------------------------
# Step 5: Prepare features for ML
# -----------------------------
feature_cols = ['test1', 'test2', 'test3', 'test4']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_df = assembler.transform(spark_df)

# -----------------------------
# Step 6: Train Linear Regression model
# -----------------------------
lr = LinearRegression(featuresCol="features", labelCol="final_score")
lr_model = lr.fit(final_df)

# -----------------------------
# Step 7: Make predictions
# -----------------------------
predictions = lr_model.transform(final_df)
predictions.show(10)

# -----------------------------
# Step 8: Write predictions back to HBase
# -----------------------------
hbase_table = connection.table('final')  # writing back to the same table

for row in predictions.collect():
    hbase_table.put(
        row['rowkey'],
        {b'cf:predicted_final_score': str(row['prediction']).encode()}
    )

print("Predictions successfully written to HBase!")

# -----------------------------
# Step 9: Stop Spark session
# -----------------------------
spark.stop()
