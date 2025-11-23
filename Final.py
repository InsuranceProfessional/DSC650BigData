# %% Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
import happybase
import pandas as pd

# %% Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("HBase Credit Score Prediction") \
    .getOrCreate()

# %% Step 2: Connect to HBase and read table
connection = happybase.Connection('master')  # replace 'master' with your HBase host
table = connection.table('final')  # HBase table name

# Fetch all rows from HBase
rows = table.scan()
data = []
for key, row in rows:
    row_dict = {}
    for k, v in row.items():
        col_name = k.decode().replace("cf:", "")  # remove cf: prefix
        row_dict[col_name] = v.decode() if v is not None else None
    row_dict['ID'] = key.decode()
    data.append(row_dict)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Rename columns to remove spaces
df.rename(columns={
    'Median Occupation Income': 'Median_Occupation_Income',
    'Income vs Occupation Median': 'Income_vs_Occupation_Median'
}, inplace=True)

# Force all columns to string initially
df = df.astype(str)

# %% Step 3: Convert numeric columns
numeric_cols = [
    'Age', 'Annual_Income', 'Monthly_In_hand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Monthly_Balance', 'Median_Occupation_Income',
    'Income_vs_Occupation_Median', 'Investment_to_Salary_Ratio'
]

for col_name in numeric_cols:
    if col_name in df.columns:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

# %% Step 4: Encode categorical columns
categorical_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour',
    'Occ_Accountant', 'Occ_Architect', 'Occ_Developer', 'Occ_Doctor', 'Occ_Engineer',
    'Occ_Entrepreneur', 'Occ_Journalist', 'Occ_Lawyer', 'Occ_Manager',
    'Occ_Mechanic', 'Occ_Media_Manager', 'Occ_Musician', 'Occ_Scientist',
    'Occ_Teacher', 'Occ_Writer'
]

# Ensure categorical columns are strings
for cat_col in categorical_cols:
    if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype(str)

# %% Step 5: Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Apply StringIndexer to categorical columns
indexers = []
for cat_col in categorical_cols:
    if cat_col in spark_df.columns:
        indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
        spark_df = indexer.fit(spark_df).transform(spark_df)
        indexers.append(cat_col + "_idx")

# %% Step 6: Prepare feature vector
feature_cols = [c for c in numeric_cols if c in spark_df.columns] + indexers
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
spark_df = assembler.transform(spark_df)

# %% Step 7: Prepare label column
label_col = "Credit_Score"
if label_col in spark_df.columns:
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep")
    spark_df = label_indexer.fit(spark_df).transform(spark_df)
else:
    raise ValueError("Credit_Score column not found in HBase table!")

# %% Step 8: Split into train/test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Step 9: Train Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                            numTrees=310, maxDepth=22, seed=42)
model = rf.fit(train_df)

# %% Step 10: Make predictions
predictions = model.transform(test_df)
predictions.select("ID", "Credit_Score", "prediction").show(10)

# %% Step 11: Write predictions back to HBase
predictions_pd = predictions.select("ID", "prediction").toPandas()

for _, row in predictions_pd.iterrows():
    table.put(str(row['ID']),
              {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()})

print("Predictions written back to HBase successfully.")

spark.stop()
connection.close()
