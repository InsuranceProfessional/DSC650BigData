# %% Imports
import happybase
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# %% HBase Connection
connection = happybase.Connection('master')  # replace with your HBase master host
table = connection.table('final')

# %% Load data from HBase into Pandas DataFrame
rows = table.scan()  # get all rows
data = []
for key, value in rows:
    row = {k.decode().split(":")[1]: v.decode() for k, v in value.items()}
    row['HBASE_ROW_KEY'] = key.decode()
    data.append(row)

df = pd.DataFrame(data)

# %% Convert numeric columns to proper types
numeric_cols = [
    'Age', 'Annual_Income', 'Monthly_In_hand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
    'Median_Occupation_Income', 'Income_vs_Occupation_Median', 'Investment_to_Salary_Ratio'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# %% Handle categorical columns
categorical_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score',
    'Occ_Accountant', 'Occ_Architect', 'Occ_Developer', 'Occ_Doctor', 'Occ_Engineer',
    'Occ_Entrepreneur', 'Occ_Journalist', 'Occ_Lawyer', 'Occ_Manager', 'Occ_Mechanic',
    'Occ_Media_Manager', 'Occ_Musician', 'Occ_Scientist', 'Occ_Teacher', 'Occ_Writer', 'Occ________'
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# %% Drop rows with missing values
df = df.dropna()

# %% Start Spark session
spark = SparkSession.builder.appName("HBase_RF").getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# %% Encode categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx").fit(spark_df) for col in categorical_cols]
for indexer in indexers:
    spark_df = indexer.transform(spark_df)

# %% Define features and label
feature_cols = [c for c in spark_df.columns if c not in ['Credit_Score', 'HBASE_ROW_KEY'] + categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
spark_df = assembler.transform(spark_df)

label_indexer = StringIndexer(inputCol="Credit_Score", outputCol="label").fit(spark_df)
spark_df = label_indexer.transform(spark_df)

# %% Train-test split
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Train Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=310, maxDepth=22, seed=42)
model = rf.fit(train_df)

# %% Predictions
predictions = model.transform(test_df)
pred_labels = predictions.select("HBASE_ROW_KEY", "prediction").collect()

# %% Evaluation
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# %% Write predictions back to HBase
for row in pred_labels:
    table.put(
        row['HBASE_ROW_KEY'],
        {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()}
    )

print("Predictions written back to HBase table 'final'.")
spark.stop()
