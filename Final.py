# %% Imports
import happybase
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# %% HBase Connection
connection = happybase.Connection('master')  # replace with your HBase master host
table = connection.table('final')

# %% Load data from HBase into Pandas DataFrame
rows = table.scan()
data = []
for key, value in rows:
    row = {k.decode().split(":")[1]: v.decode() for k, v in value.items()}
    row['HBASE_ROW_KEY'] = key.decode()
    data.append(row)

import pandas as pd
df = pd.DataFrame(data)

# %% Drop rows with missing values
df = df.dropna()

# %% Create Spark session
spark = SparkSession.builder.appName("HBase_CreditScore_Prediction").getOrCreate()

# %% Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# %% Identify numeric and categorical columns
numeric_cols = [
    'Age', 'Annual_Income', 'Monthly_In_hand_Salary', 'Num_Bank_Accounts', 
    'Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
    'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
    'Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month',
    'Amount_invested_monthly','Monthly_Balance','Median_Occupation_Income',
    'Income_vs_Occupation_Median','Investment_to_Salary_Ratio'
]

categorical_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour',
    'Occ_Accountant','Occ_Architect','Occ_Developer','Occ_Doctor','Occ_Engineer',
    'Occ_Entrepreneur','Occ_Journalist','Occ_Lawyer','Occ_Manager','Occ_Mechanic',
    'Occ_Media_Manager','Occ_Musician','Occ_Scientist','Occ_Teacher','Occ_Writer'
]

# %% Cast numeric columns
for col in numeric_cols:
    if col in spark_df.columns:
        spark_df = spark_df.withColumn(col, spark_df[col].cast(DoubleType()))

# %% Index categorical columns
indexed_cols = []
for col in categorical_cols:
    if col in spark_df.columns:
        indexer = StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
        spark_df = indexer.fit(spark_df).transform(spark_df)
        indexed_cols.append(col + "_idx")

# %% Index label
label_indexer = StringIndexer(inputCol="Credit_Score", outputCol="label", handleInvalid="keep")
spark_df = label_indexer.fit(spark_df).transform(spark_df)

# %% Assemble features
feature_cols = numeric_cols + indexed_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
spark_df = assembler.transform(spark_df)

# %% Split data
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# %% Train Random Forest classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", 
                            numTrees=310, maxDepth=22, seed=42)
model = rf.fit(train_df)

# %% Make predictions
predictions = model.transform(test_df)

# %% Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")

# %% Write predictions back to HBase
preds_df = predictions.select("HBASE_ROW_KEY", "prediction").toPandas()
for idx, row in preds_df.iterrows():
    table.put(
        row['HBASE_ROW_KEY'],
        {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()}
    )

print("Predictions written back to HBase table 'final'.")
