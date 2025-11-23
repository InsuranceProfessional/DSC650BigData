# %% Imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import happybase

# %% Spark session
spark = SparkSession.builder \
    .appName("HBase Credit Score Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# %% HBase connection
connection = happybase.Connection(host='master', port=9090)  # replace with your host/port
table = connection.table('final')

# %% Load data from HBase into a list of dictionaries
rows = table.scan()
data = []
for key, value in rows:
    row = {k.decode().split(":")[1]: v.decode() for k, v in value.items()}
    row['HBASE_ROW_KEY'] = key.decode()
    data.append(row)

if not data:
    raise ValueError("No data retrieved from HBase")

# %% Infer schema automatically (all columns as string first)
spark_df = spark.createDataFrame(data)

# %% Convert numeric columns
numeric_cols = [
    'Age', 'Annual_Income', 'Monthly_In_hand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
    'Median_Occupation_Income', 'Income_vs_Occupation_Median', 'Investment_to_Salary_Ratio'
]

for col in numeric_cols:
    if col in spark_df.columns:
        spark_df = spark_df.withColumn(col, spark_df[col].cast(DoubleType()))

# %% Handle categorical columns
label_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour',
    'Occ_Accountant', 'Occ_Architect', 'Occ_Developer', 'Occ_Doctor', 'Occ_Engineer',
    'Occ_Entrepreneur', 'Occ_Journalist', 'Occ_Lawyer', 'Occ_Manager', 'Occ_Mechanic',
    'Occ_Media_Manager', 'Occ_Musician', 'Occ_Scientist', 'Occ_Teacher', 'Occ_Writer', 'Occ________'
]

for col in label_cols:
    if col in spark_df.columns:
        indexer = StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
        spark_df = indexer.fit(spark_df).transform(spark_df)

# %% Define features and label
feature_cols = [c for c in spark_df.columns if c not in ['Credit_Score', 'HBASE_ROW_KEY']]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_df = assembler.transform(spark_df)

# Encode label
label_indexer = StringIndexer(inputCol="Credit_Score", outputCol="label", handleInvalid="keep")
final_df = label_indexer.fit(final_df).transform(final_df)

# %% Train-test split
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# %% Train Random Forest classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                            numTrees=310, maxDepth=22, maxBins=100, seed=42)
model = rf.fit(train_df)

# %% Predictions
predictions = model.transform(test_df)
predictions.select("HBASE_ROW_KEY", "prediction").show(10)

# %% Evaluation
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# %% Write predictions back to HBase
preds = predictions.select("HBASE_ROW_KEY", "prediction").collect()
for row in preds:
    table.put(row['HBASE_ROW_KEY'].encode(),
              {b'cf:Predicted_Credit_Score': str(row['prediction']).encode()})

print("Predictions written back to HBase")
