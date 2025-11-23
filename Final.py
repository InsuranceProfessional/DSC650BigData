# %% Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
import happybase

# %% Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("HBase Credit Score Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# %% Step 2: Load data from Hive table 'final'
df = spark.sql("SELECT * FROM final")
df.show(5)

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
        df = df.withColumn(col_name, col(col_name).cast("double"))

# %% Step 4: Encode categorical columns
categorical_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour',
    'Occ_Accountant', 'Occ_Architect', 'Occ_Developer', 'Occ_Doctor', 'Occ_Engineer',
    'Occ_Entrepreneur', 'Occ_Journalist', 'Occ_Lawyer', 'Occ_Manager',
    'Occ_Mechanic', 'Occ_Media_Manager', 'Occ_Musician', 'Occ_Scientist',
    'Occ_Teacher', 'Occ_Writer'
]

indexers = []
for cat_col in categorical_cols:
    if cat_col in df.columns:
        indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
        df = indexer.fit(df).transform(df)
        indexers.append(cat_col + "_idx")

# %% Step 5: Prepare feature vector
feature_cols = [c for c in numeric_cols if c in df.columns] + indexers
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# %% Step 6: Prepare label column
label_col = "Credit_Score"
if label_col in df.columns:
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep")
    df = label_indexer.fit(df).transform(df)
else:
    raise ValueError("Credit_Score column not found in table!")

# %% Step 7: Split into train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# %% Step 8: Train Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                            numTrees=310, maxDepth=22, seed=42)
model = rf.fit(train_df)

# %% Step 9: Make predictions
predictions = model.transform(test_df)
predictions.select("ID", "Credit_Score", "prediction").show(10)

# %% Step 10: Write predictions back to HBase
# Connect to HBase
connection = happybase.Connection('master')  # replace 'master' with your HBase master host
table = connection.table('final')            # HBase table name

# Convert predictions to pandas for easy row-wise writing
predictions_pd = predictions.select("ID", "prediction").toPandas()

# Write each row back to HBase
for _, row in predictions_pd.iterrows():
    table.put(str(row['ID']),
              {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()})

print("Predictions written back to HBase successfully.")

spark.stop()
