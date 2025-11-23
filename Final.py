from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# -------------------------------
# Step 1: Create Spark session
# -------------------------------
spark = SparkSession.builder \
    .appName("MLlib GradesML Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# -------------------------------
# Step 2: Load data from Hive
# -------------------------------
grades_df = spark.sql("SELECT Customer_ID, Credit_Score FROM gradesml")

# -------------------------------
# Step 3: Prepare features
# -------------------------------
assembler = VectorAssembler(inputCols=['Credit_Score'], outputCol='features')
grades_df = assembler.transform(grades_df)

# -------------------------------
# Step 4: Train linear regression model
# -------------------------------
lr = LinearRegression(featuresCol='features', labelCol='Credit_Score')
lr_model = lr.fit(grades_df)

# -------------------------------
# Step 5: Make predictions
# -------------------------------
predictions_df = lr_model.transform(grades_df) \
    .select('Customer_ID', 'Credit_Score', 'prediction')

# -------------------------------
# Step 6: Function to write to HBase per partition
# -------------------------------
def write_partition(partition):
    import happybase  # import inside function for worker scope
    connection = happybase.Connection('hbase_host', timeout=60000)
    table = connection.table('your_table')
    with table.batch(batch_size=1000) as b:
        for row in partition:
            if row['Customer_ID'] is not None:
                b.put(
                    str(int(row['Customer_ID'])),  # HBase row key as string
                    {b'cf:Predicted_Credit_Score': str(int(row['prediction'])).encode()}
                )
    connection.close()

# -------------------------------
# Step 7: Write predictions to HBase
# -------------------------------
predictions_df.foreachPartition(write_partition)

# -------------------------------
# Step 8: Stop Spark session
# -------------------------------
spark.stop()
