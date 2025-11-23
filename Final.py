from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import happybase

# ---------------------------
# Step 1: Initialize Spark
# ---------------------------
spark = SparkSession.builder \
    .appName("CreditScoreClassification") \
    .enableHiveSupport() \
    .getOrCreate()

# ---------------------------
# Step 2: Load Data
# ---------------------------
df = spark.sql("SELECT * FROM final")

# ---------------------------
# Step 3: Clean & Transform Credit_Score
# ---------------------------
df = df.withColumn(
    "Credit_Score",
    when(col("Credit_Score") == "", "unknown")
    .otherwise(col("Credit_Score"))
)

# ---------------------------
# Step 4: Convert Boolean columns to string
# ---------------------------
for c, t in df.dtypes:
    if t == 'boolean':
        df = df.withColumn(c, when(col(c) == True, "True").otherwise("False"))

# ---------------------------
# Step 5: Identify columns
# ---------------------------
all_cols = df.columns
numeric_cols = [c for c, t in df.dtypes if t in ("int", "double")]
object_cols = [c for c, t in df.dtypes if t not in ("int", "double")]

# ---------------------------
# Step 6: Handle categorical features
# ---------------------------
categorical_cols = [c for c in object_cols if c != 'Credit_Score']

# Remove high-cardinality categorical features for RandomForest
max_bins = 32
low_card_cats = []
for cat in categorical_cols:
    num_unique = df.select(cat).distinct().count()
    if num_unique < max_bins:
        low_card_cats.append(cat)
    else:
        print(f"Skipping {cat} with {num_unique} unique values")

categorical_cols = low_card_cats

# ---------------------------
# Step 7: Encode categorical features
# ---------------------------
indexers = []
for cat_col in categorical_cols + ['Credit_Score']:
    indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
    df = indexer.fit(df).transform(df)
    indexers.append(cat_col + "_idx")

# ---------------------------
# Step 8: Assemble features
# ---------------------------
feature_cols = [c for c in numeric_cols if c != 'ID'] + \
               [c for c in indexers if c != 'Credit_Score_idx']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
df = assembler.transform(df)

# ---------------------------
# Step 9: Split data
# ---------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ---------------------------
# Step 10: Train RandomForest
# ---------------------------
rf = RandomForestClassifier(
    labelCol="Credit_Score_idx",
    featuresCol="features",
    numTrees=100,
    maxDepth=10,
    maxBins=max_bins,
    seed=42
)

model = rf.fit(train_df)

# ---------------------------
# Step 11: Make predictions
# ---------------------------
predictions = model.transform(test_df)
predictions.select("ID", "Credit_Score", "prediction").show(10)

# ---------------------------
# Step 12: Evaluate performance
# ---------------------------
evaluator = MulticlassClassificationEvaluator(
    labelCol="Credit_Score_idx",
    predictionCol="prediction"
)

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ---------------------------
# Step 13: Write predictions to HBase
# ---------------------------
connection = happybase.Connection('localhost')
table = connection.table('predictions')  # Replace with your HBase table

for row in predictions.select("ID", "Credit_Score", "prediction").collect():
    table.put(str(row['ID']).encode(), {
        b'info:Credit_Score': str(row['Credit_Score']).encode(),
        b'info:prediction': str(row['prediction']).encode()
    })

# ---------------------------
# Step 14: Stop Spark
# ---------------------------
spark.stop()
