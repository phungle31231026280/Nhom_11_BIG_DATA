import os
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings('ignore')
load_dotenv()
HDFS_BASE = f"hdfs://{os.getenv('HDFS_HOST', 'localhost')}:{os.getenv('HDFS_PORT', '9000')}{os.getenv('HDFS_BASE_DIR', '/Group11_Dataset')}"
spark = SparkSession.builder.appName("TheLook_ML_BT1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
os.makedirs("../media", exist_ok=True)

def load(filename): return spark.read.option("header", "true").option("inferSchema", "true").csv(f"{HDFS_BASE}/{filename}")

# 1. PREP DATA
master_df = load("thelook_ecommerce.order_items.csv").alias("oi") \
    .join(load("thelook_ecommerce.orders.csv").alias("o"), on="order_id", how="left") \
    .join(load("thelook_ecommerce.users.csv").alias("u"), F.col("oi.user_id") == F.col("u.id"), how="left") \
    .join(load("thelook_ecommerce.products.csv").alias("p"), F.col("oi.product_id") == F.col("p.id"), how="left") \
    .select(
        F.col("oi.sale_price"),
        F.col("p.retail_price"),
        F.col("p.cost"),
        F.col("u.age"),
        F.col("u.gender"),               # Chỉ lấy gender của bảng users
        F.col("p.category"),
        F.col("u.traffic_source"),
        F.col("o.status")                # Lấy status của bảng orders
    )

df_clf = master_df.withColumn("is_returned", F.when(F.col("status") == "Returned", 1.0).otherwise(0.0)) \
    .dropna(subset=["sale_price", "retail_price", "cost", "gender", "category", "traffic_source", "age"]) \
    .withColumn("discount_rate", F.when(F.col("retail_price") > 0, (F.col("retail_price") - F.col("sale_price")) / F.col("retail_price")).otherwise(0.0)) \
    .withColumn("price_gap", F.col("retail_price") - F.col("cost"))

# 2. TRAIN MODEL
total_clf = df_clf.count()
pos_clf   = df_clf.filter(F.col("is_returned") == 1.0).count()
balance_ratio = (total_clf - pos_clf) / pos_clf
df_clf = df_clf.withColumn("classWeight", F.when(F.col("is_returned") == 1.0, balance_ratio).otherwise(1.0))

cat_cols = ["gender", "category", "traffic_source"]
num_cols = ["sale_price", "retail_price", "cost", "age", "discount_rate", "price_gap"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols]
assembler = VectorAssembler(inputCols=num_cols + [f"{c}_ohe" for c in cat_cols], outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled")
selector = ChiSqSelector(numTopFeatures=15, featuresCol="features_scaled", outputCol="features", labelCol="is_returned")
rf = RandomForestClassifier(featuresCol="features", labelCol="is_returned", weightCol="classWeight", numTrees=50, maxDepth=10, seed=42)

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, selector, rf])
train_data, test_data = df_clf.randomSplit([0.8, 0.2], seed=42)

print("⏳ Đang huấn luyện Random Forest...")
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# 3. ĐÁNH GIÁ & VẼ HÌNH
roc_auc = BinaryClassificationEvaluator(labelCol="is_returned", metricName="areaUnderROC").evaluate(predictions)
print(f" ROC-AUC Score: {roc_auc:.4f}")

roc_sample = predictions.select("is_returned", F.col("probability").getItem(1).alias("prob")).toPandas()
fpr, tpr, _ = roc_curve(roc_sample["is_returned"], roc_sample["prob"])
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.savefig("../media/fig_01_roc_rf.png")
spark.stop()