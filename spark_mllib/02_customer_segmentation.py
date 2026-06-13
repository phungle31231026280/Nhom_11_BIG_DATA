# Mục tiêu: Phân cụm khách hàng RFM (K-Means)

import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import warnings

warnings.filterwarnings('ignore')
load_dotenv()
HDFS_BASE = f"hdfs://{os.getenv('HDFS_HOST', 'localhost')}:{os.getenv('HDFS_PORT', '9000')}{os.getenv('HDFS_BASE_DIR', '/Group11_Dataset')}"
spark = SparkSession.builder.appName("TheLook_ML_BT2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
os.makedirs("../media", exist_ok=True)

# 1. PREP DATA
master_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{HDFS_BASE}/thelook_ecommerce.order_items.csv").alias("oi") \
    .join(spark.read.option("header", "true").option("inferSchema", "true").csv(f"{HDFS_BASE}/thelook_ecommerce.orders.csv").alias("o"), on="order_id", how="left")

max_date = master_df.select(F.max("o.created_at")).collect()[0][0]
rfm_df = master_df.groupBy("user_id").agg(
    F.datediff(F.lit(max_date), F.max("o.created_at")).alias("recency"),
    F.countDistinct("order_id").alias("frequency"),
    F.sum("sale_price").alias("monetary")
).dropna(subset=["recency", "frequency", "monetary"])

# 2. TRAIN MODEL K-MEANS
assembler = VectorAssembler(inputCols=["recency", "frequency", "monetary"], outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=4, seed=42)

pipeline = Pipeline(stages=[assembler, scaler, kmeans])
print("⏳ Đang huấn luyện KMeans...")
model = pipeline.fit(rfm_df)
predictions = model.transform(rfm_df)

# 3. ĐÁNH GIÁ & VẼ HÌNH
silhouette = ClusteringEvaluator(featuresCol="features", metricName="silhouette").evaluate(predictions)
print(f"✅ Silhouette Score: {silhouette:.4f}")

plot_df = predictions.select("recency", "monetary", "cluster").sample(0.2, seed=42).toPandas()
plt.figure(figsize=(8, 6))
scatter = plt.scatter(plot_df["recency"], plot_df["monetary"], c=plot_df["cluster"], cmap='viridis', alpha=0.5)
plt.title("Phân cụm RFM (Recency vs Monetary)")
plt.xlabel("Recency")
plt.ylabel("Monetary ($)")
plt.colorbar(scatter, label='Cluster')
plt.savefig("../media/fig_02_kmeans_rfm.png")
spark.stop()