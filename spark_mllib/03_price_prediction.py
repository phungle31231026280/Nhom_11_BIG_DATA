# Mục tiêu: Dự báo Giá bán sản phẩm (GBT Regressor)

import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import warnings

warnings.filterwarnings('ignore')
load_dotenv()
HDFS_BASE = f"hdfs://{os.getenv('HDFS_HOST', 'localhost')}:{os.getenv('HDFS_PORT', '9000')}{os.getenv('HDFS_BASE_DIR', '/Group11_Dataset')}"
spark = SparkSession.builder.appName("TheLook_ML_BT3").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
os.makedirs("../media", exist_ok=True)

# 1. PREP DATA
master_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{HDFS_BASE}/thelook_ecommerce.order_items.csv").alias("oi") \
    .join(spark.read.option("header", "true").option("inferSchema", "true").csv(f"{HDFS_BASE}/thelook_ecommerce.users.csv").alias("u"), F.col("oi.user_id") == F.col("u.id"), how="left") \
    .join(spark.read.option("header", "true").option("inferSchema", "true").csv(f"{HDFS_BASE}/thelook_ecommerce.products.csv").alias("p"), F.col("oi.product_id") == F.col("p.id"), how="left")

df_reg = master_df.filter(F.col("sale_price") > 0) \
    .dropna(subset=["sale_price", "retail_price", "cost", "gender", "category", "traffic_source", "age"]) \
    .withColumn("discount_rate", F.when(F.col("retail_price") > 0, (F.col("retail_price") - F.col("sale_price")) / F.col("retail_price")).otherwise(0.0))

# 2. TRAIN MODEL GBT
cat_cols = ["gender", "category", "traffic_source"]
num_cols = ["retail_price", "cost", "age", "discount_rate"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols]
assembler = VectorAssembler(inputCols=num_cols + [f"{c}_ohe" for c in cat_cols], outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled")
gbt = GBTRegressor(featuresCol="features_scaled", labelCol="sale_price", predictionCol="prediction", maxDepth=5, seed=42)

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])
train_data, test_data = df_reg.randomSplit([0.8, 0.2], seed=42)

print("⏳ Đang huấn luyện GBT Regressor...")
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# 3. ĐÁNH GIÁ & VẼ HÌNH
rmse = RegressionEvaluator(labelCol="sale_price", predictionCol="prediction", metricName="rmse").evaluate(predictions)
print(f"✅ GBT RMSE: {rmse:.4f} USD")

plot_df = predictions.select("sale_price", "prediction").sample(0.1, seed=42).toPandas()
plt.figure(figsize=(6,6))
plt.scatter(plot_df["sale_price"], plot_df["prediction"], alpha=0.3, color="green")
plt.plot([0, plot_df["sale_price"].max()], [0, plot_df["sale_price"].max()], 'r--')
plt.title(f"GBT Regressor: Actual vs Predicted (RMSE: {rmse:.2f})")
plt.xlabel("Giá thực tế ($)")
plt.ylabel("Giá dự đoán ($)")
plt.savefig("../media/fig_03_gbt_reg.png")
spark.stop()