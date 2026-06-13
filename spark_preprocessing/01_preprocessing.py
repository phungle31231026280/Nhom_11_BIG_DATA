# ============================================================
# spark_preprocessing/01_preprocessing.py
# Mục tiêu: Làm sạch, Join bảng TheLook và chuẩn bị Pipeline riêng cho 3 bài toán
# ============================================================
import os
import warnings
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

warnings.filterwarnings('ignore')

# ==========================================
# SETUP BIẾN MÔI TRƯỜNG & HDFS
# ==========================================
load_dotenv()
HDFS_HOST     = os.getenv("HDFS_HOST", "localhost")
HDFS_PORT     = os.getenv("HDFS_PORT", "9000")
HDFS_BASE_DIR = os.getenv("HDFS_BASE_DIR", "/Group11_Dataset")

HDFS_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}{HDFS_BASE_DIR}"

print(f"[*] Đang kết nối HDFS tại: {HDFS_BASE}")

# Khởi tạo Spark
spark = SparkSession.builder.appName("TheLook_Preprocessing").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def load(filename):
    path = f"{HDFS_BASE}/{filename}"
    print(f"  Đang tải: {path}")
    return spark.read.option("header", "true").option("inferSchema", "true").csv(path)

# ------------------------------------------
# 1. ĐỌC DỮ LIỆU & TẠO MASTER TABLE (TIỀN XỬ LÝ CHUNG)
# ------------------------------------------
print("\n--- BƯỚC 1: TIỀN XỬ LÝ CHUNG (CLEANING & JOIN) ---")
orders_df      = load("thelook_ecommerce.orders.csv")
order_items_df = load("thelook_ecommerce.order_items.csv")
users_df       = load("thelook_ecommerce.users.csv")
products_df    = load("thelook_ecommerce.products.csv")

master_df = order_items_df.alias("oi") \
    .join(orders_df.select("order_id", "status", "created_at").alias("o"), on="order_id", how="left") \
    .join(users_df.select("id", "gender", "age", "traffic_source").alias("u"), F.col("oi.user_id") == F.col("u.id"), how="left") \
    .join(products_df.select("id", "category", "department", "retail_price", "cost").alias("p"), F.col("oi.product_id") == F.col("p.id"), how="left")

# Xóa Null và Outliers cho các cột quan trọng
master_clean = master_df.dropna(subset=["sale_price", "retail_price", "cost", "gender", "category", "traffic_source", "age"])

def cap_outlier_iqr(df, col_name):
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df.withColumn(col_name, F.when(F.col(col_name) < lower, lower).when(F.col(col_name) > upper, upper).otherwise(F.col(col_name)))

for col_name in ["sale_price", "retail_price", "cost", "age"]:
    master_clean = cap_outlier_iqr(master_clean, col_name)

# Tạo đặc trưng mới (Feature Engineering)
master_feat = master_clean \
    .withColumn("discount_rate", F.when(F.col("retail_price") > 0, (F.col("retail_price") - F.col("sale_price")) / F.col("retail_price")).otherwise(0.0)) \
    .withColumn("price_gap", F.col("retail_price") - F.col("cost")) \
    .withColumn("is_discounted", F.when(F.col("discount_rate") > 0, 1).otherwise(0).cast("double")) \
    .withColumn("profit_margin", F.col("sale_price") - F.col("cost"))

print(f" Xong tiền xử lý chung! Dữ liệu gốc: {master_feat.count():,} dòng.")

# ============================================================
# BƯỚC 2: TÁCH PIPELINE TIỀN XỬ LÝ RIÊNG CHO TỪNG BÀI TOÁN
# ============================================================
print("\n--- BƯỚC 2: TÁCH TIỀN XỬ LÝ CHO 3 BÀI TOÁN ---")

# ------------------------------------------
# BÀI TOÁN 1: RETURN PREDICTION (CLASSIFICATION)
# Note: Tạo nhãn (is_returned), chia cột Categorical/Numeric riêng
# ------------------------------------------
print("[*] Cấu hình Preprocessing cho Bài 1 (Random Forest)...")

# Chỉ dropna dựa trên các cột sẽ được dùng để train model
df_clf = master_feat.withColumn("is_returned", F.when(F.col("o.status") == "Returned", 1.0).otherwise(0.0)) \
    .dropna(subset=["sale_price", "retail_price", "cost", "age", "gender", "category", "department", "traffic_source"])

cat_cols_clf = ["gender", "category", "department", "traffic_source"]
num_cols_clf = ["sale_price", "retail_price", "cost", "age", "discount_rate", "price_gap", "is_discounted", "profit_margin"]

indexers_clf = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols_clf]
encoders_clf = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols_clf]
assembler_clf = VectorAssembler(inputCols=num_cols_clf + [f"{c}_ohe" for c in cat_cols_clf], outputCol="features_raw", handleInvalid="keep")
scaler_clf = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withMean=False, withStd=True)


# ------------------------------------------
# BÀI TOÁN 2: CUSTOMER SEGMENTATION (CLUSTERING)
# Note: Gom nhóm theo User, tạo 3 biến RFM, xử lý Outlier riêng cho RFM
# ------------------------------------------
print("[*] Cấu hình Preprocessing cho Bài 2 (K-Means RFM)...")
# Chỉ định rõ F.max("o.created_at") để Spark biết lấy ngày tạo đơn hàng
max_date = master_feat.select(F.max("o.created_at")).collect()[0][0]
rfm_df = master_feat.groupBy("user_id").agg(
    F.datediff(F.lit(max_date), F.max("o.created_at")).alias("recency"),
    F.countDistinct("order_id").alias("frequency"),
    F.sum("sale_price").alias("monetary")
).dropna(subset=["recency", "frequency", "monetary"]) # Khóa luôn vụ dropna mơ hồ

# Capping outlier riêng cho Monetary và Frequency
for col_name in ["monetary", "frequency"]:
    rfm_df = cap_outlier_iqr(rfm_df, col_name)

assembler_rfm = VectorAssembler(inputCols=["recency", "frequency", "monetary"], outputCol="features_raw")
scaler_rfm = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)


# ------------------------------------------
# BÀI TOÁN 3: PRICE PREDICTION (REGRESSION)
# Note: Lọc bỏ giá <= 0, không cần cột department vì có thể gây nhiễu
# ------------------------------------------
print("[*] Cấu hình Preprocessing cho Bài 3 (GBT Regressor)...")
df_reg = master_feat.filter(F.col("sale_price") > 0) \
    .dropna(subset=["sale_price", "retail_price", "cost", "age", "gender", "category", "traffic_source"])

cat_cols_reg = ["gender", "category", "traffic_source"]
num_cols_reg = ["retail_price", "cost", "age", "discount_rate", "price_gap", "is_discounted", "profit_margin"]

indexers_reg = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols_reg]
encoders_reg = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols_reg]
assembler_reg = VectorAssembler(inputCols=num_cols_reg + [f"{c}_ohe" for c in cat_cols_reg], outputCol="features_raw", handleInvalid="keep")
scaler_reg = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withMean=False, withStd=True)

spark.stop()