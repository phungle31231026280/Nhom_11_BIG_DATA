
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType,
    FloatType, DoubleType, BooleanType, TimestampType,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("SparkStreaming")

# ─────────────────────────────────────────────
# CONFIGURATION 
# ─────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"
HDFS_BASE       = f"hdfs://localhost:9000{os.getenv('HDFS_STREAMING_DIR', '/data/streaming')}"
CHECKPOINT_BASE = "file:///C:/tmp/spark_checkpoints"

TOPICS = {
    "orders":    "orders-topic",
    "events":    "events-topic",
    "inventory": "inventory-topic",
    "revenue":   "revenue-topic",
}

# Spark Kafka connector — đổi version nếu dùng Spark 3.5 / Kafka 4
# Dùng 2.12:3.5.3 ổn định cho môi trường local Windows
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"


# ─────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("TheLook_StructuredStreaming")
        .config("spark.jars.packages", KAFKA_PACKAGE)
        # Tuning cho streaming
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.schemaInference", "true")
        # Hadoop / HDFS (Windows path workaround)
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
	.config("spark.sql.debug.maxToStringFields", "10")
	.config("spark.redaction.string.regex", "ksabvkbjsva")
	.config("spark.sql.shuffle.partitions", "2")  
        .config("spark.streaming.concurrentJobs", "2")
        .config("spark.python.worker.memory", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info("SparkSession created: %s", spark.version)
    return spark


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

ORDER_SCHEMA = StructType([
    StructField("order_id",       LongType(),      True),
    StructField("user_id",        LongType(),      True),
    StructField("status",         StringType(),    True),
    StructField("gender",         StringType(),    True),
    StructField("created_at",     StringType(),    True),   # parse later
    StructField("returned_at",    StringType(),    True),
    StructField("shipped_at",     StringType(),    True),
    StructField("delivered_at",   StringType(),    True),
    StructField("num_of_item",    IntegerType(),   True),
    StructField("country",        StringType(),    True),
    StructField("traffic_source", StringType(),    True),
    StructField("product_id",     LongType(),      True),
    StructField("category",       StringType(),    True),
    StructField("brand",          StringType(),    True),
    StructField("sale_price",     DoubleType(),    True),
    StructField("cost",           DoubleType(),    True),
    StructField("is_anomaly",     BooleanType(),   True),
])

EVENT_SCHEMA = StructType([
    StructField("event_id",       LongType(),      True),
    StructField("user_id",        LongType(),      True),
    StructField("session_id",     StringType(),    True),
    StructField("event_type",     StringType(),    True),
    StructField("funnel_depth",   IntegerType(),   True),
    StructField("ip_address",     StringType(),    True),
    StructField("city",           StringType(),    True),
    StructField("country",        StringType(),    True),
    StructField("traffic_source", StringType(),    True),
    StructField("browser",        StringType(),    True),
    StructField("created_at",     StringType(),    True),
    StructField("uri",            StringType(),    True),
])

INVENTORY_SCHEMA = StructType([
    StructField("inventory_item_id",    LongType(),    True),
    StructField("product_id",           LongType(),    True),
    StructField("product_name",         StringType(),  True),
    StructField("category",             StringType(),  True),
    StructField("brand",                StringType(),  True),
    StructField("distribution_center",  StringType(),  True),
    StructField("cost",                 DoubleType(),  True),
    StructField("product_retail_price", DoubleType(),  True),
    StructField("stock_quantity",       IntegerType(), True),
    StructField("sold_quantity",        IntegerType(), True),
    StructField("available_quantity",   IntegerType(), True),
    StructField("is_low_stock",         BooleanType(), True),
    StructField("is_out_of_stock",      BooleanType(), True),
    StructField("updated_at",           StringType(),  True),
])

REVENUE_SCHEMA = StructType([
    StructField("order_id",       LongType(),    True),
    StructField("user_id",        LongType(),    True),
    StructField("product_id",     LongType(),    True),
    StructField("category",       StringType(),  True),
    StructField("brand",          StringType(),  True),
    StructField("country",        StringType(),  True),
    StructField("traffic_source", StringType(),  True),
    StructField("sale_price",     DoubleType(),  True),
    StructField("cost",           DoubleType(),  True),
    StructField("gross_margin",   DoubleType(),  True),
    StructField("num_of_item",    IntegerType(), True),
    StructField("total_revenue",  DoubleType(),  True),
    StructField("total_cost",     DoubleType(),  True),
    StructField("total_margin",   DoubleType(),  True),
    StructField("status",         StringType(),  True),
    StructField("created_at",     StringType(),  True),
])


# ─────────────────────────────────────────────
# KAFKA READER HELPER
# ─────────────────────────────────────────────

def read_kafka_stream(spark: SparkSession, topic: str, schema: StructType):
    """
    Đọc stream từ Kafka, parse JSON value theo schema cho trước.
    Trả về DataFrame đã parse, chưa có event_time column.
    """
    raw_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")   # đọc từ đầu để không mất data
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 5000)        # giới hạn micro-batch
        .load()
    )
    # Kafka value là binary → cast sang STRING → parse JSON
    parsed_df = (
        raw_df
        .select(F.from_json(F.col("value").cast("string"), schema).alias("data"))
        .select("data.*")
    )
    return parsed_df


# ─────────────────────────────────────────────
# HDFS SINK HELPER
# ─────────────────────────────────────────────

def write_to_hdfs(stream_df, stream_name: str, output_mode: str = "append"):
    """Ghi DataFrame stream ra HDFS dạng Parquet."""
    hdfs_path  = f"{HDFS_BASE}/{stream_name}"
    ckpt_path  = f"{CHECKPOINT_BASE}/{stream_name}_hdfs"
    return (
        stream_df.writeStream
        .outputMode(output_mode)
        .format("parquet")
        .option("path", hdfs_path)
        .option("checkpointLocation", ckpt_path)
        .trigger(processingTime="30 seconds")   # ghi mỗi 30s
        .start()
    )


def write_to_console(stream_df, stream_name: str, output_mode: str = "complete", num_rows: int = 20):
    """Ghi kết quả ra console để debug / dashboard scrape."""
    ckpt_path = f"{CHECKPOINT_BASE}/{stream_name}_console"
    return (
        stream_df.writeStream
        .outputMode(output_mode)
        .format("console")
        .option("truncate", "false")
        .option("numRows", str(num_rows))
        .option("checkpointLocation", ckpt_path)
        .trigger(processingTime="10 seconds")
        .start()
    )


# ═════════════════════════════════════════════
# STREAM 1 : ORDER ANOMALY DETECTION
# Kỹ thuật: Tumbling Window 1 phút + Watermark 2 phút
#
# Bài toán: Phát hiện đơn hàng bất thường theo thời gian thực.
# Anomaly = đơn có sale_price > mean + 3*stddev HOẶC num_of_item > 15
# → Cho phép business team phản ứng kịp thời: hold order, review fraud.
# ═════════════════════════════════════════════

def stream1_order_anomaly(spark: SparkSession):
    log.info("Starting Stream 1: Order Anomaly Detection")

    df = read_kafka_stream(spark, TOPICS["orders"], ORDER_SCHEMA)

    # Parse timestamp + thêm event_time
    df = df.withColumn(
        "event_time",
        F.to_timestamp(F.col("created_at"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
    )

    # Watermark: chấp nhận late data tới 2 phút
    df = df.withWatermark("event_time", "2 minutes")

    # ── Tumbling Window 1 phút: thống kê tổng hợp theo window
    agg_df = (
        df
        .groupBy(
            F.window("event_time", "1 minute").alias("time_window"),
            "status",
        )
        .agg(
            F.count("order_id").alias("total_orders"),
            F.sum(F.when(F.col("is_anomaly") == True, 1).otherwise(0)).alias("anomaly_count"),
            F.avg("sale_price").alias("avg_sale_price"),
            F.max("sale_price").alias("max_sale_price"),
            F.avg("num_of_item").alias("avg_items"),
            F.max("num_of_item").alias("max_items"),
            F.sum("sale_price").alias("total_gmv"),           # Gross Merchandise Value
            F.approx_count_distinct("user_id").alias("unique_buyers"),
        )
        .withColumn("window_start",  F.col("time_window.start"))
        .withColumn("window_end",    F.col("time_window.end"))
        .withColumn(
            "anomaly_rate_pct",
            F.round(F.col("anomaly_count") / F.col("total_orders") * 100, 2),
        )
        .withColumn("stream_name",  F.lit("order_anomaly"))
        .withColumn("processed_at", F.current_timestamp())
        .drop("time_window")
    )

    # Sink 1: HDFS parquet (append mode — window data)
    hdfs_q  = write_to_hdfs(agg_df, "order_anomaly", output_mode="append")
    # Sink 2: Console
    cons_q  = write_to_console(agg_df, "order_anomaly_console", output_mode="append")
    return hdfs_q, cons_q


# ═════════════════════════════════════════════
# STREAM 2 : LIVE TRAFFIC FUNNEL
# Kỹ thuật: Sliding Window 2 phút / step 30 giây + Watermark 3 phút
#
# Bài toán: Đo chuyển đổi phễu theo thời gian thực
# home → department → category → product → cart → purchase
# Conversion rate = purchase_sessions / home_sessions * 100
# → Marketing team thấy ngay hiệu quả chiến dịch quảng cáo.
# ═════════════════════════════════════════════

def stream2_live_traffic_funnel(spark: SparkSession):
    log.info("Starting Stream 2: Live Traffic Funnel")

    df = read_kafka_stream(spark, TOPICS["events"], EVENT_SCHEMA)
    df = df.withColumn(
        "event_time",
        F.to_timestamp(F.col("created_at"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
    )
    df = df.withWatermark("event_time", "3 minutes")

    # Sliding Window 2 phút, slide 30 giây
    agg_df = (
        df
        .groupBy(
            F.window("event_time", "2 minutes", "30 seconds").alias("time_window"),
            "traffic_source",
        )
        .agg(
            F.count("event_id").alias("total_events"),
            F.approx_count_distinct("session_id").alias("total_sessions"),
            # Đếm session reach từng stage funnel
            F.approx_count_distinct(
                F.when(F.col("event_type") == "home",     F.col("session_id"))
            ).alias("stage_home"),
            F.approx_count_distinct(
                F.when(F.col("event_type") == "department", F.col("session_id"))
            ).alias("stage_department"),
            F.approx_count_distinct(
                F.when(F.col("event_type") == "product",  F.col("session_id"))
            ).alias("stage_product"),
            F.approx_count_distinct(
                F.when(F.col("event_type") == "cart",     F.col("session_id"))
            ).alias("stage_cart"),
            F.approx_count_distinct(
                F.when(F.col("event_type") == "purchase", F.col("session_id"))
            ).alias("stage_purchase"),
            F.approx_count_distinct("user_id").alias("unique_users"),
        )
        .withColumn("window_start", F.col("time_window.start"))
        .withColumn("window_end",   F.col("time_window.end"))
        # Tính conversion rate: purchase / home
        .withColumn(
            "funnel_conversion_pct",
            F.round(
                F.when(F.col("stage_home") > 0,
                       F.col("stage_purchase") / F.col("stage_home") * 100)
                .otherwise(0.0), 2
            ),
        )
        # Cart abandonment rate
        .withColumn(
            "cart_abandon_pct",
            F.round(
                F.when(F.col("stage_cart") > 0,
                       (F.col("stage_cart") - F.col("stage_purchase")) / F.col("stage_cart") * 100)
                .otherwise(0.0), 2
            ),
        )
        .withColumn("stream_name",  F.lit("traffic_funnel"))
        .withColumn("processed_at", F.current_timestamp())
        .drop("time_window")
    )

    hdfs_q = write_to_hdfs(agg_df, "traffic_funnel", output_mode="append")
    cons_q = write_to_console(agg_df, "traffic_funnel_console", output_mode="append")
    return hdfs_q, cons_q


# ═════════════════════════════════════════════
# STREAM 3 : INVENTORY ALERT (STATEFUL)
# Kỹ thuật: Stateful aggregation — groupBy product_id giữ state liên tục
#
# Bài toán: Cảnh báo real-time khi tồn kho thấp / hết hàng.
# State = available_quantity mới nhất của mỗi sản phẩm.
# Alert tiers:
#   - CRITICAL  : available_quantity == 0  (out of stock)
#   - WARNING   : available_quantity <= 10 (low stock)
#   - OK        : available_quantity > 10
# → Warehouse team nhận cảnh báo để bổ sung hàng kịp thời.
# ═════════════════════════════════════════════

def stream3_inventory_alert(spark: SparkSession):
    log.info("Starting Stream 3: Inventory Alert (Stateful)")

    df = read_kafka_stream(spark, TOPICS["inventory"], INVENTORY_SCHEMA)
    df = df.withColumn(
        "event_time",
        F.to_timestamp(F.col("updated_at"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
    )
    # Không dùng watermark khi dùng complete outputMode với groupBy không window
    # df = df.withWatermark("event_time", "5 minutes")  # ← bỏ để tránh AnalysisException

    # Stateful: lấy bản ghi mới nhất của từng product (last state)
    # Dùng groupBy + last() — sử dụng complete mode để luôn hiện state hiện tại
    agg_df = (
        df
        .groupBy("product_id", "category", "brand", "distribution_center")
        .agg(
            F.last("product_name",        ignorenulls=True).alias("product_name"),
            F.last("available_quantity",  ignorenulls=True).alias("available_quantity"),
            F.last("stock_quantity",      ignorenulls=True).alias("stock_quantity"),
            F.last("sold_quantity",       ignorenulls=True).alias("sold_quantity"),
            F.last("product_retail_price",ignorenulls=True).alias("retail_price"),
            F.last("cost",                ignorenulls=True).alias("cost"),
            F.last("is_low_stock",        ignorenulls=True).alias("is_low_stock"),
            F.last("is_out_of_stock",     ignorenulls=True).alias("is_out_of_stock"),
            F.last("event_time",          ignorenulls=True).alias("last_updated"),
            F.count("inventory_item_id").alias("update_count"),
        )
        # Gán alert tier
        .withColumn(
            "alert_tier",
            F.when(F.col("available_quantity") == 0,    F.lit("CRITICAL"))
            .when(F.col("available_quantity") <= 10,    F.lit("WARNING"))
            .when(F.col("available_quantity") <= 30,    F.lit("NOTICE"))
            .otherwise(                                  F.lit("OK"))
        )
        # Potential stock-out revenue loss (giá trị hàng bị thiếu)
        .withColumn(
            "potential_lost_revenue",
            F.when(
                F.col("available_quantity") < 10,
                F.round((10 - F.col("available_quantity")) * F.col("retail_price"), 2)
            ).otherwise(0.0)
        )
        .withColumn("stream_name",  F.lit("inventory_alert"))
        .withColumn("processed_at", F.current_timestamp())
    )

    # Console: chỉ hiện CRITICAL và WARNING để không flood console
    alert_df = agg_df.filter(F.col("alert_tier").isin(["CRITICAL", "WARNING"]))

    # HDFS: ghi toàn bộ state (complete mode không support parquet sink trực tiếp)
    # → dùng foreachBatch để ghi snapshot
    def write_batch_hdfs(batch_df, batch_id):
        path = f"{HDFS_BASE}/inventory_alert/batch_{batch_id}"
        batch_df.write.mode("overwrite").parquet(path)
        log.info("Inventory batch %d written to HDFS: %s", batch_id, path)

    hdfs_q = (
        agg_df.writeStream
        .outputMode("complete")
        .foreachBatch(write_batch_hdfs)
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/inventory_alert_hdfs")
        .trigger(processingTime="30 seconds")
        .start()
    )

    cons_q = write_to_console(alert_df, "inventory_alert_console", output_mode="complete")
    return hdfs_q, cons_q


# ═════════════════════════════════════════════
# STREAM 4 : REVENUE DASHBOARD
# Kỹ thuật: Sliding Window 2 phút / step 30 giây + join với static DataFrame
#
# Bài toán: Dashboard doanh thu real-time phân tách theo:
#   - Category / Brand
#   - Traffic source (ROI chiến dịch)
#   - Country
# Static join: enrich category với thông tin margin target từ dict tĩnh.
# ═════════════════════════════════════════════

def stream4_revenue_dashboard(spark: SparkSession):
    log.info("Starting Stream 4: Revenue Dashboard")

    df = read_kafka_stream(spark, TOPICS["revenue"], REVENUE_SCHEMA)
    df = df.withColumn(
        "event_time",
        F.to_timestamp(F.col("created_at"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
    )
    df = df.withWatermark("event_time", "3 minutes")

    agg_df = (
        df
        .groupBy(
            F.window("event_time", "2 minutes", "30 seconds").alias("time_window"),
            "category",
            "traffic_source",
        )
        .agg(
            F.count("order_id").alias("order_count"),
            F.sum("total_revenue").alias("gross_revenue"),
            F.sum("total_cost").alias("total_cost"),
            F.sum("total_margin").alias("gross_margin"),
            F.avg("gross_margin").alias("avg_margin_per_order"),
            F.sum("num_of_item").alias("units_sold"),
            F.approx_count_distinct("user_id").alias("unique_buyers"),
            F.avg("sale_price").alias("avg_order_value"),
        )
        .withColumn("window_start", F.col("time_window.start"))
        .withColumn("window_end",   F.col("time_window.end"))
        .withColumn(
            "margin_pct",
            F.round(
                F.when(F.col("gross_revenue") > 0,
                       F.col("gross_margin") / F.col("gross_revenue") * 100)
                .otherwise(0.0), 2
            ),
        )
        .drop("time_window")
    )

    enriched_df = (
        agg_df
        .withColumn(
            "margin_target_pct",
            F.when(F.col("category") == "Intimates", 35.0)
             .when(F.col("category") == "Jeans", 30.0)
             .when(F.col("category") == "Swim", 40.0)
             .when(F.col("category") == "Pants & Capris", 28.0)
             .when(F.col("category") == "Shorts", 32.0)
             .when(F.col("category") == "Tops & Tees", 25.0)
             .when(F.col("category") == "Blazers & Jackets", 38.0)
             .when(F.col("category") == "Dresses", 35.0)
             .when(F.col("category") == "Accessories", 50.0)
             .when(F.col("category") == "Socks & Hosiery", 20.0)
             .when(F.col("category") == "Suits & Sport Coats", 42.0)
             .when(F.col("category") == "Sweaters", 30.0)
             .when(F.col("category") == "Active", 33.0)
             .when(F.col("category") == "Outerwear & Coats", 40.0)
             .when(F.col("category") == "Skirts", 35.0)
             .otherwise(0.0)
        )
        .withColumn(
            "vs_margin_target",
            F.round(F.col("margin_pct") - F.col("margin_target_pct"), 2),
        )
        .withColumn(
            "margin_status",
            F.when(F.col("vs_margin_target") >= 0,  F.lit("ABOVE_TARGET"))
            .when(F.col("vs_margin_target") >= -5,  F.lit("ON_TARGET"))
            .otherwise(                             F.lit("BELOW_TARGET"))
        )
        .withColumn("stream_name",  F.lit("revenue_dashboard"))
        .withColumn("processed_at", F.current_timestamp())
    )

    hdfs_q = write_to_hdfs(enriched_df, "revenue_dashboard", output_mode="append")
    cons_q = write_to_console(enriched_df, "revenue_dashboard_console", output_mode="append", num_rows=10)
    return hdfs_q, cons_q

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    spark = create_spark_session()

    log.info("=" * 60)
    log.info("TheLook Structured Streaming — Starting all 4 streams")
    log.info("HDFS output base : %s", HDFS_BASE)
    log.info("Kafka broker     : %s", KAFKA_BOOTSTRAP)
    log.info("=" * 60)

    queries = []

    try:
        # ── Launch all 4 streams
        q1_hdfs, q1_cons = stream1_order_anomaly(spark)
        queries.extend([q1_hdfs, q1_cons])
        log.info("[Stream 1] Order Anomaly Detection   → ACTIVE")

        q2_hdfs, q2_cons = stream2_live_traffic_funnel(spark)
        queries.extend([q2_hdfs, q2_cons])
        log.info("[Stream 2] Live Traffic Funnel        → ACTIVE")

        q3_hdfs, q3_cons = stream3_inventory_alert(spark)
        queries.extend([q3_hdfs, q3_cons])
        log.info("[Stream 3] Inventory Alert (Stateful) → ACTIVE")

        q4_hdfs, q4_cons = stream4_revenue_dashboard(spark)
        queries.extend([q for q in [q4_hdfs, q4_cons] if q is not None])
        log.info("[Stream 4] Revenue Dashboard          → ACTIVE")

        log.info("All streams running. Press Ctrl+C to stop.")

        # Chờ tất cả query hoàn thành (chạy mãi cho đến khi interrupt)
        spark.streams.awaitAnyTermination()

    except KeyboardInterrupt:
        log.info("Shutdown signal received.")
    finally:
        for q in queries:
            try:
                if q is not None:
                    q.stop()
            except Exception:
                pass
        spark.stop()
        log.info("All streams stopped. SparkSession closed.")


if __name__ == "__main__":
    main()