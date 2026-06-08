import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from dotenv import load_dotenv
load_dotenv()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, unix_timestamp, when, date_format
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})

SUPTITLE_KW = dict(fontsize=14, fontweight="bold")
DIVIDER     = "=" * 60
SEP         = lambda title: print(f"\n{'-'*40}\n  {title}\n{'-'*40}")

def save_fig(path, suptitle=None):
    if suptitle:
        plt.suptitle(suptitle, **SUPTITLE_KW)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Spark session & data load
# ─────────────────────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("TheLook_EDA_Full")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = os.getenv('HDFS_BASE_DIR')
def load(name):
    path = f"{HDFS_BASE}/thelook_ecommerce.{name}.csv"
    print(f"  Loading {name} ...")
    return (
        spark.read
        .option("header",      "true")
        .option("inferSchema", "true")
        .option("multiLine",   "true")
        .option("escape",      '"')
        .csv(path)
    )

print("Loading data from HDFS...")
orders_sp      = load("orders")
order_items_sp = load("order_items")
users_sp       = load("users")
products_sp    = load("products")
inventory_sp   = load("inventory_items")
events_sp      = load("events")
dist_centers_sp = load("distribution_centers")
print("Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Parse timestamps & derive master DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def safe_ts(c):
    return (
        F.regexp_replace(
            F.regexp_replace(col(c), r'\.\d+', ''),
            r'\s*UTC\s*$', ''
        )
        .cast("timestamp")
    )

order_items_sp = (
    order_items_sp
    .withColumn("shipped_at",    safe_ts("shipped_at"))
    .withColumn("delivered_at",  safe_ts("delivered_at"))
    .withColumn("returned_at",   safe_ts("returned_at"))
    .withColumn("created_at_ts", safe_ts("created_at"))
    .withColumn("delivery_days",
        (unix_timestamp("delivered_at") - unix_timestamp("shipped_at")) / 86400
    )
)
orders_sp = orders_sp.withColumn("created_at_ts", safe_ts("created_at"))

inventory_sp = (
    inventory_sp
    .withColumn("created_at_ts", safe_ts("created_at"))
    .withColumn("sold_at_ts",    safe_ts("sold_at"))
    .withColumn("days_to_sell",
        (unix_timestamp("sold_at_ts") - unix_timestamp("created_at_ts")) / 86400
    )
)

events_sp = events_sp.withColumn("created_at_ts", safe_ts("created_at"))

full_sp = (
    order_items_sp
    .join(
        users_sp.select(
            col("id").alias("user_id_u"),
            "gender", "age", "country", "traffic_source",
        ),
        order_items_sp["user_id"] == col("user_id_u"),
        how="left",
    )
    .withColumn("is_returned", when(col("status") == "Returned", 1).otherwise(0))
    .withColumn("is_complete",  when(col("status") == "Complete",  1).otherwise(0))
)

AGE_BINS   = [0, 18, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["<18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]

def age_group_col(c="age"):
    expr = when(col(c) < AGE_BINS[1], AGE_LABELS[0])
    for i in range(1, len(AGE_LABELS)):
        expr = expr.when((col(c) >= AGE_BINS[i]) & (col(c) < AGE_BINS[i + 1]), AGE_LABELS[i])
    return expr.otherwise("Unknown")

full_sp      = full_sp.withColumn("age_group", age_group_col())
completed_sp = full_sp.filter(col("is_complete") == 1)

tables = {
    "orders":             orders_sp,
    "order_items":        order_items_sp,
    "users":              users_sp,
    "products":           products_sp,
    "inventory_items":    inventory_sp,
    "events":             events_sp,
    "distribution_centers": dist_centers_sp,
}

print(f"Master DataFrame ready.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  PHẦN 0 — KIỂM TRA SỨC KHOẺ DATA (Đơn biến phi đồ hoạ)
# ═════════════════════════════════════════════════════════════════════════════
print(DIVIDER)
print("PHẦN 0: KIỂM TRA SỨC KHOẺ DATA")
print(DIVIDER)


# ── 0.1  Shape & dtypes ──────────────────────────────────────────────────────
print("\n[0.1] Shape & dtypes")

for name, df in tables.items():
    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"\n  [{name}]  {n_rows:,} rows × {n_cols} cols")
    print(pd.DataFrame(df.dtypes, columns=["column", "dtype"]).to_string(index=False))


# ── 0.2  Descriptive statistics ──────────────────────────────────────────────
print("\n[0.2] Descriptive statistics")

SEP("order_items — sale_price")
order_items_sp.select("sale_price").describe().show()

SEP("users — age")
users_sp.select("age").describe().show()

SEP("products — cost & retail_price")
products_sp.select("cost", "retail_price").describe().show()


# ── 0.3  Null count & classification ─────────────────────────────────────────
print("\n[0.3] Null count & classification")

def null_report(name, df):
    total      = df.count()
    null_counts = (
        df.select([F.sum(F.when(col(c).isNull(), 1).otherwise(0)).alias(c)
                   for c in df.columns])
        .toPandas().T.rename(columns={0: "null_count"})
    )
    null_counts["null_pct"] = (null_counts["null_count"] / total * 100).round(2)
    null_counts = null_counts[null_counts["null_count"] > 0].sort_values("null_count", ascending=False)
    print(f"\n  [{name}]")
    if null_counts.empty:
        print("    ✅ No nulls.")
    else:
        print(null_counts.to_string())

for name, df in tables.items():
    null_report(name, df)

print("\n  Null breakdown by status (order_items):")
(
    order_items_sp
    .groupBy("status")
    .agg(
        F.count("*").alias("total"),
        F.sum(F.when(col("shipped_at").isNull(),   1).otherwise(0)).alias("null_shipped_at"),
        F.sum(F.when(col("delivered_at").isNull(), 1).otherwise(0)).alias("null_delivered_at"),
        F.sum(F.when(col("returned_at").isNull(),  1).otherwise(0)).alias("null_returned_at"),
    )
    .orderBy("status")
    .toPandas()
    .pipe(lambda df: print(df.to_string(index=False)) or df)
)
print("""
  ➡ shipped_at  null → Processing/Cancelled (intentional)
  ➡ delivered_at null → chưa nhận hàng (intentional)
  ➡ returned_at  null → chưa trả hàng (intentional)
""")


# ── 0.4  Duplicate check ─────────────────────────────────────────────────────
print("\n[0.4] Duplicate check")

for name, df, key in [
    ("orders",      orders_sp,      "order_id"),
    ("order_items", order_items_sp, "id"),
    ("users",       users_sp,       "id"),
    ("products",    products_sp,    "id"),
]:
    total   = df.count()
    unique  = df.select(col(key)).distinct().count()
    dup_cnt = total - unique
    icon    = "✅" if dup_cnt == 0 else "⚠️ "
    print(f"  {icon} [{name}] key='{key}' | total={total:,} | unique={unique:,} | dup={dup_cnt:,}")

print("\n  Items per order_id (order_items — expected ≥1):")
(
    order_items_sp.groupBy("order_id").count()
    .agg(
        F.min("count").alias("min_items"),
        F.max("count").alias("max_items"),
        F.mean("count").alias("avg_items"),
    )
    .toPandas()
    .round(2)
    .pipe(lambda df: print(df.to_string(index=False)))
)


# ── 0.5  Range check ─────────────────────────────────────────────────────────
print("\n[0.5] Range check")

SEP("sale_price (order_items)")
sp = order_items_sp.agg(
    F.min("sale_price").alias("min"),
    F.max("sale_price").alias("max"),
    F.sum(F.when(col("sale_price") <= 0, 1).otherwise(0)).alias("zero_or_neg"),
    F.sum(F.when(col("sale_price").isNull(), 1).otherwise(0)).alias("null"),
    F.percentile_approx("sale_price", [0.01, 0.25, 0.5, 0.75, 0.99]).alias("pctiles"),
).toPandas()
print(sp[["min", "max", "zero_or_neg", "null"]].to_string(index=False))
print(f"  Percentiles [p1,p25,p50,p75,p99]: {sp['pctiles'].iloc[0]}")

SEP("cost (products)")
products_sp.agg(
    F.min("cost").alias("min"),
    F.max("cost").alias("max"),
    F.sum(F.when(col("cost") <= 0, 1).otherwise(0)).alias("zero_or_neg"),
    F.sum(F.when(col("cost").isNull(), 1).otherwise(0)).alias("null"),
).toPandas().pipe(lambda df: print(df.to_string(index=False)))

SEP("age (users)")
ag = users_sp.agg(
    F.min("age").alias("min"),
    F.max("age").alias("max"),
    F.sum(F.when((col("age") < 10) | (col("age") > 100), 1).otherwise(0)).alias("suspicious"),
    F.sum(F.when(col("age").isNull(), 1).otherwise(0)).alias("null"),
    F.percentile_approx("age", [0.01, 0.25, 0.5, 0.75, 0.99]).alias("pctiles"),
).toPandas()
print(ag[["min", "max", "suspicious", "null"]].to_string(index=False))
print(f"  Percentiles [p1,p25,p50,p75,p99]: {ag['pctiles'].iloc[0]}")

SEP("Temporal sanity: delivered_at >= created_at?")
time_issues = (
    order_items_sp
    .filter(col("delivered_at").isNotNull() & col("created_at").isNotNull())
    .filter(col("delivered_at") < safe_ts("created_at"))
    .count()
)
print(f"  {'✅' if time_issues == 0 else '⚠️ '} delivered_at < created_at: {time_issues:,} rows")


# ── 0.6  Order status distribution ───────────────────────────────────────────
print("\n[0.6] Order status distribution")

EXPECTED = {"Complete", "Shipped", "Processing", "Cancelled", "Returned"}

status_dist_0 = (
    order_items_sp
    .groupBy("status").agg(F.count("*").alias("count"))
    .orderBy(col("count").desc())
    .toPandas()
)
total_items_0 = status_dist_0["count"].sum()
status_dist_0["pct"] = (status_dist_0["count"] / total_items_0 * 100).round(2)
print(f"\n  Total items: {total_items_0:,}")
print(status_dist_0.to_string(index=False))

found   = set(status_dist_0["status"].dropna())
missing = EXPECTED - found
extra   = found - EXPECTED
print(f"\n  {'✅ All 5 statuses present' if not missing else '⚠️  Missing: ' + str(missing)}")
if extra:
    print(f"  ⚠️  Unexpected: {extra}")

print("\n  Cross-check orders vs order_items:")
orders_s = (
    orders_sp.groupBy("status").agg(F.count("*").alias("orders_count"))
    .toPandas().set_index("status")
)
items_s = status_dist_0.set_index("status")[["count"]].rename(columns={"count": "items_count"})
print(orders_s.join(items_s, how="outer").fillna(0).astype(int).to_string())


print(f"\n{DIVIDER}")
print("Schema overview complete.")
print(DIVIDER)