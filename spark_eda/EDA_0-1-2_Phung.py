
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

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

spark = (
    SparkSession.builder
    .appName("TheLook_EDA_Full")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs://localhost:9000/Doan/datatest"

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
#  SECTION 0 — DATA HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════
print(DIVIDER)
print("SECTION 0: DATA HEALTH CHECK")
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


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — BUSINESS OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("SECTION 1: BUSINESS OVERVIEW")
print(DIVIDER)


# ── 1.1  Monthly revenue trend ───────────────────────────────────────────────
print("\n[1.1] Monthly Revenue Trend")

monthly_rev = (
    completed_sp
    .withColumn("month", date_format("created_at_ts", "yyyy-MM"))
    .groupBy("month")
    .agg(F.sum("sale_price").alias("revenue"))
    .orderBy("month")
    .toPandas()
)

fig, ax = plt.subplots(figsize=(16, 6))
ax.fill_between(monthly_rev["month"], monthly_rev["revenue"], alpha=0.25, color="#2980b9")
ax.plot(monthly_rev["month"], monthly_rev["revenue"],
        color="#2980b9", linewidth=2.5, marker="o", markersize=5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K"))
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
n = max(1, len(monthly_rev) // 10)
ax.set_xticks(range(0, len(monthly_rev), n))
ax.set_xticklabels(monthly_rev["month"].iloc[::n], rotation=45, ha="right", fontsize=9)
peak = monthly_rev.loc[monthly_rev["revenue"].idxmax()]
ax.annotate(
    f"Peak\n${peak['revenue']/1e3:,.0f}K",
    xy=(peak["month"], peak["revenue"]),
    xytext=(10, 15), textcoords="offset points",
    fontsize=9, color="#c0392b",
    arrowprops=dict(arrowstyle="->", color="#c0392b"),
)
save_fig("1_1_monthly_revenue.png", "Monthly Revenue Trend (Completed Orders)")

print(f"  Total revenue   : ${monthly_rev['revenue'].sum():,.0f}")
print(f"  Peak month      : {peak['month']}  —  ${peak['revenue']:,.0f}")
print(f"  MoM avg growth  : {monthly_rev['revenue'].pct_change().mean()*100:.1f}%")


# ── 1.2  Monthly orders & AOV ────────────────────────────────────────────────
print("\n[1.2] Monthly Orders & AOV Trend")

monthly_orders = (
    completed_sp
    .withColumn("month", date_format("created_at_ts", "yyyy-MM"))
    .groupBy("month")
    .agg(
        F.countDistinct("order_id").alias("orders"),
        F.sum("sale_price").alias("revenue"),
    )
    .orderBy("month")
    .toPandas()
)
monthly_orders["AOV"] = monthly_orders["revenue"] / monthly_orders["orders"]

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
axes[0].fill_between(monthly_orders["month"], monthly_orders["orders"], alpha=0.25, color="#27ae60")
axes[0].plot(monthly_orders["month"], monthly_orders["orders"],
             color="#27ae60", linewidth=2.5, marker="o", markersize=5)
axes[0].set_ylabel("Number of Orders")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].set_title("Monthly Order Count", pad=8)

axes[1].plot(monthly_orders["month"], monthly_orders["AOV"],
             color="#e67e22", linewidth=2.5, marker="s", markersize=5)
axes[1].axhline(monthly_orders["AOV"].mean(), color="gray", linestyle="--", linewidth=1.2,
                label=f"Avg AOV: ${monthly_orders['AOV'].mean():.0f}")
axes[1].set_ylabel("AOV ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
axes[1].set_title("Monthly Average Order Value (AOV)", pad=8)
axes[1].legend()
n = max(1, len(monthly_orders) // 10)
axes[1].set_xticks(range(0, len(monthly_orders), n))
axes[1].set_xticklabels(monthly_orders["month"].iloc[::n], rotation=45, ha="right", fontsize=9)
save_fig("1_2_monthly_orders_aov.png",
         "Monthly Orders vs AOV: Volume ↑ but Basket Size Stable?")

print(f"  Avg monthly orders : {monthly_orders['orders'].mean():,.0f}")
print(f"  Avg AOV            : ${monthly_orders['AOV'].mean():.2f}")
print(f"  AOV range          : ${monthly_orders['AOV'].min():.2f} – ${monthly_orders['AOV'].max():.2f}")


# ── 1.3  Order status distribution ───────────────────────────────────────────
print("\n[1.3] Order Status Distribution")

status_dist = (
    order_items_sp
    .groupBy("status").agg(F.count("*").alias("count"))
    .orderBy(col("count").desc())
    .toPandas()
)
total_s = status_dist["count"].sum()
status_dist["pct"] = status_dist["count"] / total_s * 100

STATUS_COLORS = {
    "Complete":   "#2ecc71",
    "Shipped":    "#3498db",
    "Processing": "#f39c12",
    "Cancelled":  "#95a5a6",
    "Returned":   "#e74c3c",
}
colors_s = [STATUS_COLORS.get(s, "#bdc3c7") for s in status_dist["status"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].barh(status_dist["status"], status_dist["count"], color=colors_s)
axes[0].set_title("Order Status — Absolute Count")
axes[0].set_xlabel("Number of Items")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for i, (cnt, pct) in enumerate(zip(status_dist["count"], status_dist["pct"])):
    axes[0].text(cnt + total_s * 0.003, i, f"{pct:.1f}%", va="center", fontsize=9)
wedges, texts, autotexts = axes[1].pie(
    status_dist["pct"], labels=status_dist["status"], colors=colors_s,
    autopct="%1.1f%%", startangle=140, pctdistance=0.82,
)
for at in autotexts:
    at.set_fontsize(9)
axes[1].set_title("Order Status — Share (%)")
save_fig("1_3_order_status.png", "Order Status Distribution")

print(status_dist.to_string(index=False))


# ── 1.4  Sale price distribution ─────────────────────────────────────────────
print("\n[1.4] Sale Price Distribution")

price_sample = (
    order_items_sp
    .filter(col("sale_price").isNotNull() & (col("sale_price") > 0))
    .select("sale_price")
    .sample(fraction=min(1.0, 80000 / order_items_sp.count()))
    .toPandas()
)
price_stats = order_items_sp.agg(
    F.min("sale_price").alias("min"),
    F.max("sale_price").alias("max"),
    F.mean("sale_price").alias("mean"),
    F.percentile_approx("sale_price", [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).alias("pctiles"),
).toPandas().iloc[0]
p25, p50, p75, p90, p95, p99 = price_stats["pctiles"]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.histplot(price_sample["sale_price"], bins=80, kde=True, ax=axes[0], color="#3498db")
axes[0].axvline(price_stats["mean"], color="red",    linestyle="--", linewidth=1.5,
                label=f"Mean: ${price_stats['mean']:.0f}")
axes[0].axvline(p50,                 color="green",  linestyle="--", linewidth=1.5,
                label=f"Median: ${p50:.0f}")
axes[0].axvline(p95,                 color="orange", linestyle=":",  linewidth=1.5,
                label=f"p95: ${p95:.0f}")
axes[0].set_title("Full Distribution (log-scale x)")
axes[0].set_xlabel("Sale Price ($)")
axes[0].set_xscale("log")
axes[0].legend()

sns.histplot(price_sample[price_sample["sale_price"] <= p99]["sale_price"],
             bins=60, kde=True, ax=axes[1], color="#9b59b6")
axes[1].axvline(price_stats["mean"], color="red",   linestyle="--", linewidth=1.5,
                label=f"Mean: ${price_stats['mean']:.0f}")
axes[1].axvline(p50,                 color="green", linestyle="--", linewidth=1.5,
                label=f"Median: ${p50:.0f}")
axes[1].set_title(f"Capped at p99 (${p99:.0f})")
axes[1].set_xlabel("Sale Price ($)")
axes[1].legend()
save_fig("1_4_sale_price_dist.png",
         "Sale Price Distribution: Long Tail & High-Value Outliers")

for label, val in [("min", price_stats["min"]), ("mean", price_stats["mean"]),
                   ("p50", p50), ("p75", p75), ("p90", p90),
                   ("p95", p95), ("p99", p99), ("max", price_stats["max"])]:
    print(f"  {label:>4} : ${val:.2f}")


# ── 1.5  User age distribution ───────────────────────────────────────────────
print("\n[1.5] User Age Distribution")

users_age = users_sp.select("age").filter(col("age").isNotNull()).toPandas()
users_age["age_group"] = pd.cut(
    users_age["age"],
    bins=[0, 18, 25, 35, 45, 55, 65, 120],
    labels=AGE_LABELS[:-1] + ["65+"],
    right=False,
)
group_counts = (
    users_age.groupby("age_group", observed=True)["age"]
    .count().reset_index(name="count")
)
group_counts["pct"] = group_counts["count"] / group_counts["count"].sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors_age = sns.color_palette("Blues", len(group_counts))[::-1]
bars = axes[0].bar(group_counts["age_group"].astype(str), group_counts["count"], color=colors_age)
axes[0].set_title("User Count by Age Group")
axes[0].set_xlabel("Age Group")
axes[0].set_ylabel("Number of Users")
for bar, pct in zip(bars, group_counts["pct"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + users_age.shape[0] * 0.003,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
sns.histplot(users_age["age"], bins=50, kde=True, ax=axes[1], color="#1abc9c")
axes[1].axvline(users_age["age"].mean(),   color="red",  linestyle="--", linewidth=1.5,
                label=f"Mean: {users_age['age'].mean():.1f}")
axes[1].axvline(users_age["age"].median(), color="blue", linestyle="--", linewidth=1.5,
                label=f"Median: {users_age['age'].median():.1f}")
axes[1].set_title("Age Distribution (Continuous)")
axes[1].set_xlabel("Age")
axes[1].legend()
save_fig("1_5_user_age_dist.png", "User Age Distribution: Who Are Our Customers?")

print(group_counts.to_string(index=False))
print(f"  Mean: {users_age['age'].mean():.1f}  |  Median: {users_age['age'].median():.1f}"
      f"  |  Std: {users_age['age'].std():.1f}")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CUSTOMER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("SECTION 2: CUSTOMER ANALYSIS")
print(DIVIDER)


# ── 2.1  Gender × Revenue ────────────────────────────────────────────────────
print("\n[2.1] Gender × Revenue")

gender_rev = (
    completed_sp
    .filter(col("gender").isin("M", "F"))
    .groupBy("gender")
    .agg(
        F.sum("sale_price").alias("total_revenue"),
        F.countDistinct("user_id").alias("unique_buyers"),
        F.countDistinct("order_id").alias("total_orders"),
        F.mean("sale_price").alias("avg_item_price"),
    )
    .toPandas()
    .set_index("gender")
)
gender_rev["revenue_per_buyer"] = gender_rev["total_revenue"] / gender_rev["unique_buyers"]
gender_rev["orders_per_buyer"]  = gender_rev["total_orders"]  / gender_rev["unique_buyers"]

GENDER_COLORS = {"M": "#3498db", "F": "#e91e8c"}
colors_g = [GENDER_COLORS.get(g, "#95a5a6") for g in gender_rev.index]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, (metric, title, fmt) in zip(axes, [
    ("total_revenue",    "Total Revenue ($)",     lambda x: f"${x/1e6:.1f}M"),
    ("revenue_per_buyer","Revenue per Buyer ($)", lambda x: f"${x:,.0f}"),
    ("orders_per_buyer", "Orders per Buyer",      lambda x: f"{x:.2f}"),
]):
    bars = ax.bar(gender_rev.index, gender_rev[metric], color=colors_g, width=0.5)
    ax.set_title(title)
    ax.set_xlabel("Gender")
    for bar, val in zip(bars, gender_rev[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                fmt(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
save_fig("2_1_gender_revenue.png", "Gender × Revenue: Who Spends More?")

print(gender_rev[["total_revenue", "unique_buyers", "revenue_per_buyer",
                   "orders_per_buyer", "avg_item_price"]].round(2).to_string())


# ── 2.2  Age Group × Purchase Frequency ──────────────────────────────────────
print("\n[2.2] Age Group × Purchase Frequency")

age_stats = (
    completed_sp
    .filter(col("age_group") != "Unknown")
    .groupBy("age_group")
    .agg(
        F.countDistinct("user_id").alias("unique_buyers"),
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("sale_price").alias("total_revenue"),
        F.mean("sale_price").alias("avg_item_price"),
    )
    .toPandas()
    .set_index("age_group")
    .reindex(AGE_LABELS)
    .dropna(how="all")
)
age_stats["orders_per_buyer"]  = age_stats["total_orders"]  / age_stats["unique_buyers"]
age_stats["revenue_per_buyer"] = age_stats["total_revenue"] / age_stats["unique_buyers"]

palette_age = sns.color_palette("viridis", len(age_stats))
fig, axes   = plt.subplots(2, 2, figsize=(18, 12))

for ax, (metric, title, fmt) in zip(axes.flat, [
    ("unique_buyers",    "Unique Buyers by Age Group",  lambda x, _: f"{x:,.0f}"),
    ("orders_per_buyer", "Avg Orders per Buyer",        None),
    ("revenue_per_buyer","Revenue per Buyer ($)",       lambda x, _: f"${x:,.0f}"),
    ("avg_item_price",   "Avg Item Price ($)",          lambda x, _: f"${x:.0f}"),
]):
    age_stats[metric].plot(kind="bar", ax=ax, color=palette_age, rot=30)
    ax.set_title(title)
    if fmt:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))

axes[0, 1].axhline(age_stats["orders_per_buyer"].mean(), color="red",
                   linestyle="--", linewidth=1.2, label="Overall avg")
axes[0, 1].legend()
save_fig("2_2_age_purchase_freq.png",
         "Age Group × Purchase Behaviour: Volume, Frequency & Basket Size")

print(age_stats[["unique_buyers", "orders_per_buyer",
                  "revenue_per_buyer", "avg_item_price"]].round(2).to_string())


# ── 2.3  Traffic Source → Conversion Rate ────────────────────────────────────
print("\n[2.3] Traffic Source → Conversion Rate")

traffic_conv = (
    full_sp
    .filter(col("traffic_source").isNotNull())
    .groupBy("traffic_source")
    .agg(
        F.countDistinct("user_id").alias("total_users"),
        F.countDistinct(when(col("is_complete") == 1, col("user_id"))).alias("converted_users"),
        F.sum("sale_price").alias("total_revenue"),
    )
    .toPandas()
    .set_index("traffic_source")
)
traffic_conv["conversion_rate"] = traffic_conv["converted_users"] / traffic_conv["total_users"] * 100
traffic_conv["revenue_per_user"] = traffic_conv["total_revenue"] / traffic_conv["total_users"]
traffic_conv = traffic_conv.sort_values("conversion_rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
palette_conv = ["#2ecc71" if v >= traffic_conv["conversion_rate"].median() else "#e74c3c"
                for v in traffic_conv["conversion_rate"]]
bars = axes[0].bar(traffic_conv.index, traffic_conv["conversion_rate"],
                   color=palette_conv, width=0.55)
axes[0].axhline(traffic_conv["conversion_rate"].median(), color="black",
                linestyle="--", linewidth=1.2,
                label=f"Median: {traffic_conv['conversion_rate'].median():.1f}%")
axes[0].set_title("Conversion Rate by Traffic Source\n(Users → Completed Orders)")
axes[0].set_ylabel("Conversion Rate (%)")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=30)
for bar, val in zip(bars, traffic_conv["conversion_rate"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

axes[1].scatter(
    traffic_conv["conversion_rate"], traffic_conv["revenue_per_user"],
    s=traffic_conv["total_users"] / traffic_conv["total_users"].max() * 800 + 100,
    c=range(len(traffic_conv)), cmap="tab10",
    alpha=0.85, edgecolors="white", linewidths=1.2, zorder=5,
)
for src, row in traffic_conv.iterrows():
    axes[1].annotate(src, (row["conversion_rate"], row["revenue_per_user"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
axes[1].axvline(traffic_conv["conversion_rate"].mean(),  color="gray", linestyle="--", linewidth=1)
axes[1].axhline(traffic_conv["revenue_per_user"].mean(), color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Conversion Rate vs Revenue/User\n(Ideal: top-right; bubble = volume)")
axes[1].set_xlabel("Conversion Rate (%)")
axes[1].set_ylabel("Revenue per User ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
save_fig("2_3_traffic_conversion.png",
         "Traffic Source: Conversion Rate & Revenue Quality")

print(traffic_conv[["total_users", "converted_users",
                     "conversion_rate", "revenue_per_user"]].round(2).to_string())


# ── 2.4  Top Countries by Revenue ────────────────────────────────────────────
print("\n[2.4] Top Countries by Revenue")

country_rev = (
    completed_sp
    .filter(col("country").isNotNull())
    .groupBy("country")
    .agg(
        F.sum("sale_price").alias("revenue"),
        F.countDistinct("user_id").alias("buyers"),
        F.mean("sale_price").alias("avg_item_price"),
    )
    .orderBy(col("revenue").desc())
    .limit(20)
    .toPandas()
    .set_index("country")
)
country_rev["revenue_per_buyer"] = country_rev["revenue"] / country_rev["buyers"]

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
axes[0].barh(country_rev.index, country_rev["revenue"],
             color=sns.color_palette("Blues_r", len(country_rev)))
axes[0].set_title("Top 20 Countries — Total Revenue")
axes[0].set_xlabel("Revenue ($)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K"))
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)

rpb_sorted = country_rev.sort_values("revenue_per_buyer", ascending=False)
axes[1].barh(rpb_sorted.index, rpb_sorted["revenue_per_buyer"],
             color=sns.color_palette("Oranges_r", len(rpb_sorted)))
axes[1].set_title("Top 20 Countries — Revenue per Buyer")
axes[1].set_xlabel("Revenue / Buyer ($)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)
save_fig("2_4_country_revenue.png",
         "Top Countries by Revenue: Total vs Per-Buyer Value")

print(country_rev[["revenue", "buyers", "revenue_per_buyer",
                    "avg_item_price"]].head(10).round(2).to_string())


# ── 2.5  Top Countries by Return Rate ────────────────────────────────────────
print("\n[2.5] Top Countries by Return Rate")

country_return = (
    full_sp
    .filter(col("country").isNotNull())
    .groupBy("country")
    .agg(
        F.count("status").alias("total_items"),
        F.sum("is_returned").alias("returned_items"),
    )
    .filter(col("total_items") >= 100)
    .toPandas()
    .set_index("country")
)
country_return["return_rate"] = country_return["returned_items"] / country_return["total_items"] * 100

top_return = country_return.sort_values("return_rate", ascending=False).head(20)
merged     = top_return.join(country_rev[["revenue"]], how="left").sort_values("return_rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
palette_rr = ["#c0392b" if v >= country_return["return_rate"].median() else "#3498db"
              for v in top_return["return_rate"]]
axes[0].barh(top_return.index, top_return["return_rate"], color=palette_rr)
axes[0].axvline(country_return["return_rate"].median(), color="black",
                linestyle="--", linewidth=1.2,
                label=f"Global median: {country_return['return_rate'].median():.1f}%")
axes[0].set_title("Top 20 Countries — Return Rate (%)")
axes[0].set_xlabel("Return Rate (%)")
axes[0].legend()
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)

scatter_data = merged.dropna(subset=["revenue"])
axes[1].scatter(
    scatter_data["revenue"], scatter_data["return_rate"],
    s=(scatter_data["total_items"] / scatter_data["total_items"].max()) * 600 + 60,
    c=range(len(scatter_data)), cmap="tab20",
    alpha=0.85, edgecolors="white", linewidths=1, zorder=5,
)
for country, row in scatter_data.iterrows():
    axes[1].annotate(country, (row["revenue"], row["return_rate"]),
                     textcoords="offset points", xytext=(6, 3), fontsize=8)
axes[1].axvline(scatter_data["revenue"].mean(),     color="gray", linestyle="--", linewidth=1)
axes[1].axhline(scatter_data["return_rate"].mean(), color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Revenue vs Return Rate by Country\n(Top-left quadrant = high-risk market)")
axes[1].set_xlabel("Total Revenue ($)")
axes[1].set_ylabel("Return Rate (%)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K"))
save_fig("2_5_country_return_rate.png",
         "Country Risk Map: High Revenue vs High Return Rate")

print(top_return[["return_rate", "returned_items", "total_items"]].head(10).round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — INVENTORY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("SECTION 3: INVENTORY ANALYSIS")
print(DIVIDER)

# inventory_items join products để lấy category
inv_full_sp = (
    inventory_sp
    .join(
        products_sp.select(col("id").alias("prod_id"), "category", "name"),
        inventory_sp["product_id"] == col("prod_id"),
        how="left",
    )
    .withColumn("is_sold", when(col("sold_at_ts").isNotNull(), 1).otherwise(0))
)


# ── 3.1  Inventory health: sold vs unsold ────────────────────────────────────
print("\n[3.1] Inventory Health: Sold vs Unsold")

inv_status = (
    inv_full_sp
    .groupBy("category")
    .agg(
        F.count("*").alias("total_units"),
        F.sum("is_sold").alias("sold_units"),
    )
    .toPandas()
    .set_index("category")
)
inv_status["unsold_units"] = inv_status["total_units"] - inv_status["sold_units"]
inv_status["sell_through"]  = inv_status["sold_units"] / inv_status["total_units"] * 100
inv_status = inv_status.sort_values("sell_through", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))

inv_plot = inv_status[["sold_units", "unsold_units"]]
inv_plot.plot(kind="barh", stacked=True, ax=axes[0],
              color=["#2ecc71", "#e74c3c"])
axes[0].set_title("Sold vs Unsold Units by Category")
axes[0].set_xlabel("Units")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)
axes[0].legend(["Sold", "Unsold"])

palette_st = ["#27ae60" if v >= inv_status["sell_through"].median() else "#e74c3c"
              for v in inv_status["sell_through"]]
axes[1].barh(inv_status.index, inv_status["sell_through"], color=palette_st)
axes[1].axvline(inv_status["sell_through"].median(), color="black",
                linestyle="--", linewidth=1.2,
                label=f"Median: {inv_status['sell_through'].median():.1f}%")
axes[1].set_title("Sell-Through Rate (%) by Category")
axes[1].set_xlabel("Sell-Through Rate (%)")
axes[1].legend()
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)

save_fig("3_1_inventory_sold_unsold.png",
         "Inventory Health: Sell-Through Rate by Category")

print("\nSell-through rate by category:")
print(inv_status[["total_units", "sold_units", "unsold_units", "sell_through"]].round(2).to_string())


# ── 3.2  Time-to-sell distribution by category ───────────────────────────────
print("\n[3.2] Time-to-Sell Distribution by Category")

sold_items_sp = inv_full_sp.filter(
    col("days_to_sell").isNotNull() &
    (col("days_to_sell") > 0) &
    (col("days_to_sell") < 365)
)

tts_stats = (
    sold_items_sp
    .groupBy("category")
    .agg(
        F.percentile_approx("days_to_sell", 0.5).alias("median_days"),
        F.mean("days_to_sell").alias("mean_days"),
        F.count("days_to_sell").alias("sold_count"),
    )
    .orderBy("median_days")
    .toPandas()
    .set_index("category")
)

tts_sample = (
    sold_items_sp.select("category", "days_to_sell")
    .sample(fraction=min(1.0, 60000 / sold_items_sp.count()))
    .toPandas()
)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))

sns.boxplot(
    data=tts_sample, x="days_to_sell", y="category",
    order=tts_stats.index.tolist(),
    palette="coolwarm", ax=axes[0],
    showfliers=False, width=0.6,
)
axes[0].set_title("Days to Sell by Category (Boxplot)", pad=10)
axes[0].set_xlabel("Days to Sell")
axes[0].yaxis.set_tick_params(labelsize=9)

colors_tts = sns.color_palette("Blues_r", len(tts_stats))
axes[1].barh(tts_stats.index, tts_stats["median_days"], color=colors_tts)
axes[1].set_title("Median Days to Sell by Category", pad=10)
axes[1].set_xlabel("Median Days")
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)

save_fig("3_2_time_to_sell_dist.png",
         "Time-to-Sell: How Long Before Items Find a Buyer?")

print("\nTime-to-sell stats by category:")
print(tts_stats.round(2).to_string())


# ── 3.3  Inventory turnover rate by category ─────────────────────────────────
print("\n[3.3] Inventory Turnover Rate by Category")

# Turnover = sold_units / total_units per month (avg monthly throughput)
inv_monthly = (
    inv_full_sp
    .filter(col("is_sold") == 1)
    .withColumn("month", date_format("sold_at_ts", "yyyy-MM"))
    .groupBy("category", "month")
    .agg(F.count("*").alias("sold_per_month"))
    .groupBy("category")
    .agg(F.mean("sold_per_month").alias("avg_monthly_sold"))
    .toPandas()
    .set_index("category")
)
inv_turnover = inv_status[["total_units"]].join(inv_monthly, how="left")
inv_turnover["turnover_rate"] = (
    inv_turnover["avg_monthly_sold"] / inv_turnover["total_units"] * 100
)
inv_turnover = inv_turnover.sort_values("turnover_rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))

palette_to = ["#2980b9" if v >= inv_turnover["turnover_rate"].median() else "#bdc3c7"
              for v in inv_turnover["turnover_rate"]]
axes[0].barh(inv_turnover.index, inv_turnover["turnover_rate"], color=palette_to)
axes[0].axvline(inv_turnover["turnover_rate"].median(), color="black",
                linestyle="--", linewidth=1.2,
                label=f"Median: {inv_turnover['turnover_rate'].median():.2f}%")
axes[0].set_title("Monthly Turnover Rate (%) by Category\n(avg monthly sold / total inventory)")
axes[0].set_xlabel("Turnover Rate (%/month)")
axes[0].legend()
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)

axes[1].barh(inv_turnover.index, inv_turnover["avg_monthly_sold"],
             color=sns.color_palette("Greens_r", len(inv_turnover)))
axes[1].set_title("Avg Units Sold per Month by Category")
axes[1].set_xlabel("Units / Month")
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)

save_fig("3_3_inventory_turnover.png",
         "Inventory Turnover: Which Categories Move Fastest?")

print("\nInventory turnover by category:")
print(inv_turnover[["total_units", "avg_monthly_sold", "turnover_rate"]].round(2).to_string())


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — USER BEHAVIOUR & FUNNEL
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("SECTION 4: USER BEHAVIOUR & FUNNEL")
print(DIVIDER)

# TheLook event_type sequence: home → department → product → cart → purchase
FUNNEL_STEPS = ["home", "department", "product", "cart", "purchase"]


# ── 4.1  Event type distribution ─────────────────────────────────────────────
print("\n[4.1] Event Type Distribution")

event_dist = (
    events_sp
    .groupBy("event_type")
    .agg(F.count("*").alias("count"))
    .orderBy(col("count").desc())
    .toPandas()
)
total_events = event_dist["count"].sum()
event_dist["pct"] = event_dist["count"] / total_events * 100

EVENT_COLORS = {
    "home":       "#3498db",
    "department": "#9b59b6",
    "product":    "#f39c12",
    "cart":       "#e67e22",
    "purchase":   "#2ecc71",
    "cancel":     "#e74c3c",
}
colors_ev = [EVENT_COLORS.get(e, "#95a5a6") for e in event_dist["event_type"]]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

bars = axes[0].bar(event_dist["event_type"], event_dist["count"], color=colors_ev)
axes[0].set_title("Event Count by Type")
axes[0].set_ylabel("Count")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].tick_params(axis="x", rotation=30)
for bar, pct in zip(bars, event_dist["pct"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

wedges, texts, autotexts = axes[1].pie(
    event_dist["pct"],
    labels=event_dist["event_type"],
    colors=colors_ev,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.82,
)
for at in autotexts:
    at.set_fontsize(9)
axes[1].set_title("Event Share (%)")

save_fig("4_1_event_type_dist.png", "Event Type Distribution")

print(event_dist.to_string(index=False))


# ── 4.2  Conversion funnel drop-off ──────────────────────────────────────────
print("\n[4.2] Conversion Funnel Drop-off")

funnel_counts = (
    events_sp
    .filter(col("event_type").isin(FUNNEL_STEPS))
    .groupBy("event_type")
    .agg(F.countDistinct("user_id").alias("unique_users"))
    .toPandas()
    .set_index("event_type")
    .reindex(FUNNEL_STEPS)
    .dropna()
)
funnel_counts["drop_rate"] = (
    1 - funnel_counts["unique_users"] / funnel_counts["unique_users"].shift(1)
) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

colors_funnel = ["#3498db", "#9b59b6", "#f39c12", "#e67e22", "#2ecc71"][:len(funnel_counts)]
bars = axes[0].bar(funnel_counts.index, funnel_counts["unique_users"],
                   color=colors_funnel, width=0.6)
axes[0].set_title("Unique Users per Funnel Stage")
axes[0].set_ylabel("Unique Users")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
top_val = funnel_counts["unique_users"].iloc[0]
for bar, val in zip(bars, funnel_counts["unique_users"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{val/top_val*100:.1f}%", ha="center", va="bottom", fontsize=9)

drop = funnel_counts["drop_rate"].dropna()
axes[1].bar(drop.index, drop.values,
            color=["#e74c3c" if v > 50 else "#f39c12" for v in drop.values],
            width=0.6)
axes[1].axhline(50, color="gray", linestyle="--", linewidth=1.2, label="50% drop line")
axes[1].set_title("Drop-off Rate between Funnel Stages (%)")
axes[1].set_ylabel("Drop-off (%)")
axes[1].legend()
for i, (stage, val) in enumerate(drop.items()):
    axes[1].text(i, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

save_fig("4_2_conversion_funnel.png",
         "Conversion Funnel: Where Do Users Drop Off?")

print("\nFunnel stage stats:")
print(funnel_counts.round(2).to_string())


# ── 4.3  Sessions per user & events per session ───────────────────────────────
print("\n[4.3] Session Depth Analysis")

session_stats = (
    events_sp
    .filter(col("session_id").isNotNull())
    .groupBy("user_id", "session_id")
    .agg(F.count("*").alias("events_in_session"))
    .groupBy("user_id")
    .agg(
        F.count("session_id").alias("sessions"),
        F.mean("events_in_session").alias("avg_events_per_session"),
    )
    .toPandas()
)

s_sample = session_stats.sample(n=min(50000, len(session_stats)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.histplot(s_sample["sessions"].clip(upper=s_sample["sessions"].quantile(0.99)),
             bins=40, kde=True, ax=axes[0], color="#3498db")
axes[0].axvline(session_stats["sessions"].mean(),   color="red",  linestyle="--",
                linewidth=1.5, label=f"Mean: {session_stats['sessions'].mean():.1f}")
axes[0].axvline(session_stats["sessions"].median(), color="green", linestyle="--",
                linewidth=1.5, label=f"Median: {session_stats['sessions'].median():.1f}")
axes[0].set_title("Sessions per User Distribution (capped at p99)")
axes[0].set_xlabel("Sessions")
axes[0].legend()

sns.histplot(s_sample["avg_events_per_session"].clip(
                 upper=s_sample["avg_events_per_session"].quantile(0.99)),
             bins=40, kde=True, ax=axes[1], color="#9b59b6")
axes[1].axvline(session_stats["avg_events_per_session"].mean(),   color="red",
                linestyle="--", linewidth=1.5,
                label=f"Mean: {session_stats['avg_events_per_session'].mean():.1f}")
axes[1].axvline(session_stats["avg_events_per_session"].median(), color="green",
                linestyle="--", linewidth=1.5,
                label=f"Median: {session_stats['avg_events_per_session'].median():.1f}")
axes[1].set_title("Avg Events per Session per User (capped at p99)")
axes[1].set_xlabel("Events / Session")
axes[1].legend()

save_fig("4_3_session_depth.png",
         "Session Depth: How Engaged Are Users?")

print(f"\n  Users analysed        : {len(session_stats):,}")
print(f"  Avg sessions/user     : {session_stats['sessions'].mean():.2f}")
print(f"  Median sessions/user  : {session_stats['sessions'].median():.2f}")
print(f"  Avg events/session    : {session_stats['avg_events_per_session'].mean():.2f}")


# ── 4.4  Daily event volume trend ────────────────────────────────────────────
print("\n[4.4] Daily Event Volume Trend")

daily_events = (
    events_sp
    .withColumn("date", date_format("created_at_ts", "yyyy-MM-dd"))
    .groupBy("date", "event_type")
    .agg(F.count("*").alias("count"))
    .orderBy("date")
    .toPandas()
)
daily_pivot = (
    daily_events
    .pivot(index="date", columns="event_type", values="count")
    .fillna(0)
)

fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

for etype in [e for e in FUNNEL_STEPS if e in daily_pivot.columns]:
    axes[0].plot(daily_pivot.index, daily_pivot[etype],
                 label=etype, linewidth=1.5,
                 color=EVENT_COLORS.get(etype, "#95a5a6"))
axes[0].set_title("Daily Event Count by Type (Funnel Steps)")
axes[0].set_ylabel("Events")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].legend(loc="upper left", fontsize=9)

total_daily = daily_events.groupby("date")["count"].sum().reset_index()
axes[1].fill_between(total_daily["date"], total_daily["count"], alpha=0.3, color="#2980b9")
axes[1].plot(total_daily["date"], total_daily["count"], color="#2980b9", linewidth=1.5)
axes[1].set_title("Total Daily Events (All Types)")
axes[1].set_ylabel("Total Events")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

n = max(1, len(total_daily) // 10)
axes[1].set_xticks(range(0, len(total_daily), n))
axes[1].set_xticklabels(total_daily["date"].iloc[::n], rotation=45, ha="right", fontsize=9)

save_fig("4_4_daily_event_volume.png",
         "Daily Event Volume: User Activity Trend Over Time")

print(f"\n  Total events         : {total_events:,}")
print(f"  Date range           : {total_daily['date'].min()} → {total_daily['date'].max()}")
print(f"  Avg events/day       : {total_daily['count'].mean():,.0f}")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — LOGISTICS: DISTRIBUTION CENTERS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("SECTION 5: LOGISTICS — DISTRIBUTION CENTERS")
print(DIVIDER)

# inventory_items has product_distribution_center_id → join dist_centers
# order_items has inventory_item_id → join inventory_items → join dist_centers
oi_dc_sp = (
    order_items_sp
    .join(
        inventory_sp.select(
            col("id").alias("inv_id"),
            col("product_distribution_center_id").alias("dc_id"),
        ),
        order_items_sp["inventory_item_id"] == col("inv_id"),
        how="left",
    )
    .join(
        dist_centers_sp.select(
            col("id").alias("dc_id_ref"),
            col("name").alias("dc_name"),
        ),
        col("dc_id") == col("dc_id_ref"),
        how="left",
    )
    .withColumn("is_returned", when(col("status") == "Returned", 1).otherwise(0))
    .withColumn("is_complete",  when(col("status") == "Complete",  1).otherwise(0))
    .filter(col("dc_name").isNotNull())
)


# ── 5.1  Order volume by distribution center ─────────────────────────────────
print("\n[5.1] Order Volume by Distribution Center")

dc_volume = (
    oi_dc_sp
    .groupBy("dc_name")
    .agg(
        F.count("*").alias("total_items"),
        F.countDistinct("order_id").alias("total_orders"),
        F.sum(when(col("is_complete") == 1, col("sale_price"))).alias("revenue"),
    )
    .orderBy(col("total_items").desc())
    .toPandas()
    .set_index("dc_name")
)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

axes[0].barh(dc_volume.index, dc_volume["total_items"],
             color=sns.color_palette("Blues_r", len(dc_volume)))
axes[0].set_title("Total Items Handled by Distribution Center")
axes[0].set_xlabel("Items")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)

dc_rev = dc_volume.sort_values("revenue", ascending=False)
axes[1].barh(dc_rev.index, dc_rev["revenue"],
             color=sns.color_palette("Greens_r", len(dc_rev)))
axes[1].set_title("Revenue Handled by Distribution Center")
axes[1].set_xlabel("Revenue ($)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K"))
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)

save_fig("5_1_dc_order_volume.png",
         "Distribution Center Load: Items & Revenue")

print("\nDC volume summary:")
print(dc_volume[["total_items", "total_orders", "revenue"]].round(0).to_string())


# ── 5.2  Delivery time by distribution center ────────────────────────────────
print("\n[5.2] Delivery Time by Distribution Center")

dc_delivery_sp = oi_dc_sp.filter(
    col("delivery_days").isNotNull() &
    (col("delivery_days") > 0) &
    (col("delivery_days") < 60)
)

dc_delivery_stats = (
    dc_delivery_sp
    .groupBy("dc_name")
    .agg(
        F.percentile_approx("delivery_days", 0.5).alias("median_days"),
        F.mean("delivery_days").alias("mean_days"),
        F.stddev("delivery_days").alias("std_days"),
        F.count("delivery_days").alias("count"),
    )
    .orderBy("median_days")
    .toPandas()
    .set_index("dc_name")
)

dc_delivery_pdf = (
    dc_delivery_sp.select("dc_name", "delivery_days").toPandas()
)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

dc_order = dc_delivery_stats.index.tolist()
sns.boxplot(
    data=dc_delivery_pdf, x="delivery_days", y="dc_name",
    order=dc_order, palette="coolwarm", ax=axes[0],
    showfliers=False, width=0.6,
)
global_med = dc_delivery_stats["median_days"].mean()
axes[0].axvline(global_med, color="red", linestyle="--", linewidth=1.5,
                label=f"Avg median: {global_med:.1f}d")
axes[0].set_title("Delivery Time by Distribution Center (Boxplot)")
axes[0].set_xlabel("Delivery Days")
axes[0].legend()
axes[0].yaxis.set_tick_params(labelsize=9)

axes[1].barh(dc_delivery_stats.index, dc_delivery_stats["median_days"],
             color=sns.color_palette("coolwarm", len(dc_delivery_stats)))
axes[1].set_title("Median Delivery Days by Distribution Center")
axes[1].set_xlabel("Median Days")
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)

save_fig("5_2_dc_delivery_time.png",
         "Distribution Center Speed: Which DCs Deliver Fastest?")

print("\nDelivery time by DC:")
print(dc_delivery_stats.round(2).to_string())


# ── 5.3  Return rate by distribution center ──────────────────────────────────
print("\n[5.3] Return Rate by Distribution Center")

dc_return = (
    oi_dc_sp
    .groupBy("dc_name")
    .agg(
        F.count("*").alias("total_items"),
        F.sum("is_returned").alias("returned_items"),
        F.sum("is_complete").alias("completed_items"),
    )
    .toPandas()
    .set_index("dc_name")
)
dc_return["return_rate"]   = dc_return["returned_items"] / dc_return["total_items"] * 100
dc_return["complete_rate"] = dc_return["completed_items"] / dc_return["total_items"] * 100
dc_return = dc_return.sort_values("return_rate", ascending=False)

merged_dc = dc_delivery_stats[["median_days"]].join(dc_return[["return_rate"]], how="inner")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

palette_dcr = ["#c0392b" if v >= dc_return["return_rate"].median() else "#27ae60"
               for v in dc_return["return_rate"]]
axes[0].barh(dc_return.index, dc_return["return_rate"], color=palette_dcr)
axes[0].axvline(dc_return["return_rate"].median(), color="black",
                linestyle="--", linewidth=1.2,
                label=f"Median: {dc_return['return_rate'].median():.1f}%")
axes[0].set_title("Return Rate (%) by Distribution Center")
axes[0].set_xlabel("Return Rate (%)")
axes[0].legend()
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)

pearson_dc = merged_dc["median_days"].corr(merged_dc["return_rate"])
axes[1].scatter(
    merged_dc["median_days"], merged_dc["return_rate"],
    s=300, c=range(len(merged_dc)), cmap="tab10",
    alpha=0.85, edgecolors="white", linewidths=1.5, zorder=5,
)
for dc, row in merged_dc.iterrows():
    axes[1].annotate(dc, (row["median_days"], row["return_rate"]),
                     textcoords="offset points", xytext=(7, 4), fontsize=8.5)
x_v = merged_dc["median_days"].values
y_v = merged_dc["return_rate"].values
m, b = np.polyfit(x_v, y_v, 1)
x_l  = np.linspace(x_v.min(), x_v.max(), 100)
axes[1].plot(x_l, m * x_l + b, color="red", linestyle="--",
             linewidth=1.5, label=f"Trend (r = {pearson_dc:.2f})")
axes[1].set_title("Delivery Speed vs Return Rate by DC\n(Ideal: fast delivery + low return → bottom-left)")
axes[1].set_xlabel("Median Delivery Days")
axes[1].set_ylabel("Return Rate (%)")
axes[1].legend()

save_fig("5_3_dc_return_rate.png",
         "Distribution Center Quality: Return Rate & Delivery Speed")

print("\nReturn rate by DC:")
print(dc_return[["total_items", "returned_items", "return_rate", "complete_rate"]].round(2).to_string())
print(f"\n  Pearson r (delivery days vs return rate across DCs): {pearson_dc:.3f}")


print(f"\n{DIVIDER}")
print("EDA COMPLETE — Charts saved:")
for f in [
    "1_1_monthly_revenue.png",
    "1_2_monthly_orders_aov.png",
    "1_3_order_status.png",
    "1_4_sale_price_dist.png",
    "1_5_user_age_dist.png",
    "2_1_gender_revenue.png",
    "2_2_age_purchase_freq.png",
    "2_3_traffic_conversion.png",
    "2_4_country_revenue.png",
    "2_5_country_return_rate.png",
    "3_1_inventory_sold_unsold.png",
    "3_2_time_to_sell_dist.png",
    "3_3_inventory_turnover.png",
    "4_1_event_type_dist.png",
    "4_2_conversion_funnel.png",
    "4_3_session_depth.png",
    "4_4_daily_event_volume.png",
    "5_1_dc_order_volume.png",
    "5_2_dc_delivery_time.png",
    "5_3_dc_return_rate.png",
"5_1_revenue_vs_quantity.png",
    "5_2_profit_margin.png",
    "5_3_return_rate.png",
    "5_4_double_risk_scatter.png",
    "5_5_category_gender.png",
    "6_1_traffic_aov.png",
    "6_2_traffic_return.png",
    "6_3_delivery_time.png",
    "6_4_delivery_return_corr.png"
]:
    print(f"  {f}")
print(DIVIDER)


print("\nDelivery vs Return Rate summary:")
print(corr_df[["median_delivery_days", "return_rate", "total_items"]].round(2))

if pearson_r > 0.4:
    print(f"\n  ➡️  r = {pearson_r:.2f}: Positive correlation — longer delivery → higher return rate.")
elif pearson_r < -0.4:
    print(f"\n  ➡️  r = {pearson_r:.2f}: Negative correlation — unexpected, investigate outliers.")
else:
    print(f"\n  ➡️  r = {pearson_r:.2f}: Weak correlation — delivery time alone may not explain returns.")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA COMPLETE — Charts saved:")
for f in [
    "5_1_revenue_vs_quantity.png",
    "5_2_profit_margin.png",
    "5_3_return_rate.png",
    "5_4_double_risk_scatter.png",
    "5_5_category_gender.png",
    "6_1_traffic_aov.png",
    "6_2_traffic_return.png",
    "6_3_delivery_time.png",
    "6_4_delivery_return_corr.png",
]:
    print(f"  {f}")
print("=" * 60)