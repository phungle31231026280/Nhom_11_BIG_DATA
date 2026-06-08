"""
EDA: TheLook E-commerce (loaded from HDFS)
File 02: Charts & Statistics — Vẽ biểu đồ và thống kê

Phần 1: Tổng quan kinh doanh (Đồ hoạ đơn biến)
    1.1  Monthly revenue trend
    1.2  Monthly orders & AOV trend
    1.3  Order status distribution
    1.4  Sale price distribution
    1.5  User age distribution

Phần 2: Phân tích khách hàng (Đồ hoạ đơn + đa biến)
    2.1  Gender × Revenue
    2.2  Age Group × Purchase Frequency
    2.3  Traffic Source → Conversion Rate
    2.4  Top Countries by Revenue
    2.5  Top Countries by Return Rate


Phần 5: Phân tích sản phẩm (Đồ hoạ đơn + đa biến)
    5.1  Revenue vs Quantity by Category
    5.2  Profit Margin by Category
    5.3  Return Rate by Category
    5.4  Return Rate × Profit Margin (Double-Risk Scatter)
    5.5  Category × Gender

Phần 6: Phân tích kênh & hành vi (Đồ hoạ đa biến)
    6.1  Traffic Source × AOV
    6.2  Traffic Source × Return Rate
    6.3  Delivery Time by Country & Distribution
    6.4  Delivery Time × Return Rate Correlation
"""

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

# full_sp for phần 5 (with products join)
full_sp_prod = (
    order_items_sp
    .join(
        products_sp.select(col("id").alias("prod_id"), "category", "cost", "department"),
        order_items_sp["product_id"] == col("prod_id"),
        how="left",
    )
    .join(
        users_sp.select(col("id").alias("user_id_u"), "gender", "country", "traffic_source"),
        order_items_sp["user_id"] == col("user_id_u"),
        how="left",
    )
    .withColumn("is_returned", when(col("status") == "Returned", 1).otherwise(0))
    .withColumn("is_complete",  when(col("status") == "Complete",  1).otherwise(0))
    .withColumn("profit",       col("sale_price") - col("cost"))
    .withColumn("margin_pct",
        when(col("sale_price") > 0,
             (col("profit") / col("sale_price")) * 100
        ).otherwise(None)
    )
)
completed_sp_prod = full_sp_prod.filter(col("is_complete") == 1)

# inventory join for phần 3
inv_full_sp = (
    inventory_sp
    .join(
        products_sp.select(col("id").alias("prod_id"), "category", "name"),
        inventory_sp["product_id"] == col("prod_id"),
        how="left",
    )
    .withColumn("is_sold", when(col("sold_at_ts").isNotNull(), 1).otherwise(0))
)

print(f"Master DataFrames ready.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  PHẦN 1 — TỔNG QUAN KINH DOANH (Đồ hoạ đơn biến)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("PHẦN 1: TỔNG QUAN KINH DOANH")
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

axes[0].bar(monthly_orders["month"], monthly_orders["orders"],
            color="#3498db", alpha=0.8)
axes[0].set_title("Monthly Order Count")
axes[0].set_ylabel("Orders")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

axes[1].plot(monthly_orders["month"], monthly_orders["AOV"],
             color="#e67e22", linewidth=2.5, marker="o", markersize=5)
axes[1].fill_between(monthly_orders["month"], monthly_orders["AOV"], alpha=0.2, color="#e67e22")
axes[1].set_title("Average Order Value (AOV)")
axes[1].set_ylabel("AOV ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
n = max(1, len(monthly_orders) // 10)
axes[1].set_xticks(range(0, len(monthly_orders), n))
axes[1].set_xticklabels(monthly_orders["month"].iloc[::n], rotation=45, ha="right", fontsize=9)

save_fig("1_2_monthly_orders_aov.png", "Monthly Orders & AOV Trend")

print(f"  Avg monthly orders : {monthly_orders['orders'].mean():,.0f}")
print(f"  Overall AOV        : ${monthly_orders['AOV'].mean():,.2f}")


# ── 1.3  Order status distribution ───────────────────────────────────────────
print("\n[1.3] Order Status Distribution")

status_dist = (
    order_items_sp
    .groupBy("status").agg(F.count("*").alias("count"))
    .orderBy(col("count").desc())
    .toPandas()
)
total_items = status_dist["count"].sum()
status_dist["pct"] = (status_dist["count"] / total_items * 100).round(2)

STATUS_COLORS = {
    "Complete":   "#2ecc71",
    "Shipped":    "#3498db",
    "Processing": "#f39c12",
    "Cancelled":  "#e74c3c",
    "Returned":   "#9b59b6",
}
colors_s = [STATUS_COLORS.get(s, "#95a5a6") for s in status_dist["status"]]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
bars = axes[0].bar(status_dist["status"], status_dist["count"], color=colors_s, width=0.6)
axes[0].set_title("Order Item Count by Status")
axes[0].set_ylabel("Count")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for bar, pct in zip(bars, status_dist["pct"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)

wedges, texts, autotexts = axes[1].pie(
    status_dist["pct"],
    labels=status_dist["status"],
    colors=colors_s,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.82,
)
for at in autotexts:
    at.set_fontsize(9)
axes[1].set_title("Order Status Share (%)")

save_fig("1_3_order_status.png", "Order Status Distribution")
print(status_dist.to_string(index=False))


# ── 1.4  Sale price distribution ─────────────────────────────────────────────
print("\n[1.4] Sale Price Distribution")

price_stats_row = order_items_sp.agg(
    F.min("sale_price").alias("min"),
    F.max("sale_price").alias("max"),
    F.mean("sale_price").alias("mean"),
    F.percentile_approx("sale_price", [0.5, 0.75, 0.9, 0.95, 0.99]).alias("pctiles"),
).toPandas().iloc[0]
price_stats = {
    "min":  price_stats_row["min"],
    "max":  price_stats_row["max"],
    "mean": price_stats_row["mean"],
}
p50, p75, p90, p95, p99 = price_stats_row["pctiles"]

price_sample = (
    order_items_sp.select("sale_price").filter(col("sale_price").isNotNull())
    .sample(fraction=min(1.0, 100000 / order_items_sp.count()))
    .toPandas()
)

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
#  PHẦN 2 — PHÂN TÍCH KHÁCH HÀNG (Đồ hoạ đơn + đa biến)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("PHẦN 2: PHÂN TÍCH KHÁCH HÀNG")
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


# ═════════════════════════════════════════════════════════════════════════════
#  PHẦN 5 — PHÂN TÍCH SẢN PHẨM (Đồ hoạ đơn + đa biến)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PHẦN 5: PHÂN TÍCH SẢN PHẨM")
print("=" * 60)


# ── 5.1  Revenue vs Quantity by Category ─────────────────────────────────────
print("\n[5.1] Revenue vs Quantity by Category")

cat_rev = (
    completed_sp_prod
    .groupBy("category")
    .agg(
        F.sum("sale_price").alias("revenue"),
        F.count("sale_price").alias("quantity"),
    )
    .orderBy(col("revenue").desc())
    .toPandas()
    .set_index("category")
)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))

axes[0].barh(cat_rev.index, cat_rev["revenue"],
             color=sns.color_palette("Blues_r", len(cat_rev)))
axes[0].set_title("Total Revenue by Category", pad=10)
axes[0].set_xlabel("Revenue ($)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)

cat_qty = cat_rev.sort_values("quantity", ascending=False)
axes[1].barh(cat_qty.index, cat_qty["quantity"],
             color=sns.color_palette("Oranges_r", len(cat_qty)))
axes[1].set_title("Total Quantity Sold by Category", pad=10)
axes[1].set_xlabel("Units Sold")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)

save_fig("5_1_revenue_vs_quantity.png",
         "Revenue vs Quantity: Same Category? Different Story?")

cat_rev["rev_rank"]  = cat_rev["revenue"].rank(ascending=False)
cat_rev["qty_rank"]  = cat_rev["quantity"].rank(ascending=False)
cat_rev["rank_diff"] = (cat_rev["qty_rank"] - cat_rev["rev_rank"]).abs()
print("\nCategories with biggest rank gap (high volume ≠ high revenue):")
print(cat_rev[["revenue", "quantity", "rev_rank", "qty_rank", "rank_diff"]]
      .sort_values("rank_diff", ascending=False).head(5))


# ── 5.2  Profit Margin by Category ───────────────────────────────────────────
print("\n[5.2] Profit Margin by Category")

cat_margin = (
    completed_sp_prod
    .groupBy("category")
    .agg(
        F.mean("margin_pct").alias("avg_margin"),
        F.sum("profit").alias("total_profit"),
        F.mean("sale_price").alias("avg_sale_price"),
        F.mean("cost").alias("avg_cost"),
    )
    .orderBy(col("avg_margin").desc())
    .toPandas()
    .set_index("category")
)

margin_med_global = cat_margin["avg_margin"].median()
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

palette_margin = ["#2ecc71" if v >= margin_med_global else "#e74c3c"
                  for v in cat_margin["avg_margin"]]
axes[0].barh(cat_margin.index, cat_margin["avg_margin"], color=palette_margin)
axes[0].axvline(margin_med_global, color="black", linestyle="--", linewidth=1.2,
                label=f"Median: {margin_med_global:.1f}%")
axes[0].set_title("Average Profit Margin % by Category", pad=10)
axes[0].set_xlabel("Margin (%)")
axes[0].legend()
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)

cat_profit_sorted = cat_margin.sort_values("total_profit", ascending=False)
axes[1].barh(cat_profit_sorted.index, cat_profit_sorted["total_profit"],
             color=sns.color_palette("Greens_r", len(cat_profit_sorted)))
axes[1].set_title("Total Profit ($) by Category", pad=10)
axes[1].set_xlabel("Total Profit ($)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)

save_fig("5_2_profit_margin.png",
         "Profit Margin Analysis: High Margin vs High Profit Volume")

print("\nTop 5 categories by avg margin:")
print(cat_margin[["avg_margin", "total_profit", "avg_sale_price", "avg_cost"]].head(5).round(2))


# ── 5.3  Return Rate by Category ─────────────────────────────────────────────
print("\n[5.3] Return Rate by Category")

cat_return = (
    full_sp_prod
    .groupBy("category")
    .agg(
        F.count("status").alias("total_items"),
        F.sum("is_returned").alias("returned_items"),
    )
    .toPandas()
    .set_index("category")
)
cat_return["return_rate"] = cat_return["returned_items"] / cat_return["total_items"] * 100
cat_return = cat_return.sort_values("return_rate", ascending=False)

return_med_global = cat_return["return_rate"].median()
fig, axes = plt.subplots(1, 2, figsize=(24, 11))

palette_ret = ["#c0392b" if v >= return_med_global else "#3498db"
               for v in cat_return["return_rate"]]
axes[0].barh(cat_return.index, cat_return["return_rate"], color=palette_ret)
axes[0].axvline(return_med_global, color="black", linestyle="--", linewidth=1.2,
                label=f"Median: {return_med_global:.1f}%")
axes[0].set_title("Return Rate (%) by Category", pad=10)
axes[0].set_xlabel("Return Rate (%)")
axes[0].legend()
axes[0].invert_yaxis()
axes[0].yaxis.set_tick_params(labelsize=9)
axes[0].tick_params(axis="y", pad=6)

cat_return_vol = cat_return.sort_values("returned_items", ascending=False)
axes[1].barh(cat_return_vol.index, cat_return_vol["returned_items"],
             color=sns.color_palette("Reds_r", len(cat_return_vol)))
axes[1].set_title("Absolute Return Volume by Category", pad=10)
axes[1].set_xlabel("Number of Returned Items")
axes[1].invert_yaxis()
axes[1].yaxis.set_tick_params(labelsize=9)
axes[1].tick_params(axis="y", pad=6)

plt.subplots_adjust(left=0.22, right=0.97, wspace=0.45, top=0.90, bottom=0.07)
plt.suptitle("Return Analysis: Rate vs Volume — Not Always the Same Problem",
             **SUPTITLE_KW)
plt.savefig("5_3_return_rate.png", bbox_inches="tight", dpi=150)
plt.show()

print("\nTop 5 categories by return rate:")
print(cat_return[["return_rate", "returned_items", "total_items"]].head(5).round(2))


# ── 5.4  Return Rate × Profit Margin — Double-Risk Scatter ───────────────────
print("\n[5.4] Return Rate × Profit Margin (Double-Risk Analysis)")

risk_df = cat_margin[["avg_margin", "total_profit"]].join(
    cat_return[["return_rate", "total_items"]], how="inner"
)

margin_med = risk_df["avg_margin"].median()
return_med = risk_df["return_rate"].median()

def quadrant_color(row):
    hi_ret = row["return_rate"] >= return_med
    lo_mar = row["avg_margin"]  <  margin_med
    if   hi_ret and     lo_mar: return "#e74c3c"  # double risk
    elif hi_ret and not lo_mar: return "#e67e22"  # risky revenue
    elif not hi_ret and lo_mar: return "#3498db"  # low value
    else:                       return "#27ae60"  # ideal

colors_risk = [quadrant_color(r) for _, r in risk_df.iterrows()]
size_scale  = (risk_df["total_items"] / risk_df["total_items"].max()) * 1200 + 100

fig, ax = plt.subplots(figsize=(16, 9))

ax.scatter(
    risk_df["avg_margin"], risk_df["return_rate"],
    s=size_scale, c=colors_risk,
    alpha=0.80, edgecolors="white", linewidths=1.2, zorder=5,
)
for cat, row in risk_df.iterrows():
    ax.annotate(cat, (row["avg_margin"], row["return_rate"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8.5, color="#2c3e50")

ax.axvline(margin_med, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax.axhline(return_med, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

y_rng = risk_df["return_rate"].max() - risk_df["return_rate"].min()
x_rng = risk_df["avg_margin"].max()  - risk_df["avg_margin"].min()
ax.set_ylim(risk_df["return_rate"].min() - y_rng * 0.18,
            risk_df["return_rate"].max() + y_rng * 0.18)
ax.set_xlim(risk_df["avg_margin"].min()  - x_rng * 0.05,
            risk_df["avg_margin"].max()  + x_rng * 0.05)

quadrant_labels = [
    (0.97, 0.05, "right", "bottom", "#27ae60", "✅ Ideal\n(High margin, Low return)"),
    (0.03, 0.05, "left",  "bottom", "#3498db", "⚠️ Low value\n(Low margin, Low return)"),
    (0.97, 0.95, "right", "top",    "#e67e22", "🔶 Risky revenue\n(High margin, High return)"),
    (0.03, 0.95, "left",  "top",    "#e74c3c", "🚨 Double Risk\n(Low margin, High return)"),
]
for x, y, ha, va, color, label in quadrant_labels:
    ax.text(x, y, label, ha=ha, va=va, fontsize=9, color=color,
            fontweight="bold", transform=ax.transAxes)

ax.set_title(
    "Return Rate × Profit Margin: Identifying Double-Risk Categories\n"
    "(Bubble size = total order volume)",
    fontsize=14, fontweight="bold", pad=16,
)
ax.set_xlabel("Average Profit Margin (%)", fontsize=11)
ax.set_ylabel("Return Rate (%)", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("5_4_double_risk_scatter.png", bbox_inches="tight", dpi=150)
plt.show()

double_risk = risk_df[
    (risk_df["return_rate"] >= return_med) & (risk_df["avg_margin"] < margin_med)
].sort_values("return_rate", ascending=False)
print("\n🚨 Double-Risk categories (High return + Low margin):")
print(double_risk[["avg_margin", "return_rate", "total_items"]].round(2))


# ── 5.5  Category × Gender ───────────────────────────────────────────────────
print("\n[5.5] Category × Gender")

cat_gender_sp = (
    completed_sp_prod
    .filter(col("gender").isin("M", "F"))
    .groupBy("category", "gender")
    .agg(F.sum("sale_price").alias("revenue"))
    .toPandas()
)

cat_gender = (
    cat_gender_sp
    .pivot(index="category", columns="gender", values="revenue")
    .fillna(0)
)
cat_gender["total"] = cat_gender.sum(axis=1)
cat_gender = cat_gender.sort_values("total", ascending=False).drop(columns="total")
cat_gender_pct = cat_gender.div(cat_gender.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

cat_gender.plot(kind="barh", stacked=True, ax=axes[0],
                color={"M": "#3498db", "F": "#e91e8c"})
axes[0].set_title("Revenue by Category & Gender (Absolute)")
axes[0].set_xlabel("Revenue ($)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].invert_yaxis()
axes[0].legend(title="Gender")

sns.heatmap(
    cat_gender_pct, ax=axes[1],
    annot=True, fmt=".1f", cmap="RdBu", center=50,
    linewidths=0.5, cbar_kws={"label": "% of Revenue"},
)
axes[1].set_title("Gender Revenue Split (%) — Blue = Male, Red = Female")
axes[1].set_ylabel("")

save_fig("5_5_category_gender.png",
         "Category × Gender: Which Categories Are Gender-Dominant?")

print("\nGender split % per category:")
print(cat_gender_pct.round(1))


# ═════════════════════════════════════════════════════════════════════════════
#  PHẦN 6 — PHÂN TÍCH KÊNH & HÀNH VI (Đồ hoạ đa biến)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHẦN 6: PHÂN TÍCH KÊNH & HÀNH VI")
print("=" * 60)


# ── 6.1  Traffic Source × AOV ────────────────────────────────────────────────
print("\n[6.1] Traffic Source × AOV")

traffic_aov = (
    completed_sp_prod
    .groupBy("traffic_source")
    .agg(
        F.sum("sale_price").alias("total_revenue"),
        F.countDistinct("order_id").alias("total_orders"),
        F.countDistinct("user_id").alias("total_users"),
    )
    .toPandas()
    .set_index("traffic_source")
)
traffic_aov["AOV"]              = traffic_aov["total_revenue"] / traffic_aov["total_orders"]
traffic_aov["revenue_per_user"] = traffic_aov["total_revenue"] / traffic_aov["total_users"]
traffic_aov = traffic_aov.sort_values("AOV", ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].bar(traffic_aov.index, traffic_aov["AOV"],
            color=sns.color_palette("viridis", len(traffic_aov)))
axes[0].axhline(traffic_aov["AOV"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean AOV: ${traffic_aov['AOV'].mean():.0f}")
axes[0].set_title("Average Order Value (AOV) by Traffic Source")
axes[0].set_ylabel("AOV ($)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
axes[0].legend()
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(traffic_aov.index, traffic_aov["revenue_per_user"],
            color=sns.color_palette("plasma", len(traffic_aov)))
axes[1].set_title("Revenue per User by Traffic Source")
axes[1].set_ylabel("Revenue / User ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
axes[1].tick_params(axis="x", rotation=30)

axes[2].bar(traffic_aov.index, traffic_aov["total_users"],
            color=sns.color_palette("crest", len(traffic_aov)))
axes[2].set_title("Total Unique Users by Traffic Source")
axes[2].set_ylabel("Users")
axes[2].tick_params(axis="x", rotation=30)

save_fig("6_1_traffic_aov.png",
         "Traffic Source Quality: AOV, Revenue/User, Volume")

print("\nTraffic source summary:")
print(traffic_aov[["AOV", "revenue_per_user", "total_users", "total_revenue"]].round(2))


# ── 6.2  Traffic Source × Return Rate ────────────────────────────────────────
print("\n[6.2] Traffic Source × Return Rate")

traffic_return = (
    full_sp_prod
    .filter(col("traffic_source").isNotNull())
    .groupBy("traffic_source")
    .agg(
        F.count("status").alias("total_items"),
        F.sum("is_returned").alias("returned"),
    )
    .toPandas()
    .set_index("traffic_source")
)
traffic_return["return_rate"] = traffic_return["returned"] / traffic_return["total_items"] * 100
traffic_return = traffic_return.sort_values("return_rate", ascending=False)

merged_traffic = traffic_aov[["AOV"]].join(traffic_return[["return_rate"]], how="inner")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

palette_rr = ["#e74c3c" if v >= merged_traffic["return_rate"].median() else "#27ae60"
              for v in merged_traffic["return_rate"]]
axes[0].bar(merged_traffic.index, merged_traffic["return_rate"], color=palette_rr)
axes[0].axhline(merged_traffic["return_rate"].median(), color="black", linestyle="--",
                linewidth=1.2, label=f"Median: {merged_traffic['return_rate'].median():.1f}%")
axes[0].set_title("Return Rate (%) by Traffic Source")
axes[0].set_ylabel("Return Rate (%)")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=30)

axes[1].scatter(merged_traffic["AOV"], merged_traffic["return_rate"],
                s=200, c=range(len(merged_traffic)), cmap="tab10", zorder=5)
for src, row in merged_traffic.iterrows():
    axes[1].annotate(src, (row["AOV"], row["return_rate"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=10)
axes[1].axvline(merged_traffic["AOV"].mean(),         color="gray", linestyle="--", linewidth=1)
axes[1].axhline(merged_traffic["return_rate"].mean(), color="gray", linestyle="--", linewidth=1)
axes[1].set_title("AOV vs Return Rate by Traffic Source\n(Ideal: high AOV + low return → bottom-right)")
axes[1].set_xlabel("AOV ($)")
axes[1].set_ylabel("Return Rate (%)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))

save_fig("6_2_traffic_return.png",
         "Traffic Source: Which Channel Brings Quality Customers?")

print("\nReturn rate by traffic source:")
print(traffic_return[["return_rate", "returned", "total_items"]].round(2))


# ── 6.3  Delivery Time Analysis ──────────────────────────────────────────────
print("\n[6.3] Delivery Time Analysis")

delivered_sp = full_sp_prod.filter(
    col("delivery_days").isNotNull() &
    (col("delivery_days") > 0) &
    (col("delivery_days") < 60)
)

deliv_stats = delivered_sp.agg(
    F.count("delivery_days").alias("n"),
    F.mean("delivery_days").alias("mean"),
    F.percentile_approx("delivery_days", 0.5).alias("median"),
).toPandas().iloc[0]

print(f"  Valid delivered rows : {int(deliv_stats['n']):,}")
print(f"  Mean delivery days   : {deliv_stats['mean']:.1f}")
print(f"  Median delivery days : {deliv_stats['median']:.1f}")

top_countries = (
    delivered_sp.groupBy("country").count()
    .orderBy(col("count").desc()).limit(12)
    .toPandas()["country"].tolist()
)

delivered_pdf = (
    delivered_sp
    .filter(col("country").isin(top_countries))
    .select("country", "delivery_days")
    .toPandas()
)

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

country_order = (
    delivered_pdf.groupby("country")["delivery_days"]
    .median().sort_values().index.tolist()
)
sns.boxplot(data=delivered_pdf, x="delivery_days", y="country",
            order=country_order, palette="coolwarm", ax=axes[0],
            showfliers=False, width=0.6)
axes[0].axvline(deliv_stats["median"], color="red", linestyle="--", linewidth=1.5,
                label=f"Global median: {deliv_stats['median']:.1f}d")
axes[0].set_title("Delivery Time by Country (Top 12 by volume)", pad=10)
axes[0].set_xlabel("Delivery Days")
axes[0].legend()
axes[0].yaxis.set_tick_params(labelsize=9)

deliv_sample = (
    delivered_sp.select("delivery_days")
    .sample(fraction=min(1.0, 50000 / delivered_sp.count()))
    .toPandas()
)
sns.histplot(deliv_sample["delivery_days"], bins=40, kde=True, ax=axes[1], color="steelblue")
axes[1].axvline(deliv_stats["mean"],   color="red",   linestyle="--", linewidth=1.5,
                label=f"Mean: {deliv_stats['mean']:.1f}d")
axes[1].axvline(deliv_stats["median"], color="green", linestyle="--", linewidth=1.5,
                label=f"Median: {deliv_stats['median']:.1f}d")
axes[1].set_title("Overall Delivery Time Distribution", pad=10)
axes[1].set_xlabel("Delivery Days")
axes[1].legend()

plt.subplots_adjust(left=0.15, right=0.97, wspace=0.35, top=0.90, bottom=0.08)
plt.suptitle("Logistics Performance: Delivery Time Analysis", **SUPTITLE_KW)
plt.savefig("6_3_delivery_time.png", bbox_inches="tight", dpi=150)
plt.show()

country_stats = (
    delivered_sp
    .filter(col("country").isin(top_countries))
    .groupBy("country")
    .agg(
        F.percentile_approx("delivery_days", 0.5).alias("median"),
        F.mean("delivery_days").alias("mean"),
        F.stddev("delivery_days").alias("std"),
        F.count("delivery_days").alias("count"),
    )
    .orderBy("median")
    .toPandas()
    .set_index("country")
    .round(2)
)
print("\nDelivery stats by country (sorted by median):")
print(country_stats)


# ── 6.4  Delivery Time × Return Rate Correlation ─────────────────────────────
print("\n[6.4] Delivery Time × Return Rate Correlation by Country")

country_return_sp = (
    full_sp_prod
    .filter(col("country").isin(top_countries))
    .groupBy("country")
    .agg(
        F.count("status").alias("total_items"),
        F.sum("is_returned").alias("returned_items"),
    )
    .toPandas()
    .set_index("country")
)
country_return_sp["return_rate"] = (
    country_return_sp["returned_items"] / country_return_sp["total_items"] * 100
)

corr_df = country_stats[["median", "mean", "std"]].join(
    country_return_sp[["return_rate", "total_items"]], how="inner"
)
corr_df.rename(columns={"median": "median_delivery_days"}, inplace=True)

pearson_r = corr_df["median_delivery_days"].corr(corr_df["return_rate"])
print(f"  Pearson r (delivery days vs return rate): {pearson_r:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

size_s         = (corr_df["total_items"] / corr_df["total_items"].max()) * 600 + 80
scatter_colors = sns.color_palette("tab10", len(corr_df))

axes[0].scatter(corr_df["median_delivery_days"], corr_df["return_rate"],
                s=size_s, c=scatter_colors, alpha=0.85,
                edgecolors="white", linewidths=1, zorder=5)
for country, row in corr_df.iterrows():
    axes[0].annotate(country,
                     (row["median_delivery_days"], row["return_rate"]),
                     textcoords="offset points", xytext=(7, 3), fontsize=9)

x_vals = corr_df["median_delivery_days"].values
y_vals = corr_df["return_rate"].values
m, b   = np.polyfit(x_vals, y_vals, 1)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
axes[0].plot(x_line, m * x_line + b, color="red", linestyle="--",
             linewidth=1.5, label=f"Trend (r = {pearson_r:.2f})")
axes[0].set_title("Delivery Time (Median) vs Return Rate by Country\n(Bubble size = order volume)", pad=10)
axes[0].set_xlabel("Median Delivery Days")
axes[0].set_ylabel("Return Rate (%)")
axes[0].legend()

corr_sorted = corr_df.sort_values("median_delivery_days")
x_idx = np.arange(len(corr_sorted))

ax2_bar  = axes[1]
ax2_line = ax2_bar.twinx()

ax2_bar.bar(x_idx, corr_sorted["median_delivery_days"],
            width=0.6, color="#5dade2", alpha=0.75, label="Median Delivery Days")
ax2_line.plot(x_idx, corr_sorted["return_rate"],
              color="#e74c3c", marker="o", linewidth=2, markersize=7, label="Return Rate (%)")

ax2_bar.set_xticks(x_idx)
ax2_bar.set_xticklabels(corr_sorted.index, rotation=35, ha="right", fontsize=9)
ax2_bar.set_ylabel("Median Delivery Days", color="#5dade2", fontsize=10)
ax2_line.set_ylabel("Return Rate (%)",     color="#e74c3c", fontsize=10)
ax2_bar.set_title("Delivery Days & Return Rate by Country\n(Sorted by delivery speed)", pad=10)

h1, l1 = ax2_bar.get_legend_handles_labels()
h2, l2 = ax2_line.get_legend_handles_labels()
ax2_bar.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

save_fig("6_4_delivery_return_corr.png",
         f"Logistics → Quality: Does Slower Delivery Drive More Returns?  (r = {pearson_r:.2f})")

print("\nDelivery vs Return Rate summary:")
print(corr_df[["median_delivery_days", "return_rate", "total_items"]].round(2))

if pearson_r > 0.4:
    print(f"\n  ➡️  r = {pearson_r:.2f}: Positive correlation — longer delivery → higher return rate.")
elif pearson_r < -0.4:
    print(f"\n  ➡️  r = {pearson_r:.2f}: Negative correlation — unexpected, investigate outliers.")
else:
    print(f"\n  ➡️  r = {pearson_r:.2f}: Weak correlation — delivery time alone may not explain returns.")


# ─────────────────────────────────────────────────────────────────────────────
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
print(DIVIDER)