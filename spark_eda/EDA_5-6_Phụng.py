"""
EDA: Phân tích Sản phẩm & Kênh Marketing
Dataset: TheLook E-commerce (loaded from HDFS)

Section 5: Product Analysis
    5.1  Revenue vs Quantity by Category
    5.2  Profit Margin by Category
    5.3  Return Rate by Category
    5.4  Return Rate × Profit Margin (Double-Risk Scatter)
    5.5  Category × Gender

Section 6: Channel & Behavior Analysis
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

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, unix_timestamp, when

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})

SUPTITLE_KW = dict(fontsize=14, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: save figure with suptitle always visible
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(path, suptitle=None):
    if suptitle:
        plt.suptitle(suptitle, **SUPTITLE_KW)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  1. Spark session
# ─────────────────────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("TheLook_ECommerce_Analysis")
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
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(path)
    )

print("Loading data from HDFS...")
orders_sp      = load("orders")
order_items_sp = load("order_items")
users_sp       = load("users")
products_sp    = load("products")
print("Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Parse timestamps & derive columns
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
    .withColumn("delivery_days",
        (unix_timestamp("delivered_at") - unix_timestamp("shipped_at")) / 86400
    )
)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Build master DataFrame
# ─────────────────────────────────────────────────────────────────────────────
full_sp = (
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

completed_sp = full_sp.filter(col("is_complete") == 1)

print(f"Master DataFrame ready. Columns: {full_sp.columns}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PRODUCT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 5: PRODUCT ANALYSIS")
print("=" * 60)


# ── 5.1  Revenue vs Quantity by Category ─────────────────────────────────────
print("\n[5.1] Revenue vs Quantity by Category")

cat_rev = (
    completed_sp
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
    completed_sp
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
    full_sp
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
    completed_sp
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
#  SECTION 6 — CHANNEL & BEHAVIOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: CHANNEL & BEHAVIOR ANALYSIS")
print("=" * 60)


# ── 6.1  Traffic Source × AOV ────────────────────────────────────────────────
print("\n[6.1] Traffic Source × AOV")

traffic_aov = (
    completed_sp
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
    full_sp
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

delivered_sp = full_sp.filter(
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
    full_sp
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