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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, "media")

# Đảm bảo thư mục media tồn tại (tránh lỗi nếu lỡ xóa folder)
os.makedirs(MEDIA_DIR, exist_ok=True)


def save_fig(filename, suptitle=None):
    # Nối đường dẫn thư mục media với tên file ảnh
    full_path = os.path.join(MEDIA_DIR, filename)

    if suptitle:
        plt.suptitle(suptitle, **SUPTITLE_KW)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Lưu vào thư mục media
    plt.savefig(full_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"  [+] Đã lưu ảnh tại: media/{filename}")
# ─────────────────────────────────────────────
#  Khởi tạo Spark
# ─────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("TheLook_EDA_Full")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Lấy từng mảnh ghép từ .env và ghép lại thành URL hoàn chỉnh
HDFS_HOST     = os.getenv("HDFS_HOST", "localhost")
HDFS_PORT     = os.getenv("HDFS_PORT", "9000")
HDFS_BASE_DIR = os.getenv("HDFS_BASE_DIR", "/data/datatest")

# Bắt buộc phải nối thành chuỗi có chữ hdfs://
HDFS_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}{HDFS_BASE_DIR}"

def load(filename):
    path = f"{HDFS_BASE}/{filename}"
    print(f"  Loading {path} ...")  # In ra toàn bộ đường dẫn để dễ debug
    return (
        spark.read
        .option("header",      "true")
        .option("inferSchema", "true")
        .option("multiLine",   "true")
        .option("escape",      '"')
        .csv(path)
    )
print("Loading data from HDFS...")
orders_sp       = load("thelook_ecommerce.orders.csv")

# Gọi trực tiếp 2 file đã clean
order_items_sp  = load("order_items_cleaned.csv")
events_sp       = load("events_cleaned.csv")

users_sp        = load("thelook_ecommerce.users.csv")
products_sp     = load("thelook_ecommerce.products.csv")
inventory_sp    = load("thelook_ecommerce.inventory_items.csv")
dist_centers_sp = load("thelook_ecommerce.distribution_centers.csv")
print("Done.\n")

# Đăng ký TempView
orders_sp.createOrReplaceTempView("orders")
order_items_sp.createOrReplaceTempView("order_items")
users_sp.createOrReplaceTempView("users")
products_sp.createOrReplaceTempView("products")
inventory_sp.createOrReplaceTempView("inventory_items")
events_sp.createOrReplaceTempView("events")
dist_centers_sp.createOrReplaceTempView("distribution_centers")

print("TempViews registered.\n")
print(DIVIDER)


# =============================================================
#  Q1 — Doanh thu MoM + tăng trưởng theo danh mục
#       Kỹ thuật: JOIN · GROUP BY · LAG · SUM OVER · Time Series
# =============================================================
SEP("Q1 — Doanh thu MoM + tăng trưởng theo danh mục")

query_1 = """
SELECT
    category,
    order_month,
    monthly_revenue,
    LAG(monthly_revenue, 1) OVER (PARTITION BY category ORDER BY order_month)
        AS prev_month_revenue,
    ROUND(
        (monthly_revenue
         - LAG(monthly_revenue, 1) OVER (PARTITION BY category ORDER BY order_month))
        / LAG(monthly_revenue, 1) OVER (PARTITION BY category ORDER BY order_month) * 100,
    2) AS mom_growth_pct,
    SUM(monthly_revenue) OVER (
        PARTITION BY category ORDER BY order_month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM (
    SELECT
        p.category,
        DATE_FORMAT(TO_TIMESTAMP(oi.created_at), 'yyyy-MM') AS order_month,
        ROUND(SUM(oi.sale_price), 2)                        AS monthly_revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.status IN ('Complete', 'Shipped')
    GROUP BY p.category, DATE_FORMAT(TO_TIMESTAMP(oi.created_at), 'yyyy-MM')
) monthly_data
ORDER BY category, order_month
"""

df1 = spark.sql(query_1)
df1.show(10, truncate=False)
df1_pd = df1.toPandas()

# ── Visualisation Q1 ──────────────────────────────────────────
top_cats = (
    df1_pd.groupby("category")["monthly_revenue"].sum()
    .nlargest(5).index.tolist()
)
df1_top = df1_pd[df1_pd["category"].isin(top_cats)].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Doanh thu theo tháng (line)
for cat, grp in df1_top.groupby("category"):
    axes[0].plot(grp["order_month"], grp["monthly_revenue"], marker="o",
                 linewidth=1.8, label=cat)
axes[0].set_title("Doanh thu theo tháng — Top 5 danh mục")
axes[0].set_xlabel("Tháng")
axes[0].set_ylabel("Doanh thu ($)")
axes[0].tick_params(axis="x", rotation=45)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].legend(fontsize=9)

# Biểu đồ 2: MoM growth trung bình theo danh mục (bar)
mom_avg = (
    df1_pd.dropna(subset=["mom_growth_pct"])
    .groupby("category")["mom_growth_pct"].mean()
    .sort_values(ascending=False)
    .reset_index()
)
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in mom_avg["mom_growth_pct"]]
axes[1].bar(mom_avg["category"], mom_avg["mom_growth_pct"], color=colors)
axes[1].set_title("Tăng trưởng MoM trung bình theo danh mục")
axes[1].set_xlabel("Danh mục")
axes[1].set_ylabel("MoM Growth (%)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")

save_fig("q1_mom_growth.png", "Q1 — Doanh thu MoM & Tăng trưởng theo danh mục")


# =============================================================
#  Q2 — Top 5 khách hàng LTV cao nhất theo quốc gia
#       Kỹ thuật: JOIN 3 bảng · GROUP BY · RANK() · Subquery
# =============================================================
SEP("Q2 — Top 5 khách hàng LTV cao nhất theo quốc gia")

query_2 = """
SELECT * FROM (
    SELECT
        u.country,
        u.id                                        AS user_id,
        CONCAT(u.first_name, ' ', u.last_name)      AS full_name,
        u.gender,
        u.traffic_source,
        COUNT(DISTINCT o.order_id)                  AS total_orders,
        COUNT(oi.id)                                AS total_items,
        ROUND(SUM(oi.sale_price), 2)                AS total_spent,
        ROUND(AVG(oi.sale_price), 2)                AS avg_item_price,
        RANK() OVER (
            PARTITION BY u.country
            ORDER BY SUM(oi.sale_price) DESC
        )                                           AS rank_in_country
    FROM users u
    JOIN orders o       ON u.id = o.user_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE oi.status IN ('Complete', 'Shipped')
    GROUP BY u.country, u.id, u.first_name, u.last_name, u.gender, u.traffic_source
) ranked
WHERE rank_in_country <= 5
ORDER BY country, rank_in_country
"""

df2 = spark.sql(query_2)
df2.show(10, truncate=False)
df2_pd = df2.toPandas()

# ── Visualisation Q2 ──────────────────────────────────────────
top_countries = (
    df2_pd.groupby("country")["total_spent"].sum()
    .nlargest(6).index.tolist()
)
df2_top = df2_pd[df2_pd["country"].isin(top_countries)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Tổng chi tiêu Top-1 KH mỗi quốc gia
top1 = df2_top[df2_top["rank_in_country"] == 1].sort_values("total_spent", ascending=False)
axes[0].barh(top1["country"] + " — " + top1["full_name"],
             top1["total_spent"], color=sns.color_palette("Blues_d", len(top1)))
axes[0].set_title("Top-1 KH chi tiêu cao nhất mỗi quốc gia")
axes[0].set_xlabel("Tổng chi tiêu ($)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Biểu đồ 2: Phân phối total_spent top-5 theo quốc gia
for country, grp in df2_top.groupby("country"):
    axes[1].scatter(grp["rank_in_country"], grp["total_spent"], label=country, s=80)
axes[1].set_title("Phân phối chi tiêu Top 5 KH theo quốc gia")
axes[1].set_xlabel("Xếp hạng trong quốc gia")
axes[1].set_ylabel("Tổng chi tiêu ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].legend(fontsize=8, ncol=2)

save_fig("q2_ltv_top5.png", "Q2 — Top 5 khách hàng LTV cao nhất theo quốc gia")


# =============================================================
#  Q3 — Return rate theo brand vs trung bình ngành
#       Kỹ thuật: CTE 2 tầng · JOIN · Conditional AGG · Benchmark
# =============================================================
SEP("Q3 — Return rate theo brand vs trung bình ngành")

query_3 = """
WITH brand_stats AS (
    SELECT
        p.brand,
        p.category,
        COUNT(oi.id)                                                              AS total_sold,
        SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END)                  AS total_returned,
        ROUND(
            SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END)
            / COUNT(oi.id) * 100, 2
        )                                                                         AS return_rate_pct,
        ROUND(AVG(oi.sale_price), 2)                                              AS avg_sale_price
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE p.brand IS NOT NULL
    GROUP BY p.brand, p.category
),
category_avg AS (
    SELECT
        category,
        ROUND(AVG(return_rate_pct), 2) AS category_avg_return_rate
    FROM brand_stats
    GROUP BY category
)
SELECT
    bs.brand,
    bs.category,
    bs.total_sold,
    bs.total_returned,
    bs.return_rate_pct,
    ca.category_avg_return_rate,
    ROUND(bs.return_rate_pct - ca.category_avg_return_rate, 2) AS diff_from_avg,
    CASE
        WHEN bs.return_rate_pct > ca.category_avg_return_rate THEN 'Above Average'
        WHEN bs.return_rate_pct < ca.category_avg_return_rate THEN 'Below Average'
        ELSE 'At Average'
    END AS performance_flag
FROM brand_stats bs
JOIN category_avg ca ON bs.category = ca.category
ORDER BY bs.category, bs.return_rate_pct DESC
"""

df3 = spark.sql(query_3)
df3.show(10, truncate=False)
df3_pd = df3.toPandas()

# ── Visualisation Q3 ──────────────────────────────────────────
top5_cats = (
    df3_pd.groupby("category")["total_sold"].sum()
    .nlargest(5).index.tolist()
)
df3_top = df3_pd[df3_pd["category"].isin(top5_cats)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Phân phối diff_from_avg theo danh mục (box)
df3_top.boxplot(column="diff_from_avg", by="category", ax=axes[0],
                grid=False, patch_artist=True)
axes[0].set_title("Phân phối độ lệch return rate so với TB ngành")
axes[0].set_xlabel("Danh mục")
axes[0].set_ylabel("Diff from avg (%)")
axes[0].axhline(0, color="red", linewidth=1, linestyle="--")
plt.sca(axes[0])
plt.xticks(rotation=30)

# Biểu đồ 2: Above/Below Average stacked bar
flag_counts = (
    df3_top.groupby(["category", "performance_flag"])
    .size().unstack(fill_value=0)
)
flag_counts.plot(kind="bar", ax=axes[1],
                 color={"Above Average": "#e74c3c",
                        "At Average":    "#f39c12",
                        "Below Average": "#2ecc71"})
axes[1].set_title("Số brand Above / Below trung bình ngành")
axes[1].set_xlabel("Danh mục")
axes[1].set_ylabel("Số brand")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Performance", fontsize=9)

save_fig("q3_return_rate_brand.png", "Q3 — Return rate theo brand vs trung bình ngành")


# =============================================================
#  Q4 — Cohort Retention + avg spend theo tháng đăng ký
#       Kỹ thuật: CTE 3 tầng · JOIN · FIRST_VALUE · Time Series
# =============================================================
SEP("Q4 — Cohort Retention theo tháng đăng ký")

query_4 = """
WITH user_cohort AS (
    SELECT
        id AS user_id,
        DATE_FORMAT(TO_TIMESTAMP(created_at), 'yyyy-MM') AS cohort_month
    FROM users
),
user_orders AS (
    SELECT
        oi.user_id,
        DATE_FORMAT(TO_TIMESTAMP(oi.created_at), 'yyyy-MM') AS order_month,
        SUM(oi.sale_price)                                  AS monthly_spend
    FROM order_items oi
    WHERE oi.status IN ('Complete', 'Shipped')
    GROUP BY oi.user_id, DATE_FORMAT(TO_TIMESTAMP(oi.created_at), 'yyyy-MM')
),
cohort_data AS (
    SELECT
        uc.cohort_month,
        uo.order_month,
        COUNT(DISTINCT uo.user_id)      AS active_users,
        ROUND(SUM(uo.monthly_spend), 2) AS cohort_revenue,
        ROUND(AVG(uo.monthly_spend), 2) AS avg_spend_per_user
    FROM user_cohort uc
    JOIN user_orders uo ON uc.user_id = uo.user_id
    GROUP BY uc.cohort_month, uo.order_month
)
SELECT
    cohort_month,
    order_month,
    active_users,
    cohort_revenue,
    avg_spend_per_user,
    FIRST_VALUE(active_users) OVER (
        PARTITION BY cohort_month ORDER BY order_month
    ) AS cohort_size,
    ROUND(
        active_users * 100.0
        / FIRST_VALUE(active_users) OVER (PARTITION BY cohort_month ORDER BY order_month),
    2) AS retention_rate_pct
FROM cohort_data
ORDER BY cohort_month, order_month
"""

df4 = spark.sql(query_4)
df4.show(10, truncate=False)
df4_pd = df4.toPandas()
df4_pd["retention_rate_pct"] = df4_pd["retention_rate_pct"].astype(float)
df4_pd["avg_spend_per_user"] = df4_pd["avg_spend_per_user"].astype(float)

# ── Visualisation Q4 ──────────────────────────────────────────
df4_pd["month_index"] = df4_pd.groupby("cohort_month")["order_month"].transform(
    lambda x: (pd.to_datetime(x) - pd.to_datetime(x.min())).dt.days // 30
)

pivot = (
    df4_pd.pivot_table(index="cohort_month", columns="month_index",
                        values="retention_rate_pct", aggfunc="mean")
    .iloc[:, :7]
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Heatmap cohort retention
sns.heatmap(pivot, ax=axes[0], annot=True, fmt=".0f", cmap="YlGnBu",
            linewidths=0.4, cbar_kws={"label": "Retention (%)"})
axes[0].set_title("Cohort Retention Rate (%)")
axes[0].set_xlabel("Month Index (0 = tháng đầu mua)")
axes[0].set_ylabel("Cohort Month")

# Biểu đồ 2: Avg spend per user theo month index
avg_spend_pivot = (
    df4_pd.pivot_table(index="cohort_month", columns="month_index",
                        values="avg_spend_per_user", aggfunc="mean")
    .iloc[:, :6]
)
for col_name in avg_spend_pivot.columns:
    axes[1].plot(avg_spend_pivot.index, avg_spend_pivot[col_name],
                 marker="o", linewidth=1.5, label=f"M+{col_name}")
axes[1].set_title("Avg Spend per User theo Cohort Month")
axes[1].set_xlabel("Cohort Month")
axes[1].set_ylabel("Avg Spend ($)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend(title="Month Index", fontsize=8, ncol=3)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

save_fig("q4_cohort_retention.png", "Q4 — Cohort Retention & Avg Spend theo tháng đăng ký")


# =============================================================
#  Q5 — Phễu chuyển đổi theo kênh traffic source
#       Kỹ thuật: CTE · LEFT JOIN 3 · Conditional AGG · RANK()
# =============================================================
SEP("Q5 — Phễu chuyển đổi theo kênh traffic source")

query_5 = """
WITH traffic_funnel AS (
    SELECT
        u.traffic_source,
        COUNT(DISTINCT u.id)                                                      AS total_users,
        COUNT(DISTINCT o.order_id)                                                AS total_orders,
        COUNT(DISTINCT CASE WHEN o.status IN ('Complete','Shipped')
                            THEN o.order_id END)                                  AS completed_orders,
        COUNT(DISTINCT CASE WHEN o.status = 'Returned'
                            THEN o.order_id END)                                  AS returned_orders,
        ROUND(SUM(CASE WHEN oi.status IN ('Complete','Shipped')
                       THEN oi.sale_price ELSE 0 END), 2)                        AS total_revenue
    FROM users u
    LEFT JOIN orders o       ON u.id = o.user_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY u.traffic_source
)
SELECT
    traffic_source,
    total_users,
    total_orders,
    completed_orders,
    returned_orders,
    total_revenue,
    ROUND(total_orders * 100.0 / total_users, 2)                                  AS order_conversion_rate_pct,
    ROUND(completed_orders * 100.0 / NULLIF(total_orders, 0), 2)                  AS completion_rate_pct,
    ROUND(returned_orders  * 100.0 / NULLIF(total_orders, 0), 2)                  AS return_rate_pct,
    ROUND(total_revenue / NULLIF(total_users, 0), 2)                              AS revenue_per_user,
    RANK() OVER (ORDER BY total_revenue DESC)                                     AS revenue_rank
FROM traffic_funnel
ORDER BY total_revenue DESC
"""

df5 = spark.sql(query_5)
df5.show(truncate=False)
df5_pd = df5.toPandas()

# ── Visualisation Q5 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Funnel stages grouped bar
stages = ["total_users", "total_orders", "completed_orders"]
x      = np.arange(len(df5_pd))
width  = 0.25
colors = ["#3498db", "#2ecc71", "#9b59b6"]
labels = ["Total Users", "Total Orders", "Completed Orders"]

for i, (stage, color, label) in enumerate(zip(stages, colors, labels)):
    axes[0].bar(x + i * width, df5_pd[stage], width, label=label, color=color, alpha=0.85)

axes[0].set_title("Phễu chuyển đổi theo kênh traffic")
axes[0].set_xlabel("Traffic Source")
axes[0].set_ylabel("Số lượng")
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(df5_pd["traffic_source"], rotation=30)
axes[0].legend()
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Biểu đồ 2: Conversion rate & Revenue per user
ax2b = axes[1].twinx()
bars = axes[1].bar(df5_pd["traffic_source"], df5_pd["order_conversion_rate_pct"],
                   color="#3498db", alpha=0.7, label="Conversion Rate (%)")
line = ax2b.plot(df5_pd["traffic_source"], df5_pd["revenue_per_user"],
                 "o-", color="#e74c3c", linewidth=2, label="Revenue / User ($)")
axes[1].set_title("Conversion Rate & Revenue per User")
axes[1].set_xlabel("Traffic Source")
axes[1].set_ylabel("Conversion Rate (%)", color="#3498db")
ax2b.set_ylabel("Revenue per User ($)", color="#e74c3c")
axes[1].tick_params(axis="x", rotation=30)

lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

save_fig("q5_traffic_funnel.png", "Q5 — Phễu chuyển đổi theo kênh traffic source")


# =============================================================
#  Q6 — Thời gian giao hàng vs tỷ lệ hoàn trả theo DC
#       Kỹ thuật: CTE · JOIN 4 · PERCENTILE_APPROX · STDDEV · RANK OVER
# =============================================================
SEP("Q6 — Thời gian giao hàng vs tỷ lệ hoàn trả theo Distribution Center")

query_6 = """
WITH delivery_stats AS (
    SELECT
        dc.name                                                                    AS distribution_center,
        dc.id                                                                      AS dc_id,
        p.category,
        COUNT(oi.id)                                                               AS total_items,

        -- Tính khoảng cách số ngày giao hàng thực tế
        ROUND(AVG(DATEDIFF(TO_TIMESTAMP(oi.delivered_at), TO_TIMESTAMP(oi.created_at))), 2)             AS avg_delivery_days,
        ROUND(PERCENTILE_APPROX(DATEDIFF(TO_TIMESTAMP(oi.delivered_at), TO_TIMESTAMP(oi.created_at)), 0.5), 2) AS median_delivery_days,
        ROUND(STDDEV(DATEDIFF(TO_TIMESTAMP(oi.delivered_at), TO_TIMESTAMP(oi.created_at))), 2)          AS stddev_delivery_days,

        SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END)                   AS returns,
        ROUND(
            SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END) * 100.0
            / COUNT(oi.id), 2
        )                                                                          AS return_rate_pct
    FROM order_items oi
    JOIN inventory_items inv      ON oi.inventory_item_id = inv.id
    JOIN distribution_centers dc  ON inv.product_distribution_center_id = dc.id
    JOIN products p               ON oi.product_id = p.id

    -- Chỉ tính những đơn đã giao thành công (có ngày delivered_at)
    WHERE oi.delivered_at IS NOT NULL
    GROUP BY dc.name, dc.id, p.category
)
SELECT
    distribution_center,
    category,
    total_items,
    avg_delivery_days,
    median_delivery_days,
    stddev_delivery_days,
    return_rate_pct,
    AVG(avg_delivery_days) OVER (PARTITION BY distribution_center) AS dc_avg_delivery,
    AVG(return_rate_pct)   OVER (PARTITION BY distribution_center) AS dc_avg_return_rate,
    RANK() OVER (PARTITION BY distribution_center ORDER BY return_rate_pct DESC) AS return_rank_in_dc
FROM delivery_stats
ORDER BY distribution_center, return_rate_pct DESC
"""

df6 = spark.sql(query_6)
df6.show(10, truncate=False)
df6_pd = df6.toPandas()
df6_pd["avg_delivery_days"] = df6_pd["avg_delivery_days"].astype(float)
df6_pd["return_rate_pct"]   = df6_pd["return_rate_pct"].astype(float)
df6_pd["total_items"]       = df6_pd["total_items"].astype(float)
# ── Visualisation Q6 ──────────────────────────────────────────
dc_summary = (
    df6_pd.groupby("distribution_center")
    .agg(avg_delivery=("avg_delivery_days",  "mean"),
         avg_return   =("return_rate_pct",    "mean"),
         total_items  =("total_items",         "sum"))
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Scatter — avg_delivery vs return_rate (bubble size = total_items)
sc = axes[0].scatter(
    dc_summary["avg_delivery"], dc_summary["avg_return"],
    s=dc_summary["total_items"] / 20,
    c=range(len(dc_summary)), cmap="tab10", alpha=0.8
)
for _, row in dc_summary.iterrows():
    axes[0].annotate(row["distribution_center"].split(" ")[0],
                     (row["avg_delivery"], row["avg_return"]),
                     fontsize=8, ha="center", va="bottom")
axes[0].set_title("Avg Delivery Days vs Return Rate theo DC")
axes[0].set_xlabel("Avg Delivery Days")
axes[0].set_ylabel("Avg Return Rate (%)")

# Biểu đồ 2: Heatmap — return rate theo DC x category
pivot6 = df6_pd.pivot_table(index="distribution_center", columns="category",
                              values="return_rate_pct", aggfunc="mean")
sns.heatmap(pivot6, ax=axes[1], annot=True, fmt=".1f", cmap="Reds",
            linewidths=0.3, cbar_kws={"label": "Return Rate (%)"})
axes[1].set_title("Return Rate (%) theo DC × Danh mục")
axes[1].set_xlabel("Danh mục")
axes[1].set_ylabel("Distribution Center")
axes[1].tick_params(axis="x", rotation=30)

save_fig("q6_delivery_return.png", "Q6 — Delivery Time vs Return Rate theo Distribution Center")


# =============================================================
#  Q7 — RFM Segmentation khách hàng
#       Kỹ thuật: CTE 2 tầng · JOIN 3 · NTILE(5) · CASE WHEN
# =============================================================
SEP("Q7 — RFM Segmentation khách hàng")

query_7 = """
WITH rfm_raw AS (
    SELECT
        u.id                                              AS user_id,
        CONCAT(u.first_name, ' ', u.last_name)            AS full_name,
        u.country,
        u.gender,
        DATEDIFF(
    TO_DATE('2024-01-01'),
    MAX(TO_TIMESTAMP(oi.created_at))
) AS recency_days,
        COUNT(DISTINCT o.order_id)                         AS frequency,
        ROUND(SUM(oi.sale_price), 2)                       AS monetary
    FROM users u
    JOIN orders o       ON u.id = o.user_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE oi.status IN ('Complete', 'Shipped')
    GROUP BY u.id, u.first_name, u.last_name, u.country, u.gender
),
rfm_scored AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
        NTILE(5) OVER (ORDER BY frequency DESC)    AS f_score,
        NTILE(5) OVER (ORDER BY monetary DESC)     AS m_score
    FROM rfm_raw
)
SELECT
    user_id, full_name, country, gender,
    recency_days, frequency, monetary,
    r_score, f_score, m_score,
    (r_score + f_score + m_score) AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3                  THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'New Customers'
        WHEN r_score <= 2 AND f_score >= 3                  THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2                  THEN 'Lost'
        ELSE 'Potential Loyalists'
    END AS rfm_segment
FROM rfm_scored
ORDER BY rfm_total DESC
"""

df7 = spark.sql(query_7)
df7.show(10, truncate=False)
df7_pd = df7.toPandas()
df7_pd["monetary"] = df7_pd["monetary"].astype(float)
# ── Visualisation Q7 ──────────────────────────────────────────
seg_order  = ["Champions", "Loyal Customers", "Potential Loyalists",
              "New Customers", "At Risk", "Lost"]
seg_colors = ["#2ecc71", "#27ae60", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]

seg_stats = (
    df7_pd.groupby("rfm_segment")
    .agg(count=("user_id", "count"), avg_monetary=("monetary", "mean"))
    .reindex(seg_order).reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Pie chart số lượng KH theo segment
axes[0].pie(seg_stats["count"], labels=seg_stats["rfm_segment"],
            colors=seg_colors, autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2})
axes[0].set_title("Phân bố khách hàng theo RFM Segment")

# Biểu đồ 2: Avg monetary theo segment
bars = axes[1].bar(seg_stats["rfm_segment"], seg_stats["avg_monetary"],
                   color=seg_colors, edgecolor="white", linewidth=0.8)
axes[1].set_title("Avg Monetary theo RFM Segment")
axes[1].set_xlabel("Segment")
axes[1].set_ylabel("Avg Spend ($)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 5,
                 f"${bar.get_height():,.0f}",
                 ha="center", va="bottom", fontsize=8)

save_fig("q7_rfm_segmentation.png", "Q7 — RFM Segmentation khách hàng")

# ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  Hoàn thành toàn bộ 7 câu truy vấn & visualization.")
print(f"{DIVIDER}\n")