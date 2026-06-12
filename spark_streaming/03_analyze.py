import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HDFS_HOST      = os.getenv("HDFS_HOST",             "localhost")
HDFS_PORT      = os.getenv("HDFS_PORT",             "9000")
HDFS_STREAMING_DIR = os.getenv("HDFS_STREAMING_DIR", "/data/streaming")
HDFS_DATATEST_DIR  = os.getenv("HDFS_DATATEST_DIR",  "/data/datatest")
STREAM_OUT_DIR     = f"{HDFS_URI}{HDFS_STREAMING_DIR}"
DATATEST_OUT_DIR   = f"{HDFS_URI}{HDFS_DATATEST_DIR}"

REFRESH_SEC = int(os.getenv("DASHBOARD_REFRESH_SEC", "10"))
MAX_HISTORY = 30

# ──────────────────────────────────────────────────────────────────────────────
# THEME
# ──────────────────────────────────────────────────────────────────────────────
BG      = "#f1f5f9"
PANEL   = "#ffffff"
C1      = "#2563eb"   # xanh dương
C2      = "#f97316"   # cam
C3      = "#16a34a"   # xanh lá
C4      = "#dc2626"   # đỏ
C5      = "#7c3aed"   # tím
TXT     = "#1e293b"
GRID    = "#e2e8f0"
BORDER  = "#cbd5e1"
YELL    = "#d97706"
CAT_CLR = [C1, C2, C3, C4, C5, "#0891b2", "#059669", "#db2777", "#65a30d", "#ea580c"]

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TXT,
    "axes.titlecolor":   TXT,
    "xtick.color":       TXT,
    "ytick.color":       TXT,
    "text.color":        TXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    8.5,
    "axes.titleweight":  "bold",
})

# ──────────────────────────────────────────────────────────────────────────────
# SPARK SINGLETON
# ──────────────────────────────────────────────────────────────────────────────
_spark = None

def get_spark():
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession
        _spark = (
            SparkSession.builder
            .appName("Dashboard_Reader")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}")
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("ERROR")
        print(f"[DASHBOARD] SparkSession OK (Stream: {STREAM_OUT_DIR}, DataTest: {DATATEST_OUT_DIR})")
    return _spark


# ──────────────────────────────────────────────────────────────────────────────
# HDFS READER
# ──────────────────────────────────────────────────────────────────────────────
def read_parquet(stream_name: str) -> pd.DataFrame:
    """
    Đọc Parquet từ HDFS, trả về pandas DataFrame.
    inventory_alert dùng foreachBatch → ghi ra batch_* subdirs.
    Dùng recursive=True để đọc tất cả sub-partition nếu có.
    """
    try:
        spark = get_spark()
        # Phân luồng thư mục đọc dữ liệu
        if stream_name in ["order_anomaly", "traffic_funnel"]:
            base_dir = STREAM_OUT_DIR
        else:
            base_dir = DATATEST_OUT_DIR
            
        if stream_name == "inventory_alert":
            # foreachBatch ghi batch_0, batch_1,... → đọc toàn bộ dir
            path = f"{base_dir}/{stream_name}"
            sdf = (spark.read
                   .option("mergeSchema", "true")
                   .option("recursiveFileLookup", "true")
                   .parquet(path))
        else:
            path = f"{base_dir}/{stream_name}"
            sdf = (spark.read
                   .option("mergeSchema", "true")
                   .option("recursiveFileLookup", "true")
                   .parquet(path))

        # Bỏ count() tốn kém — dùng toPandas trực tiếp
        pdf = sdf.toPandas()
        if pdf.empty:
            print(f"[INFO] {stream_name}: 0 rows")
        else:
            print(f"[INFO] {stream_name}: {len(pdf)} rows, cols={list(pdf.columns)}")
        return pdf
    except Exception as e:
        print(f"[WARN] {stream_name}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────────────────────
def _no_data(ax, msg="No data yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color=TXT, fontsize=9, transform=ax.transAxes,
            style="italic", alpha=0.7)


# ──────────────────────────────────────────────────────────────────────────────
# PANEL 1 — ORDER ANOMALY (Stream 1)
# ──────────────────────────────────────────────────────────────────────────────
def draw_s1_bar(ax, df: pd.DataFrame):
    """Stacked bar: GMV by order status."""
    ax.clear()
    ax.set_title("💰 GMV by Order Status", pad=6)
    if df.empty or "total_gmv" not in df.columns:
        _no_data(ax); return

    agg = df.groupby("status")["total_gmv"].sum().reset_index() \
            .sort_values("total_gmv", ascending=False)
    clr_map = {"Complete": C3, "Returned": C4, "Cancelled": YELL,
                "Pending": C1, "Processing": C5, "Shipped": C2}
    colors = [clr_map.get(s, C1) for s in agg["status"]]

    bars = ax.bar(agg["status"], agg["total_gmv"],
                  color=colors, edgecolor=BG, linewidth=0.7, zorder=3)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h * 1.01,
                f"${h/1000:.0f}K", ha="center", va="bottom",
                fontsize=7, color=TXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.tick_params(axis="x", labelsize=7)


def draw_s1_line(ax, df: pd.DataFrame):
    """Line: anomaly rate over time."""
    ax.clear()
    ax.set_title("📈 Anomaly Rate % (over time)", pad=6)
    if df.empty or "window_start" not in df.columns or "anomaly_rate_pct" not in df.columns:
        _no_data(ax); return

    trend = (df.groupby("window_start")["anomaly_rate_pct"]
               .mean().reset_index()
               .sort_values("window_start").tail(MAX_HISTORY))
    if len(trend) < 2:
        _no_data(ax, "Need ≥2 time windows"); return

    ax.plot(trend["window_start"], trend["anomaly_rate_pct"],
            color=C4, lw=2, marker="o", ms=3, zorder=3)
    ax.fill_between(trend["window_start"], trend["anomaly_rate_pct"],
                    alpha=0.12, color=C4)
    ax.axhline(10, color=C4, ls="--", lw=1, alpha=0.7, label="10% threshold")
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.grid(alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=6.5)


# ──────────────────────────────────────────────────────────────────────────────
# PANEL 2 — TRAFFIC FUNNEL (Stream 2)
# ──────────────────────────────────────────────────────────────────────────────
def draw_s2_funnel(ax, df: pd.DataFrame):
    """Horizontal funnel bars."""
    ax.clear()
    ax.set_title("🔽 Live Conversion Funnel", pad=6)
    if df.empty or "stage_home" not in df.columns:
        _no_data(ax); return

    stages = ["stage_home", "stage_department", "stage_product",
              "stage_cart", "stage_purchase"]
    labels = ["🏠 Home", "📂 Dept", "📦 Product", "🛒 Cart", "✅ Purchase"]
    values = [float(df[s].sum()) for s in stages]
    max_v  = max(values) if max(values) > 0 else 1
    colors = [C1, C5, C2, C3, C4]

    for i, (lbl, clr) in enumerate(zip(labels, colors)):
        v    = values[i]
        w    = v / max_v
        ax.barh(i, 1.0, color=GRID, height=0.55, alpha=0.4, zorder=1)
        ax.barh(i, w,   color=clr,  height=0.55, alpha=0.85, zorder=2)
        ax.text(min(w + 0.02, 1.15), i, f"{v:,.0f}",
                va="center", fontsize=7.5, color=TXT)
        if i > 0 and values[i-1] > 0:
            cvr = v / values[i-1] * 100
            ax.text(-0.02, i, f"{cvr:.0f}%",
                    ha="right", va="center", fontsize=7,
                    color=YELL if cvr < 30 else C3)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.18, 1.28)
    ax.set_xlabel("Relative Volume", fontsize=7.5)
    ax.grid(axis="x", alpha=0.3)


def draw_s2_donut(ax, df: pd.DataFrame):
    """Donut chart: traffic sources."""
    ax.clear()
    ax.set_title("🌐 Traffic Sources", pad=6)
    if df.empty or "traffic_source" not in df.columns:
        _no_data(ax); return

    src = (df.groupby("traffic_source")["unique_users"]
             .sum().reset_index()
             .sort_values("unique_users", ascending=False).head(6))
    if src.empty or src["unique_users"].sum() == 0:
        _no_data(ax); return

    wedges, texts, autos = ax.pie(
        src["unique_users"], labels=src["traffic_source"],
        autopct="%1.0f%%", colors=CAT_CLR[:len(src)],
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=1.5)
    )
    for t in texts:  t.set_fontsize(7)
    for a in autos:  a.set_fontsize(6.5)


# ──────────────────────────────────────────────────────────────────────────────
# PANEL 3 — INVENTORY ALERT (Stream 3)
# ──────────────────────────────────────────────────────────────────────────────
def draw_s3_heat(ax, df: pd.DataFrame):
    """Heatmap: stock_quantity by category × brand."""
    ax.clear()
    ax.set_title("🗂 Stock Heatmap (Category × Brand)", pad=6)
    if df.empty or "category" not in df.columns:
        _no_data(ax); return

    top_brands = df.groupby("brand")["sold_quantity"].sum().nlargest(5).index
    top_cats   = df.groupby("category")["sold_quantity"].sum().nlargest(7).index
    pivot = (
        df[df["brand"].isin(top_brands) & df["category"].isin(top_cats)]
        .pivot_table(index="category", columns="brand",
                     values="stock_quantity", aggfunc="min")
        .fillna(0)
    )
    if pivot.empty:
        _no_data(ax); return

    cmap = LinearSegmentedColormap.from_list(
        "stock", ["#ef4444", "#f59e0b", "#22c55e"], N=256)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=120)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7.5)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.values[r, c]
            ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5, color="white" if v < 60 else TXT)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                 label="Min Stock").ax.tick_params(labelsize=7)


def draw_s3_alert(ax, df: pd.DataFrame):
    """Bar: count by alert_tier."""
    ax.clear()
    ax.set_title("⚠ Alert Tier Count", pad=6)
    if df.empty or "alert_tier" not in df.columns:
        _no_data(ax); return

    agg = df.groupby("alert_tier").size().reset_index(name="count")
    tier_order = {"CRITICAL": 0, "WARNING": 1, "NOTICE": 2, "OK": 3}
    agg["order"] = agg["alert_tier"].map(tier_order).fillna(99)
    agg = agg.sort_values("order")

    clr_map = {"CRITICAL": C4, "WARNING": YELL, "NOTICE": C2, "OK": C3}
    colors  = [clr_map.get(t, C1) for t in agg["alert_tier"]]
    bars = ax.bar(agg["alert_tier"], agg["count"],
                  color=colors, edgecolor=BG, linewidth=0.5, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                str(int(b.get_height())),
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("SKU Count", fontsize=7.5)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.tick_params(axis="x", labelsize=7.5)


# ──────────────────────────────────────────────────────────────────────────────
# PANEL 4 — REVENUE DASHBOARD (Stream 4)
# ──────────────────────────────────────────────────────────────────────────────
def draw_s4_bar(ax, df: pd.DataFrame):
    """Stacked bar: revenue + profit by category."""
    ax.clear()
    ax.set_title("💵 Revenue & Profit by Category", pad=6)
    if df.empty or "gross_revenue" not in df.columns:
        _no_data(ax); return

    cat = (df.groupby("category")[["gross_revenue", "gross_margin"]]
             .sum().reset_index()
             .sort_values("gross_revenue", ascending=False).head(7))
    if cat.empty:
        _no_data(ax); return

    x    = np.arange(len(cat))
    cost = cat["gross_revenue"] - cat["gross_margin"]
    ax.bar(x, cost, color=C2, label="Cost", edgecolor=BG, lw=0.4, zorder=3)
    ax.bar(x, cat["gross_margin"], bottom=cost,
           color=C3, label="Profit", edgecolor=BG, lw=0.4, zorder=3)
    for i, (rev, marg) in enumerate(
            zip(cat["gross_revenue"], cat["gross_margin"])):
        pct = marg / rev * 100 if rev > 0 else 0
        ax.text(i, rev * 1.01, f"{pct:.0f}%",
                ha="center", va="bottom", fontsize=6.5,
                color=C3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c[:9] for c in cat["category"]],
        rotation=25, ha="right", fontsize=7)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
    ax.grid(axis="y", alpha=0.4, zorder=0)


def draw_s4_margin(ax, df: pd.DataFrame):
    """Line: profit margin % trend over time."""
    ax.clear()
    ax.set_title("📊 Profit Margin Trend", pad=6)
    if df.empty or "window_start" not in df.columns:
        _no_data(ax); return

    trend = (
        df.groupby("window_start")
          .apply(lambda g: g["gross_margin"].sum() /
                           max(g["gross_revenue"].sum(), 1) * 100)
          .reset_index(name="margin_pct")
          .sort_values("window_start").tail(MAX_HISTORY)
    )
    if len(trend) < 2:
        _no_data(ax, "Need ≥2 windows"); return

    y_min = trend["margin_pct"].min()
    y_max = trend["margin_pct"].max()
    y_range = max(y_max - y_min, 1.0)
    # Line nằm ở ~80% chiều cao: thêm padding 25% ở trên, 5% ở dưới
    ax.set_ylim(max(0, y_min - y_range * 0.05),
                y_max + y_range * 0.25)

    ax.plot(trend["window_start"], trend["margin_pct"],
            color=C5, lw=2.5, marker="o", ms=4, zorder=3)
    ax.fill_between(trend["window_start"], trend["margin_pct"],
                    y2=max(0, y_min - y_range * 0.05),
                    alpha=0.15, color=C5)
    # Thêm avg reference line
    avg = trend["margin_pct"].mean()
    ax.axhline(avg, color=C5, ls="--", lw=1, alpha=0.5,
               label=f"Avg {avg:.1f}%")
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=6.5)


# ──────────────────────────────────────────────────────────────────────────────
# KPI HEADER
# ──────────────────────────────────────────────────────────────────────────────
def draw_kpi(ax, o_df, f_df, i_df, r_df):
    ax.clear()
    ax.set_facecolor(BG)
    ax.axis("off")

    rev      = r_df["gross_revenue"].sum() if not r_df.empty and "gross_revenue" in r_df.columns else 0
    profit   = r_df["gross_margin"].sum()  if not r_df.empty and "gross_margin"  in r_df.columns else 0
    orders   = o_df["total_orders"].sum()  if not o_df.empty and "total_orders"  in o_df.columns else 0
    oos      = int((i_df["is_out_of_stock"] == True).sum()) \
               if not i_df.empty and "is_out_of_stock" in i_df.columns else 0
    visitors = f_df["unique_users"].sum() \
               if not f_df.empty and "unique_users" in f_df.columns else 0
    cvr = (f_df["stage_purchase"].sum() /
           max(f_df["stage_home"].sum(), 1) * 100
           if not f_df.empty and "stage_purchase" in f_df.columns else 0)

    kpis = [
        ("💰 Total Revenue", f"${rev:,.0f}",    C3),
        ("📈 Total Profit",  f"${profit:,.0f}", C1),
        ("📦 Orders",        f"{orders:,.0f}",  C5),
        ("👤 Visitors",      f"{visitors:,.0f}",C2),
        ("✅ Conversion",    f"{cvr:.1f}%",     C4),
        ("🔴 Out-of-Stock",  f"{oos} SKUs",     C4 if oos > 3 else C3),
    ]

    # Tiêu đề dashboard ở hàng trên cùng, canh giữa
    ax.text(0.5, 0.95,
            "TheLook E-commerce  •  Real-time Streaming Dashboard",
            ha="center", va="top", fontsize=10, color=C1,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.995, 0.95,
            f"⏱ {datetime.now().strftime('%H:%M:%S')}",
            ha="right", va="top", fontsize=7.5, color=TXT,
            alpha=0.5, transform=ax.transAxes)

    # 6 KPI metrics ở hàng dưới, cách đều nhau
    n = len(kpis)
    step = 1.0 / n
    for i, (lbl, val, clr) in enumerate(kpis):
        cx = step * i + step / 2
        # Value
        ax.text(cx, 0.52, val, ha="center", va="center",
                fontsize=13, fontweight="bold", color=clr,
                transform=ax.transAxes)
        # Label
        ax.text(cx, 0.12, lbl, ha="center", va="center",
                fontsize=7.5, color=TXT,
                transform=ax.transAxes, alpha=0.72)


# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
def build_figure():
    """
    4 rows × 4 cols, không có nested gridspec, không có blank cells thừa:

    ┌──────────────────────────────────────────────────────┐ row 0 (KPI)
    ├────────────┬──────────────┬───────────────────────────┤
    │ S1 GMV Bar │ S1 AnomalyLn │ S2 Funnel  (span 2 cols)  │ row 1
    ├────────────┴──────────────┴───────────────────────────┤
    │ S3 Heatmap     (span 2 cols)   │ S3 Alert  │ S2 Donut  │ row 2
    ├────────────────────────────────┴───────────┴──────────┤
    │ S4 Revenue Bar  (span 2 cols)  │ S4 Margin │ (blank)   │ row 3
    └────────────────────────────────────────────────────────┘
    """
    fig = plt.figure(figsize=(21, 12), facecolor=BG)

    gs = gridspec.GridSpec(
        nrows=4, ncols=4, figure=fig,
        height_ratios=[0.38, 1.1, 1.1, 1.0],
        hspace=0.62, wspace=0.38,
        left=0.04, right=0.98, top=0.97, bottom=0.06
    )

    # Row 0: KPI
    ax_kpi = fig.add_subplot(gs[0, :])

    # Row 1: S1-bar | S1-line | S2-funnel (2 cols)
    ax_s1_bar    = fig.add_subplot(gs[1, 0])
    ax_s1_line   = fig.add_subplot(gs[1, 1])
    ax_s2_funnel = fig.add_subplot(gs[1, 2:4])

    # Row 2: S3-heat (2 cols) | S3-alert | S2-donut
    ax_s3_heat   = fig.add_subplot(gs[2, 0:2])
    ax_s3_alert  = fig.add_subplot(gs[2, 2])
    ax_s2_donut  = fig.add_subplot(gs[2, 3])

    # Row 3: S4-bar (2 cols) | S4-margin | blank
    ax_s4_bar    = fig.add_subplot(gs[3, 0:2])
    ax_s4_margin = fig.add_subplot(gs[3, 2])
    ax_blank     = fig.add_subplot(gs[3, 3])
    ax_blank.axis("off")

    axes = dict(
        kpi=ax_kpi,
        s1_bar=ax_s1_bar, s1_line=ax_s1_line,
        s2_funnel=ax_s2_funnel, s2_donut=ax_s2_donut,
        s3_heat=ax_s3_heat, s3_alert=ax_s3_alert,
        s4_bar=ax_s4_bar, s4_margin=ax_s4_margin,
    )
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE LOOP
# ──────────────────────────────────────────────────────────────────────────────
def update(frame, axes):
    o_df = read_parquet("order_anomaly")
    f_df = read_parquet("traffic_funnel")
    i_df = read_parquet("inventory_alert")
    r_df = read_parquet("revenue_dashboard")

    draw_kpi(axes["kpi"], o_df, f_df, i_df, r_df)

    draw_s1_bar(axes["s1_bar"],       o_df)
    draw_s1_line(axes["s1_line"],     o_df)
    draw_s2_funnel(axes["s2_funnel"], f_df)
    draw_s2_donut(axes["s2_donut"],   f_df)
    draw_s3_heat(axes["s3_heat"],     i_df)
    draw_s3_alert(axes["s3_alert"],   i_df)
    draw_s4_bar(axes["s4_bar"],       r_df)
    draw_s4_margin(axes["s4_margin"], r_df)

    axes["kpi"].get_figure().canvas.draw_idle()
    print(
        f"[FRAME {frame}] {datetime.now().strftime('%H:%M:%S')} | "
        f"orders={len(o_df)} funnel={len(f_df)} inv={len(i_df)} rev={len(r_df)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  TheLook Dashboard — Paths: {HDFS_STREAMING_DIR} | {HDFS_DATATEST_DIR}")
    print(f"  Refresh: {REFRESH_SEC}s")
    print(f"{'='*55}\n")

    fig, axes = build_figure()
    ani = FuncAnimation(
        fig,
        func=lambda frame: update(frame, axes),
        interval=REFRESH_SEC * 1000,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()