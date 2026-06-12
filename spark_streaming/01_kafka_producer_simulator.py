import json
import time
import random
import argparse
import logging
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("KafkaProducer")

# ─────────────────────────────────────────────
# KAFKA CONFIG  (chỉnh bootstrap_servers nếu cần)
# ─────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"
TOPICS = {
    "orders":    "orders-topic",
    "events":    "events-topic",
    "inventory": "inventory-topic",
    "revenue":   "revenue-topic",
}

# ─────────────────────────────────────────────
# DOMAIN DATA  (tổng hợp từ TheLook dataset)
# ─────────────────────────────────────────────
PRODUCT_CATEGORIES = [
    "Intimates", "Jeans", "Swim", "Pants & Capris", "Shorts",
    "Tops & Tees", "Blazers & Jackets", "Dresses", "Accessories",
    "Socks & Hosiery", "Suits & Sport Coats", "Sweaters",
    "Active", "Outerwear & Coats", "Skirts",
]

BRANDS = [
    "Calvin Klein", "Tommy Hilfiger", "Nike", "Adidas", "Levi's",
    "Ralph Lauren", "Zara", "H&M", "Gucci", "Prada",
    "Under Armour", "The North Face", "Patagonia", "Mango", "Uniqlo",
]

TRAFFIC_SOURCES = ["Organic", "Adwords", "Facebook", "Email", "Display", "Search", "Referral"]
COUNTRIES       = ["United States", "United Kingdom", "France", "Germany", "Japan",
                   "Brazil", "China", "Australia", "Canada", "India"]
ORDER_STATUSES  = ["Pending", "Processing", "Shipped", "Complete", "Cancelled", "Returned"]
EVENT_TYPES     = ["home", "department", "category", "brand", "product",
                   "cart", "purchase", "cancel"]
DISTRIBUTION_CENTERS = [
    "Memphis TN", "Chicago IL", "Houston TX", "Los Angeles CA",
    "New York NY", "Philadelphia PA", "Port Authority of New York/New Jersey NY",
    "Savannah GA", "Shanghai China", "Yokohama Japan",
]

# ─────────────────────────────────────────────
# GENERATOR HELPERS
# ─────────────────────────────────────────────

def _now_iso() -> str:
    """Trả về timestamp UTC hiện tại dạng ISO-8601."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _rand_user_id() -> int:
    return random.randint(1, 100_000)


def _rand_product_id() -> int:
    return random.randint(1, 29_120)


def _rand_order_id() -> int:
    return random.randint(1, 500_000)


def _rand_price() -> float:
    return round(random.uniform(5.0, 900.0), 2)


def _rand_cost() -> float:
    return round(random.uniform(2.0, 500.0), 2)


# ─────────────────────────────────────────────
# 4 EVENT GENERATORS
# ─────────────────────────────────────────────

def generate_order_event(anomaly: bool = False) -> dict:
    """
    Tạo sự kiện đặt hàng.
    anomaly=True  → giá trị bất thường: giá quá cao hoặc số lượng lớn bất thường
                    dùng để test Stream 1 (Order Anomaly Detection).
    """
    order_id   = _rand_order_id()
    user_id    = _rand_user_id()
    num_items  = random.randint(1, 5) if not anomaly else random.randint(20, 50)
    unit_price = _rand_price()       if not anomaly else round(random.uniform(5000, 15000), 2)

    return {
        "order_id":         order_id,
        "user_id":          user_id,
        "status":           random.choice(ORDER_STATUSES),
        "gender":           random.choice(["M", "F"]),
        "created_at":       _now_iso(),
        "returned_at":      None,
        "shipped_at":       _now_iso() if random.random() > 0.4 else None,
        "delivered_at":     _now_iso() if random.random() > 0.6 else None,
        "num_of_item":      num_items,
        "country":          random.choice(COUNTRIES),
        "traffic_source":   random.choice(TRAFFIC_SOURCES),
        # order_items nested (flatten khi consume)
        "product_id":       _rand_product_id(),
        "category":         random.choice(PRODUCT_CATEGORIES),
        "brand":            random.choice(BRANDS),
        "sale_price":       unit_price,
        "cost":             round(unit_price * random.uniform(0.3, 0.75), 2),
        "is_anomaly":       anomaly,   # label để visualize
    }


def generate_event_event() -> dict:
    """
    Tạo sự kiện hành vi người dùng trên website (traffic funnel).
    Dùng cho Stream 2 (Live Traffic Funnel).
    """
    user_id  = _rand_user_id()
    sequence = random.choice([
        ["home"],
        ["home", "department"],
        ["home", "department", "category"],
        ["home", "department", "category", "brand"],
        ["home", "department", "category", "brand", "product"],
        ["home", "department", "category", "brand", "product", "cart"],
        ["home", "department", "category", "brand", "product", "cart", "purchase"],
    ])
    event_type = sequence[-1]

    return {
        "event_id":       random.randint(1, 10_000_000),
        "user_id":        user_id,
        "session_id":     f"sess_{user_id}_{random.randint(1, 9999)}",
        "event_type":     event_type,
        "funnel_depth":   len(sequence),
        "ip_address":     f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
        "city":           f"City_{random.randint(1, 200)}",
        "country":        random.choice(COUNTRIES),
        "traffic_source": random.choice(TRAFFIC_SOURCES),
        "browser":        random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
        "created_at":     _now_iso(),
        "uri":            f"/{event_type}/{random.randint(1, 500)}",
    }


def generate_inventory_event() -> dict:
    """
    Tạo sự kiện cập nhật tồn kho.
    Dùng cho Stream 3 (Inventory Alert).
    LOW_STOCK: sold_quantity > stock_quantity * 0.8
    """
    product_id      = _rand_product_id()
    stock_quantity  = random.randint(0, 200)
    sold_quantity   = random.randint(0, stock_quantity + 10)   # đôi khi oversell
    available       = max(stock_quantity - sold_quantity, 0)

    return {
        "inventory_item_id":  random.randint(1, 1_000_000),
        "product_id":         product_id,
        "product_name":       f"Product_{product_id}",
        "category":           random.choice(PRODUCT_CATEGORIES),
        "brand":              random.choice(BRANDS),
        "distribution_center": random.choice(DISTRIBUTION_CENTERS),
        "cost":               _rand_cost(),
        "product_retail_price": _rand_price(),
        "stock_quantity":     stock_quantity,
        "sold_quantity":      sold_quantity,
        "available_quantity": available,
        "is_low_stock":       available <= 10,
        "is_out_of_stock":    available == 0,
        "updated_at":         _now_iso(),
    }


def generate_revenue_event() -> dict:
    """
    Tạo sự kiện thanh toán / doanh thu.
    Dùng cho Stream 4 (Revenue Dashboard).
    """
    order_id   = _rand_order_id()
    num_items  = random.randint(1, 8)
    sale_price = _rand_price()
    cost       = round(sale_price * random.uniform(0.3, 0.75), 2)
    margin     = round(sale_price - cost, 2)

    return {
        "order_id":         order_id,
        "user_id":          _rand_user_id(),
        "product_id":       _rand_product_id(),
        "category":         random.choice(PRODUCT_CATEGORIES),
        "brand":            random.choice(BRANDS),
        "country":          random.choice(COUNTRIES),
        "traffic_source":   random.choice(TRAFFIC_SOURCES),
        "sale_price":       sale_price,
        "cost":             cost,
        "gross_margin":     margin,
        "num_of_item":      num_items,
        "total_revenue":    round(sale_price * num_items, 2),
        "total_cost":       round(cost * num_items, 2),
        "total_margin":     round(margin * num_items, 2),
        "status":           random.choice(["Complete", "Processing", "Shipped"]),
        "created_at":       _now_iso(),
    }


# ─────────────────────────────────────────────
# PRODUCER FACTORY
# ─────────────────────────────────────────────

def create_producer() -> KafkaProducer:
    """Khởi tạo KafkaProducer với JSON serializer."""
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",                 # đảm bảo ghi thành công
        retries=5,
        linger_ms=10,               # micro-batching
        compression_type="gzip",
    )
    log.info("KafkaProducer connected to %s", KAFKA_BOOTSTRAP)
    return producer


def delivery_report(record_metadata):
    log.debug(
        "Delivered → topic=%s  partition=%d  offset=%d",
        record_metadata.topic,
        record_metadata.partition,
        record_metadata.offset,
    )


# ─────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────

def run_simulation(rate: float, duration: int):
    """
    rate     : số event/giây mỗi topic
    duration : tổng thời gian chạy (giây), -1 = chạy mãi
    """
    producer   = create_producer()
    start_time = time.time()
    counters   = {k: 0 for k in TOPICS}
    interval   = 1.0 / max(rate, 0.1)
    anomaly_prob = 0.03   # 3% chance anomaly order

    log.info("Simulation started  rate=%.1f ev/s  duration=%ds", rate, duration)
    log.info("Topics: %s", list(TOPICS.values()))

    try:
        while True:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                log.info("Duration reached. Stopping.")
                break

            # ── orders-topic
            is_anomaly = random.random() < anomaly_prob
            order_ev   = generate_order_event(anomaly=is_anomaly)
            producer.send(
                TOPICS["orders"],
                key=order_ev["order_id"],
                value=order_ev,
            ).add_callback(delivery_report)
            counters["orders"] += 1

            # ── events-topic  (traffic lebih tinggi ~3x)
            for _ in range(random.randint(2, 5)):
                ev = generate_event_event()
                producer.send(TOPICS["events"], key=ev["user_id"], value=ev).add_callback(delivery_report)
                counters["events"] += 1

            # ── inventory-topic
            inv_ev = generate_inventory_event()
            producer.send(TOPICS["inventory"], key=inv_ev["product_id"], value=inv_ev).add_callback(delivery_report)
            counters["inventory"] += 1

            # ── revenue-topic
            rev_ev = generate_revenue_event()
            producer.send(TOPICS["revenue"], key=rev_ev["order_id"], value=rev_ev).add_callback(delivery_report)
            counters["revenue"] += 1

            # Flush + log progress mỗi 50 vòng
            if sum(counters.values()) % 200 == 0:
                producer.flush()
                log.info(
                    "[+%.0fs] Sent → orders=%d  events=%d  inventory=%d  revenue=%d",
                    elapsed, counters["orders"], counters["events"],
                    counters["inventory"], counters["revenue"],
                )

            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down producer.")
    finally:
        producer.flush()
        producer.close()
        log.info(
            "Producer closed. Total sent → orders=%d  events=%d  inventory=%d  revenue=%d",
            counters["orders"], counters["events"],
            counters["inventory"], counters["revenue"],
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TheLook Kafka Producer Simulator")
    parser.add_argument(
        "--rate", type=float, default=2.0,
        help="Số event orders/giây (default: 2.0). Events-topic sẽ cao hơn ~3-5x.",
    )
    parser.add_argument(
        "--duration", type=int, default=-1,
        help="Thời gian chạy tính bằng giây (default: -1 = chạy mãi đến Ctrl+C).",
    )
    args = parser.parse_args()
    run_simulation(rate=args.rate, duration=args.duration)