"""
producer.py  —  Synthetic transaction producer
Streams fake card transactions to Kafka every 0.3 seconds.
Run inside Docker:  docker compose run --rm fraud-api python producer.py
"""
import json, os, random, time, uuid
from datetime import datetime
from kafka import KafkaProducer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC        = "transactions"
FRAUD_RATE   = 0.08   # 8% fraud for visible demo

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks="all", retries=3,
)

MERCHANTS  = ["Amazon","Flipkart","Zomato","HDFC ATM","Shell Fuel",
               "BookMyShow","Swiggy","MakeMyTrip","IRCTC","Uber"]
CATEGORIES = ["grocery","fuel","online","atm","travel"]


def make_transaction(is_fraud=False):
    amount  = round(random.uniform(800, 5000), 2) if is_fraud \
              else round(random.uniform(10, 500), 2)
    country = random.choice(["NG","CN","AE","US"]) if is_fraud else "IN"
    return {
        "transaction_id": str(uuid.uuid4()),
        "card_id":        f"card_{random.randint(1000,9999)}",
        "amount":         amount,
        "merchant_id":    f"merch_{random.randint(100,999)}",
        "merchant_cat":   random.choice(CATEGORIES),
        "merchant_name":  random.choice(MERCHANTS),
        "country":        country,
        "hour_of_day":    datetime.utcnow().hour,
        "is_weekend":     int(datetime.utcnow().weekday() >= 5),
        "timestamp":      datetime.utcnow().isoformat(),
    }


def main():
    print(f"[producer] → Kafka {KAFKA_BROKER}  topic={TOPIC}  (Ctrl-C to stop)")
    while True:
        is_fraud = random.random() < FRAUD_RATE
        event    = make_transaction(is_fraud)
        future   = producer.send(TOPIC, key=event["card_id"].encode(), value=event)
        try:
            meta = future.get(timeout=5)
            tag  = "FRAUD " if is_fraud else "normal"
            print(f"  [{tag}] {event['transaction_id'][:8]}  "
                  f"₹{event['amount']:>8.2f}  {event['country']}  "
                  f"→ partition {meta.partition}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
