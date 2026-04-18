"""
consumer.py  —  Fraud detection Kafka consumer
Reads transactions → calls Flask /predict → routes block/flag/allow
→ publishes enriched result to fraud.predictions topic.
"""
import json, logging, os, time
from typing import Optional
import requests
from kafka import KafkaConsumer, KafkaProducer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

KAFKA_BROKER   = os.getenv("KAFKA_BROKER",     "localhost:9092")
INPUT_TOPIC    = os.getenv("INPUT_TOPIC",      "transactions")
OUTPUT_TOPIC   = os.getenv("OUTPUT_TOPIC",     "fraud.predictions")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP",   "fraud-detector-v1")
FLASK_API_URL  = os.getenv("FLASK_API_URL",    "http://localhost:5000")
API_KEY        = os.getenv("API_KEY",          "secret-key-dev-only")
TIMEOUT        = float(os.getenv("REQUEST_TIMEOUT_S", "2.0"))
MAX_RETRIES    = int(os.getenv("MAX_RETRIES",  "3"))

SESSION = requests.Session()
SESSION.headers.update({"X-API-Key": API_KEY, "Content-Type": "application/json"})


def call_predict(txn: dict) -> Optional[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.post(f"{FLASK_API_URL}/predict", json=txn, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            log.warning("API %d on attempt %d", r.status_code, attempt)
        except requests.exceptions.Timeout:
            log.warning("Timeout attempt %d/%d", attempt, MAX_RETRIES)
        except requests.exceptions.ConnectionError as e:
            log.error("Connection error: %s", e)
            time.sleep(0.5 * attempt)
    return None


def main():
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_records=50,
        session_timeout_ms=30_000,
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
    )
    log.info("Consumer ready  group=%s  topic=%s", CONSUMER_GROUP, INPUT_TOPIC)

    try:
        for msg in consumer:
            txn = msg.value
            t0  = time.perf_counter()
            res = call_predict(txn)
            if res is None:
                log.error("Skipping %s — API unavailable", txn.get("transaction_id","?")[:12])
                consumer.commit()
                continue

            decision = res.get("decision", "allow")
            score    = res.get("fraud_score", 0)
            card     = txn.get("card_id","?")

            if decision == "block":
                log.warning("BLOCK  card=%-12s  score=%.3f  txn=%s",
                            card, score, txn.get("transaction_id","?")[:12])
            elif decision == "flag":
                log.info("FLAG   card=%-12s  score=%.3f  txn=%s",
                         card, score, txn.get("transaction_id","?")[:12])
            else:
                log.debug("allow  card=%-12s  score=%.3f", card, score)

            producer.send(OUTPUT_TOPIC,
                          key=card.encode(),
                          value={**res, "original_event": txn})

            ms = round((time.perf_counter()-t0)*1000, 1)
            log.info("Processed %.1fms  offset=%d  partition=%d",
                     ms, msg.offset, msg.partition)
            consumer.commit()

    except KeyboardInterrupt:
        log.info("Shutting down …")
    finally:
        consumer.close()
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
