# Fraud Detection System

## What you'll see when it's running

| URL | What opens |
|-----|-----------|
| http://localhost:8050 | Live fraud dashboard (transactions, alerts, metrics) |
| http://localhost:5000/health | API health — shows models loaded status |
| http://localhost:5000/metrics | Scored/blocked/flagged counters |
| http://localhost:9090 | Prometheus metrics graphs |

---

## Folder structure — put ALL these files together

```
fraud-detection/
├── setup.bat                      ← double-click this to run everything
├── dashboard.py
├── app.py
├── consumer.py
├── producer.py
├── train_models.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .dockerignore
├── prometheus.yml
└── reduced_creditcard_small.csv   ← copy your CSV here
```

---

## Quickstart (Windows)

### Option A — Easiest (one double-click)
1. Copy `reduced_creditcard_small.csv` into this folder
2. Make sure Docker Desktop is open and running (green icon in taskbar)
3. Double-click **`setup.bat`**
4. Wait ~5 minutes — it does everything automatically

### Option B — Manual step by step

**Step 1 — Install Python packages**
```
python -m pip install -r requirements.txt
```

**Step 2 — Train the ML models**
```
python train_models.py
```
You should see a `models/` folder appear with `.pkl` files.

**Step 3 — Build and start Docker**
```
docker compose up -d
```

**Step 4 — Start the dashboard**
```
python dashboard.py
```
Open http://localhost:8050

---

## Stream live transactions (optional)

Open **two separate terminals**:

Terminal 1 — send fake transactions into Kafka:
```
docker compose run --rm fraud-api python producer.py
```

Terminal 2 — watch fraud decisions in real time:
```
docker compose logs -f consumer
```

You will see output like:
```
12:05:01  WARNING   BLOCK  card=card_4821    score=0.892  txn=3F8A2B1C
12:05:01  INFO      FLAG   card=card_7734    score=0.541  txn=9D2E4F6A
12:05:02  DEBUG     allow  card=card_1293    score=0.043
```

---

## Test the prediction API

```
curl -X POST http://localhost:5000/predict ^
  -H "X-API-Key: fraud-demo-key-2024" ^
  -H "Content-Type: application/json" ^
  -d "{\"transaction_id\":\"test1\",\"card_id\":\"card_1234\",\"amount\":3500,\"merchant_cat\":\"atm\",\"country\":\"NG\",\"hour_of_day\":2,\"is_weekend\":1}"
```

Expected response:
```json
{
  "transaction_id": "test1",
  "fraud_score": 0.84,
  "decision": "block",
  "model_scores": {
    "isolation_forest": 0.91,
    "gradient_boosting": 0.79
  },
  "latency_ms": 3.2
}
```

---

## Check all services are healthy

```
docker compose ps
```

Expected output:
```
NAME            STATUS
zookeeper       running (healthy)
kafka           running (healthy)
kafka-init      exited (0)        <- normal, one-shot job
fraud-api       running (healthy)
consumer        running (running)
prometheus      running (running)
```

---

## Stop everything

```
docker compose down
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `reduced_creditcard_small.csv not found` | Copy the CSV into the same folder as setup.bat |
| Docker Desktop not running | Open Docker Desktop, wait for green icon |
| Port 5000 already in use | Stop other apps using port 5000, or restart Docker |
| Models folder missing | Run `python train_models.py` manually |
| `fraud-api` not healthy | Check `docker compose logs fraud-api` |
