@echo off
title Fraud Detection System — Setup
color 0A

echo.
echo  ============================================
echo   FRAUD DETECTION SYSTEM — SETUP
echo  ============================================
echo.

REM ── Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)

REM ── Check Docker ──────────────────────────────────────────────────────────
docker info >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Docker Desktop is not running.
    echo  Please start Docker Desktop and run this script again.
    pause & exit /b 1
)
echo  [OK] Docker Desktop is running

REM ── Check CSV file ────────────────────────────────────────────────────────
if not exist "reduced_creditcard_small.csv" (
    echo.
    echo  [ERROR] reduced_creditcard_small.csv not found in this folder!
    echo  Please copy the CSV file into this folder and run setup.bat again.
    echo.
    pause & exit /b 1
)
echo  [OK] CSV file found

REM ── Install Python dependencies ───────────────────────────────────────────
echo.
echo  [1/4] Installing Python packages...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo  [ERROR] pip install failed
    pause & exit /b 1
)
echo  [OK] Python packages installed

REM ── Train models ──────────────────────────────────────────────────────────
echo.
echo  [2/4] Training ML models (takes 1-2 minutes)...
python train_models.py
if errorlevel 1 (
    echo  [ERROR] Model training failed
    pause & exit /b 1
)
echo  [OK] Models saved to models/

REM ── Build Docker image ────────────────────────────────────────────────────
echo.
echo  [3/4] Building Docker image (first time takes 3-5 minutes)...
docker compose build
if errorlevel 1 (
    echo  [ERROR] Docker build failed
    pause & exit /b 1
)
echo  [OK] Docker image built

REM ── Start all services ────────────────────────────────────────────────────
echo.
echo  [4/4] Starting all services...
docker compose up -d
if errorlevel 1 (
    echo  [ERROR] docker compose up failed
    pause & exit /b 1
)

echo.
echo  ============================================
echo   ALL SERVICES STARTED!
echo  ============================================
echo.
echo  Waiting 15 seconds for services to be ready...
timeout /t 15 /nobreak >nul

echo.
echo  Open these URLs in your browser:
echo.
echo    Dashboard :  http://localhost:8050
echo    API Health:  http://localhost:5000/health
echo    Metrics   :  http://localhost:5000/metrics
echo    Prometheus:  http://localhost:9090
echo.
echo  To stream live transactions, open a NEW terminal and run:
echo    docker compose run --rm fraud-api python producer.py
echo.
echo  To watch consumer decisions, open a NEW terminal and run:
echo    docker compose logs -f consumer
echo.
echo  To stop everything:
echo    docker compose down
echo.

REM ── Start dashboard in browser ────────────────────────────────────────────
echo  Starting dashboard in browser...
start http://localhost:8050

REM ── Also start the standalone dashboard ──────────────────────────────────
echo  Starting standalone dashboard server...
start "FraudSentinel Dashboard" python dashboard.py

pause
