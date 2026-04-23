@echo off
echo ===============================
echo Starting Brain Tumor App...
echo ===============================

REM Activate virtual environment if exists
IF EXIST venv (
    call venv\Scripts\activate
)

REM Install dependencies
echo Installing requirements...
pip install -r requirements.txt

REM Check if model exists, else train
IF NOT EXIST models\model.pkl (
    echo Training model...
    python train_model.py
)

REM Start FastAPI server in background
start cmd /k python main.py

REM Wait for server to start
timeout /t 3 >nul

REM Open browser
start http://localhost:8000

echo ===============================
echo App running at http://localhost:8000
echo ===============================
pause
