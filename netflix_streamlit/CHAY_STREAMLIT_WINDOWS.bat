@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo =================================================
echo NETFLIX DATA MINING - STREAMLIT DEMO
 echo =================================================
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Khong the cai thu vien. Kiem tra Python va Internet.
    pause
    exit /b 1
)
python -m streamlit run app.py
pause
