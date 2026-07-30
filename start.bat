@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting AI analysis server on port 8000...
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
