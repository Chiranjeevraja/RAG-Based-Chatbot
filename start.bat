@echo off
echo ========================================
echo  YouTube RAG Chatbot - Startup Script
echo ========================================

:: Check .env exists
if not exist ".env" (
    echo [ERROR] .env file not found!
    echo Copy .env.example to .env and fill in your API keys.
    pause
    exit /b 1
)

:: Start backend
echo.
echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
cd backend
start "RAG Backend" cmd /k "pip install -r requirements.txt -q && python main.py"
cd ..

:: Wait a moment then start frontend
timeout /t 3 /nobreak >nul

echo.
echo [2/2] Starting React frontend on http://localhost:5173 ...
cd frontend
start "RAG Frontend" cmd /k "npm install && npm run dev"
cd ..

echo.
echo Both servers are starting. Open http://localhost:5173 in your browser.
echo.
pause
