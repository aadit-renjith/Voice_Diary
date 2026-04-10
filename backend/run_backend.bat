@echo off
cd /d "c:\Users\m\OneDrive\Desktop\Projects\Voice_Diary"
call venv\Scripts\activate.bat
cd backend
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
pause
