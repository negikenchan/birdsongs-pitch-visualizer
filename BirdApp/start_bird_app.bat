@echo off
cd /d %~dp0
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base
streamlit run bird_app.py
pause
