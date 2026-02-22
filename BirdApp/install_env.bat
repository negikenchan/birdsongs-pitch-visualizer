@echo off
echo [BirdApp] Pythonパッケージをインストールしています...
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base
pip install -r requirements.txt
pause
