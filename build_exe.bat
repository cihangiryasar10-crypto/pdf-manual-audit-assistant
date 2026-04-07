@echo off
setlocal

if exist .venv (
  call .venv\Scripts\activate
)

pyinstaller ^
  --noconfirm ^
  --clean ^
  --name ManualAuditAssistant ^
  --onedir ^
  --add-data "app.py;." ^
  --add-data ".streamlit;.streamlit" ^
  run_app.py

echo.
echo Build tamamlandi. Cikti klasoru: dist\ManualAuditAssistant
pause
