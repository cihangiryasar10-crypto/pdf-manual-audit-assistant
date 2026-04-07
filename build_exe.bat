@echo off
setlocal

if exist .venv (
  call .venv\Scripts\activate
)

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name ManualAuditAssistant ^
  --onedir ^
  --add-data "app.py;." ^
  run_app.py

if errorlevel 1 exit /b 1
if not exist dist\ManualAuditAssistant exit /b 1

echo Build tamamlandi. Cikti klasoru: dist\ManualAuditAssistant
dir dist
