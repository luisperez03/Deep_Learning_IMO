@echo off
cd /d "%~dp0"
".venv-cpu\Scripts\python.exe" "scripts\prueba_carga.py" --model fcn
echo.
echo Resultado guardado en resultados\preentrenamiento\carga_fcn.json
pause
