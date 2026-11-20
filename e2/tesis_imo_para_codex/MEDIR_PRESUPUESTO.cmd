@echo off
cd /d "%~dp0"
echo Medira UNA epoca por modelo. Cierra ChatGPT y conecta el cargador.
pause
".venv-cpu\Scripts\python.exe" -u scripts\verificar_reanudacion.py
if errorlevel 1 goto fin
".venv-cpu\Scripts\python.exe" -u scripts\piloto_local.py --measure-only
:fin
echo.
echo Consulta resultados\piloto_local_v1\presupuesto.json
pause
