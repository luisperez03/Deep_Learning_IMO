@echo off
cd /d "%~dp0"
echo PILOTO: FCN, SegNet y U-Net, cinco semillas cada uno.
echo Puede tardar muchas horas. Conecta el cargador y cierra ChatGPT.
echo Ctrl+C detiene. Este mismo archivo reanuda desde la ultima epoca completa.
echo No ejecuta test ni modifica los datos originales.
pause
".venv-cpu\Scripts\python.exe" -u scripts\verificar_reanudacion.py
if errorlevel 1 goto fin
".venv-cpu\Scripts\python.exe" -u scripts\piloto_local.py
:fin
echo.
echo Consulta resultados\piloto_local_v1 y los mensajes anteriores.
pause
