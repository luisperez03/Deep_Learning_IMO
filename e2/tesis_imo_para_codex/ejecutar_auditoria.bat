@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
)
echo [2/3] Instalando dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo [3/3] Auditando datasets originales sin modificarlos...
".venv\Scripts\python.exe" scripts\auditar_datasets.py
if errorlevel 1 goto :error
echo Creando muestrarios visuales para validar clases...
".venv\Scripts\python.exe" scripts\crear_muestrarios.py
if errorlevel 1 goto :error
echo.
echo Auditoria finalizada. Lee resultados\auditoria\INFORME_AUDITORIA.md
pause
exit /b 0
:error
echo.
echo Error. Revisa que tengas Python 3.10+ y acceso para instalar dependencias.
pause
exit /b 1
