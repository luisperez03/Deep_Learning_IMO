@echo off
cd /d "%~dp0"
echo Cierra ChatGPT por completo antes de continuar.
echo Se probaran U-Net y SegNet, y luego UNA epoca de FCN con validacion.
pause
".venv-cpu\Scripts\python.exe" -u scripts\prueba_carga.py --model unet > resultados\preentrenamiento\carga_unet.log 2>&1
if errorlevel 1 goto errorunet
type resultados\preentrenamiento\carga_unet.log
".venv-cpu\Scripts\python.exe" -u scripts\prueba_carga.py --model segnet > resultados\preentrenamiento\carga_segnet.log 2>&1
if errorlevel 1 goto errorsegnet
type resultados\preentrenamiento\carga_segnet.log
".venv-cpu\Scripts\python.exe" -u scripts\medir_epoca_local.py
goto fin
:errorsegnet
type resultados\preentrenamiento\carga_segnet.log
goto fin
:errorunet
type resultados\preentrenamiento\carga_unet.log
:fin
echo.
echo Finalizado. Los resultados quedaron guardados. Puedes volver al chat.
pause
