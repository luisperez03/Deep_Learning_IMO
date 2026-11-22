"""Integra el Jupyter Book desde resultados auditados; no entrena ni publica.

  python scripts/integrar_entregable2.py                  # textos a resultados/integracion_e2/libro_generado/
  python scripts/integrar_entregable2.py --escribir-libro # sobrescribe los capítulos de libro/ (no recomendado)

Los capítulos que genera (index, _toc, resultados_e2, modelos_e2, checklist_e2, reproducibilidad_e2,
e2_optimizadores.inc) y referencias_e2.bib se han ampliado a mano después de la primera integración: el BibTeX
tiene 21 entradas y este script solo produce las diez de arquitecturas. Por eso, por defecto, el texto va a un
área aparte para cotejarlo con el libro y solo se sobrescribe libro/ con la bandera explícita. Las figuras se
regeneran siempre en libro/figuras/e2 (dependen solo de los datos).
"""
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from analisis_integracion_e2 import NAMES

P=Path(__file__).resolve().parents[1]; B=P/'libro'; D=P/'docs/revision_entregable2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
ESCRIBIR_LIBRO='--escribir-libro' in sys.argv
GENERADO=P/'resultados/integracion_e2/libro_generado'
def put(p,s):
    p=Path(p)
    if p.parent==B and not ESCRIBIR_LIBRO:
        GENERADO.mkdir(parents=True,exist_ok=True); p=GENERADO/p.name
    p.write_text(s.strip()+'\n',encoding='utf-8')

def main():
    a=read(P/'resultados/auditoria_test_e2/AUDITORIA.json')
    s=read(P/'resultados/integracion_e2/analisis.json')
    config=read(P/'config/experimento_e2.json')
    (B/'figuras/e2').mkdir(parents=True,exist_ok=True)
    shutil.copy2(P/'resultados/auditoria_test_e2/comparacion_validacion_test.png',B/'figuras/e2/comparacion.png')
    shutil.copy2(P/'resultados/revision_visual_e2/panel_validacion.png',B/'figuras/e2/panel_validacion.png')
    put(B/'index.md',r'''
# Segmentación de etiquetas de mercancías peligrosas

**Deep Learning · Entregable 2 · Integración local del 20 de noviembre de 2026**

**Autores:** Luis Perez y Daniel Carbono. **Entrega prevista:** domingo 22 de noviembre.

## Resumen

Se compararon diez adaptaciones de redes para segmentación binaria de rótulos mediante cinco semillas por modelo. Se conservaron las particiones agrupadas del Entregable 1 y se entrenó desde cero a 256×256 píxeles, con selección del checkpoint por Dice de validación. La evaluación final comprendió 218 imágenes de 122 grupos de similitud. TransUNet alcanzó Dice 0,8626 y FCN compacta 0,8535. La diferencia pareada fue 0,0091, con IC95 de grupos [−0,0083; 0,0278], condicional a las cinco semillas; no establece mejora clara ni equivalencia práctica. FCN ofrece una referencia compacta de 72.145 parámetros. Se auditaron las cincuenta corridas y las máscaras guardadas de semilla 11, y se añadieron análisis exploratorios corregidos por multiplicidad, curvas y comparación de costo. La interpretación reconoce exposición previa del test en E1, dependencia residual entre escenas y ausencia de imágenes totalmente negativas. La réplica en condiciones originales del paper guía y varios diagnósticos exigidos por la guía permanecen pendientes; se identifican expresamente en el checklist. Este documento integra evidencia ejecutada, sin convertir planes en resultados.

## Estado de entrega

El benchmark principal está cerrado y auditado. **La entrega integral todavía no está completa.** Consultar el [checklist requisito por requisito](checklist_e2.md) y el [plan de cierre](cierre_e2.md). El profesor acepta diez modelos totales, según confirmación del usuario; no se asume una exención de los demás requisitos.

```{tableofcontents}
```
''')
    put(B/'_toc.yml','''format: jb-book
root: index
parts:
  - caption: Entregable 2
    chapters:
      - file: problema
      - file: eda
      - file: literatura
      - file: modelos_e2
      - file: metodos_e2
      - file: resultados_e2
      - file: complementos_e2
      - file: complementos_datos_e2
      - file: cierre_e2
      - file: checklist_e2
      - file: reproducibilidad_e2
  - caption: Antecedentes del Entregable 1
    chapters:
      - file: implementacion
      - file: resultados
''')
    rows=['| Red | Optimizador | Tasa | Weight decay |','|---|---|---:|---:|']
    for m,c in config['models'].items():rows.append(f"| {NAMES[m]} | {c['optimizer']} | {c['lr']} | {c['weight_decay']} |")
    put(B/'e2_optimizadores.inc','\n'.join(rows)+'\n\nSGD de TransUNet utiliza momentum 0,9. No hubo búsqueda posterior para seleccionar estas tasas.')
    results=(P/'resultados/auditoria_test_e2/CONCLUSIONES.md').read_text(encoding='utf-8')
    results=results.split('## Qué procede')[0].replace('# Resultados finales E2: auditoría y conclusiones','# Resultados E2: benchmark e inferencia')
    results=results.replace('si el conteo de máscaras vacías es cero','en este test, que no tiene máscaras vacías')
    results+='\n![Comparación validación y test, con intervalos de grupos y semillas fijas](figuras/e2/comparacion.png)\n'
    secondary=['\n## Métricas secundarias completas\n','| Red | IoU etiqueta | mIoU | Exactitud válida |','|---|---:|---:|---:|']
    for r in a['table']:
        runs=[read(P.parent/'resultados_test_e2_final'/f"{r['model']}_seed{seed}.json") for seed in config['seeds']]
        import numpy as np
        secondary.append('| '+NAMES[r['model']]+' | '+' | '.join(f"{np.mean([x['means'][k] for x in runs]):.4f} ± {np.std([x['means'][k] for x in runs],ddof=1):.4f}" for k in ['IoU','mIoU','accuracy'])+' |')
    results+='\n'.join(secondary)+'\n\nLa dispersión es DE entre cinco semillas, no error estándar. Las métricas promedian grupos con igual peso.\n'
    results+='\n## Ampliación estadística exploratoria\n\n'
    analysis=(P/'resultados/integracion_e2/ANALISIS.md').read_text(encoding='utf-8')
    results+='\n'.join(analysis.splitlines()[1:])
    results+='\n\n![Comparación de Dice frente a parámetros](figuras/e2/pareto.png)\n\n![Rangos medios y diferencia crítica de Nemenyi](figuras/e2/rangos_cd.png)\n\nEn el gráfico de rangos, dos rangos medios separados por más de la CD son diferentes según Nemenyi al 5 %, bajo el supuesto de bloques independientes. No es el mismo criterio que el contraste de medias contra FCN.\n'
    results+='\n## Sensibilidad y potencia de planificación\n\nUNet++ ilustra la necesidad de no elegir pruebas por su resultado: intercambio de medias y Wilcoxon de rangos no producen la misma decisión tras Holm. La distribución de errores puede combinar muchos casos competitivos y algunas caídas grandes. Se conservan ambas respuestas y sus hipótesis.\n\n'
    results+='| Red frente a FCN | DE de diferencia entre semillas | MDD aproximada con 5 semillas | R aproximado para δ=0,02 |\n|---|---:|---:|---:|\n'
    for c in s['contrasts']:results+=f"| {NAMES[c['model']]} | {c['paired_seed_difference_sd']:.4f} | {c['planning_MDD_R5_9contrasts']:.4f} | {c['planning_R_delta002_9contrasts']} |\n"
    results+='\nSon aproximaciones normales para planificar una evaluación futura, con α/9 y potencia objetivo 0,80. No justifican retrospectivamente el tamaño del experimento ni garantizan potencia de Wilcoxon.\n'
    results+='\n## Desempeño por tamaño anotado\n\nCuartiles definidos por fracción de área en train: '+', '.join(f'{v:.4f}' for v in s['strata_cutpoints_train'])+'. Las columnas muestran Dice y no representan pruebas entre subgrupos independientes.\n\n| Red | Q1 menor área | Q2 | Q3 | Q4 mayor área |\n|---|---:|---:|---:|---:|\n'
    for m in s['model_order']:
        ss=[x for x in s['strata'] if x['model']==m]
        results+='| '+NAMES[m]+' | '+' | '.join(f"{x['Dice']:.4f} [{x['CI95'][0]:.4f}, {x['CI95'][1]:.4f}]" for x in ss)+' |\n'
    results+='\n## Revisión cualitativa anterior al test\n\n![Casos dirigidos de validación con FCN, U-Net y TransUNet, semilla 11](figuras/e2/panel_validacion.png)\n\nSe seleccionaron seis casos de validación por criterios explícitos: peor y segundo peor promedio, menor área, mayor desacuerdo, caso central y mejor promedio, en grupos diferentes. La visualización fija semilla 11; no representa una frecuencia de errores ni una explicación Grad-CAM. La evidencia detallada se conserva en `resultados/revision_visual_e2/casos.json` y `OBSERVACIONES.md`. Los ejemplos muestran dificultades con oclusión, iluminación y cobertura de anotación; no se editaron etiquetas tras observar predicciones.\n'
    put(B/'resultados_e2.md',results)
    put(B/'reproducibilidad_e2.md',r'''
# Reproducción y archivos de entrega

## Regenerar el informe local

Desde la raíz `tesis_imo_para_codex`, con el entorno `.venv-libro` ya instalado:

```powershell
.\.venv-libro\Scripts\python.exe scripts\reproducir_entregable2.py
```

Este comando regenera análisis exploratorios, figuras, capítulos e índice, ejecuta las pruebas del análisis y compila el libro con advertencias tratadas como errores. Lee los resultados originales en las carpetas hermanas `resultados_test_e2_final` y `tesis_imo_e2_arquitecturas_reales_v2`, los manifiestos y datos locales. No entrena ni publica. La reproducción del informe no es una repetición independiente de inferencia.

El resultado se abre en `libro/_build/html/index.html`. Dependencias exactas del informe en {download}`requirements-libro.txt <../requirements-libro.txt>`. El comando supone la estructura local y datos disponibles; no descarga silenciosamente datasets.

## Reproducción experimental

Los notebooks de entrenamiento v2 y test cerrado están en la carpeta superior: `Benchmark_E2_Colab_v2.ipynb` y `Evaluacion_Final_E2_Colab.ipynb`, junto con sus ZIP versionados. El paquete antiguo de prototipos no describe este benchmark. Los pesos se conservan en las cincuenta carpetas originales y no se incluyen en el libro HTML.

La normalización y geometría se fijaron con train; cada corrida selecciona por validación. Test E2 se ejecutó con checkpoints y umbral fijados. Repetir entrenamientos/inferencia debe ser una decisión explícita de reproducción; no es necesario para recompilar el documento.

## Evidencia descargable

- {download}`Configuración aplicada <../config/experimento_e2.json>`
- {download}`Protocolo cerrado de test <../config/protocolo_test_e2_cerrado.json>`
- {download}`Fuentes de autores y hashes <../third_party/e2/sources.lock.json>`
- {download}`Auditoría del entrenamiento <../resultados/auditoria_e2_recibida/auditoria.json>`
- {download}`Auditoría del test <../resultados/auditoria_test_e2/AUDITORIA.json>`
- {download}`Análisis exploratorio completo <../resultados/integracion_e2/analisis.json>`
- {download}`Código del análisis <../scripts/analisis_integracion_e2.py>`
- {download}`Código de auditoría de test <../scripts/auditar_test_e2.py>`
- {download}`Pruebas de la integración <../tests/test_integracion_e2.py>`
- {download}`Checklist de entrega <../docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md>`
- {download}`Referencias BibTeX de las diez arquitecturas <referencias_e2.bib>`

Las auditorías verificaron la consistencia de los artefactos recibidos. Falta una repetición experimental independiente en entorno limpio que informe discrepancia de inferencia, no solo de agregación aritmética. Los originales, fotografías y pesos no deben publicarse sin revisar licencias. No se realizó push ni despliegue; cualquier publicación requiere autorización del usuario.
''')
    put(B/'checklist_e2.md','# Verificación de requisitos del Entregable 2\n\n```{include} ../docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md\n:start-line: 1\n```')
    # The architecture diagrams summarize mechanisms, not exact layer-by-layer graphs.
    cards=[
      ('fcn','CNN','1411.4038','Long, Jonathan and Shelhamer, Evan and Darrell, Trevor','2014','Fully Convolutional Networks for Semantic Segmentation',
       ['RGB 256×256','Conv 16/32/64\ny pooling','Cabeza 1×1','Upsampling\n1 logit'],
       'Implementación compacta local; no FCN-8s original, sin sus fusiones laterales. Dos reducciones espaciales.',r'$h_j=\mathrm{ReLU}(b_j+\sum_i W_{ji}*x_i)$; la interpolación devuelve la salida al lienzo.',
       'Costo convolucional O(HW k² Ci Co). Bajo tamaño; el mapa grueso puede perder detalle. Artículo fundacional excluido de la ventana reciente con justificación.'),
      ('segnet','CNN','1511.00561','Badrinarayanan, Vijay and Kendall, Alex and Cipolla, Roberto','2015','SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation',
       ['RGB','Encoder\npooling + índices','Unpooling\ncon índices','Conv + logit'],
       'Compacta local 16/32/64; no encoder VGG16 completo del artículo.',r'$u=\mathrm{Unpool}(z,\operatorname{argmaxpool}(h))$ reutiliza posiciones, no todos los valores del encoder.',
       'Convoluciones más almacenamiento de índices; ahorra mapas de skip pero no certifica menor latencia sin medición.'),
      ('unet','CNN','1505.04597','Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas','2015','U-Net: Convolutional Networks for Biomedical Image Segmentation',
       ['RGB','Encoder\n16/32/64','Decoder + skips\nconcatenados','Cabeza 1×1'],
       'U-Net compacta local; encoder propio, no ResNet-18. Dos niveles de reducción.',r'$d_l=\mathrm{Conv}([e_l,\mathrm{Up}(d_{l+1})])$ preserva detalle por concatenación.',
       'Convoluciones y activaciones de skip; fundamento médico, adaptación binaria HAZMAT sin preentrenamiento.'),
      ('deeplabv3plus','CNN','1802.02611','Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig','2018','Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation',
       ['RGB','ResNet-18\nSMP','ASPP + decoder\nfusión multiescala','Logit binario'],
       'SMP 0.5.0, ResNet-18; no Xception de los resultados principales del paper.',r'$(x*_r w)(i)=\sum_k x(i+r k)w(k)$; ASPP combina distintas dilataciones.',
       'Contexto multiescala y refinamiento. Cambian backbone, clases y receta; sus cifras originales no son una referencia numérica directamente comparable.'),
      ('unetplusplus','CNN','1912.05074','Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming','2019','UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation',
       ['RGB','ResNet-18\nSMP','Nodos anidados\nskips densos','Una cabeza\nbinaria'],
       'SMP sin supervisión profunda ni poda. Su encoder y tamaño difieren de la U-Net compacta.',r'$x^{i,j}=H([x^{i,0},\ldots,x^{i,j-1},U(x^{i+1,j-1})])$ representa la fusión anidada.',
       'Más conexiones y memoria de activaciones. Comparar estas redes no aísla el efecto de los skips.'),
      ('pidnet_s','CNN','2206.02066','Xu, Jiacong and Xiong, Zixiang and Bhattacharyya, Shankar P.','2022','PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers',
       ['RGB','Ramas P / I / D\ndetalle/contexto/borde','Fusión guiada\npor borde','Principal + aux\n+ borde'],
       'Código de autores; m=2, n=3, planes=32, ppm=96, head=128. Tres salidas conservadas.',r'$L=L_{principal}+0,4L_{aux}+1,0L_{borde}$ en nuestra receta; el objetivo de borde deriva de la máscara.',
       'La analogía PID es estructural; no es un controlador físico. Las pérdidas auxiliares impiden atribuir toda diferencia solo a la topología.'),
      ('segnext_t','CNN con atención convolucional','2209.08575','Guo, Meng-Hao and Lu, Cheng-Ze and Hou, Qibin and Liu, Zhengning and Cheng, Ming-Ming and Hu, Shi-Min','2022','SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation',
       ['RGB','MSCAN-T','LightHamHead\nmultiescala + NMF','Logit + resize'],
       'Fuentes de autores con capa de compatibilidad local; SyncBN adaptada a ejecución monodispositivo. No es un Transformer.',r'$\mathrm{Att}(F)=F\odot \mathrm{Conv}_{1\times1}(\sum_i \mathrm{Branch}_i(\mathrm{DWConv}(F)))$ resume la atención convolucional.',
       'Filtros separables multiescala; NMF de la cabeza puede usar inicialización aleatoria, por lo que se fija RNG de evaluación.'),
      ('segformer_b0','Transformer','2105.15203','Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M. and Luo, Ping','2021','SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers',
       ['RGB','MiT-B0\n4 escalas','MLP + resize\nfusión','Logit binario'],
       'SMP 0.5.0; encoder mit_b0, sin pesos externos. Paper guía; réplica original pendiente.',r'$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt d)V$; reducción espacial de K,V disminuye costo.',
       'Atención densa O(N²d), reducida según el factor espacial de cada etapa. No afirmar costo cuadrático idéntico al ViT estándar.'),
      ('transunet_2d','Híbrido CNN–Transformer','2102.04306','Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L. and Zhou, Yuyin','2021','TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation',
       ['RGB','R50 + tokens','ViT-B/16\n12 bloques','Decoder + 3 skips\nlogit'],
       'Código de autores; variante R50-ViT-B/16, tamaño 256, grid 16×16, n_skip=3. Sin carga de pesos.',r'$z_{l+1}=z_l+\mathrm{MSA}(\mathrm{LN}(z_l))$ seguido de bloque MLP residual; decoder recupera resolución.',
       'Atención de tokens y CNN; gran capacidad y costo. Su liderazgo de media no prueba que la atención cause la mejora.'),
      ('vmunet','SSM visual','2402.02491','Ruan, Jiacheng and Li, Jincheng and Xiang, Suncheng','2024','VM-UNet: Vision Mamba UNet for Medical Image Segmentation',
       ['RGB','VSS encoder\n[2,2,9,2]','VSS decoder\n[2,9,2,2]','Logit binario'],
       'VSSM de autores, drop_path=0,2; selective_scan CUDA real. No sustitución por GRU.',r'$h_t=\bar A_t h_{t-1}+\bar B_t x_t,\quad y_t=C_t h_t$ resume el mecanismo selectivo; escaneos visuales relacionan posiciones.',
       'Costo lineal en longitud para dimensiones fijas del estado. Requiere extensiones CUDA; no extrapolar eficiencia teórica a latencia no medida.')]
    text=['# Fichas de las diez implementaciones','',
      'Las fichas describen el código ejecutado. Cada diagrama es un esquema de mecanismos, no un volcado completo de capas. Las dimensiones exactas se consultan en el código y la configuración fijada. Las referencias arXiv conservan el año del preprint; no se confunde con el año de publicación editorial.',
      'Las cifras de este benchmark proceden de los artefactos auditados. Las tablas numéricas originales de los artículos, FLOPs adaptados y tiempos de los papers no están certificados por estas fichas; su cotejo continúa pendiente en el checklist. No se inventan cuartiles de conferencias.','']
    bib=[]
    for m,family,arxiv,authors,year,title,blocks,adapt,maths,limits in cards:
        fig,ax=plt.subplots(figsize=(12,1.7));ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,1.7)
        for i,label in enumerate(blocks):
            x=.1+i*3
            ax.add_patch(FancyBboxPatch((x,.35),2.65,1,boxstyle='round,pad=.08',facecolor='#E9F2F6',edgecolor='#176B87'))
            ax.text(x+1.325,.85,label,ha='center',va='center',fontsize=10)
            if i<3:ax.annotate('',xy=(x+2.93,.85),xytext=(x+2.68,.85),arrowprops={'arrowstyle':'->'})
        fig.tight_layout();fig.savefig(B/'figuras/e2'/f'diagrama_{m}.png',dpi=150);plt.close(fig)
        r=next(x for x in a['table'] if x['model']==m);c=config['models'][m]
        text += [f'## {NAMES[m]}', '',f'**Familia:** {family}. **Parámetros medidos:** {r["parameters"]:,}.',
          f'**Fuente primaria:** {authors.replace(" and ", "; ")} ({year}). *{title}*. [DOI del preprint](https://doi.org/10.48550/arXiv.{arxiv}); [texto y versiones](https://arxiv.org/abs/{arxiv}).',
          '',f'![Esquema de {NAMES[m]}](figuras/e2/diagrama_{m}.png)','',adapt,'',maths,'',limits,'',
          f'**Receta:** {c["optimizer"]}, lr={c["lr"]}, weight decay={c["weight_decay"]}; cinco semillas; mismo tamaño, criterio de parada y salida de un logit.',
          f'**Resultado local:** Dice test {r["Dice_test"]:.4f} ± {r["Dice_sd_seeds"]:.4f}; no sustituye el resultado del artículo original.','']
        bib.append(f'@misc{{{m},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{year}}},\n  eprint = {{{arxiv}}},\n  archivePrefix = {{arXiv}},\n  doi = {{10.48550/arXiv.{arxiv}}},\n  url = {{https://arxiv.org/abs/{arxiv}}}\n}}')
    put(B/'modelos_e2.md','\n'.join(text));put(B/'referencias_e2.bib','\n\n'.join(bib))
    print('Capítulos, tablas, referencias y diagramas E2 integrados.')

if __name__=='__main__':main()
