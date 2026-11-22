"""Puntuación definitiva de las candidatas con la rúbrica de PROTOCOLO_BUSQUEDA.md (seis componentes, 0-100).

Uso: python docs/revision_entregable2/puntuar_candidatos.py   (después de bibliometria_candidatos.py y cotejar_fichas.py)
Entradas: fichas_extraccion.json (evidencia por página), bibliometria_candidatos.json (citas y licencias).
Salidas: PUNTUACION_DEFINITIVA.md y puntuacion_definitiva.json.

Candidatas elegibles: las siete seleccionadas más las dos alternativas excluidas que pasan el filtro previo del
protocolo (salida de segmentación RGB 2D sin prompts): Attention U-Net y Swin-Unet. SAM y YOLO-seg no se puntúan:
el filtro previo los separa (prompts; instancia en vez de máscara semántica), ver PUNTUACION_Y_EXCLUSIONES.md §3.

Reglas operativas (fijadas aquí porque la escala del protocolo deja margen de interpretación):
- Pertinencia: 4 solo con evidencia en rótulos o dominio comparable (ninguna la tiene); 3 si el artículo trae un
  análisis cuantitativo DEDICADO a una dificultad de HAZMAT-13 (bordes, tamaño u objetos pequeños, corrupción),
  no una métrica general de la tabla principal; 2 si la dificultad es motivación o se muestra solo cualitativamente.
- Reproducibilidad: 3 código de autores y protocolo; 4 si además publica pesos entrenados o configuraciones
  verificables (README oficial, consultado por la API de GitHub el día de la corrida). Pesos solo de inicialización
  (ImageNet) no dan 4. Enlaces que el propio autor declara rotos no se cuentan.
- Evidencia: 2 baselines y partición clara con selección en validación; 3 añade repeticiones (corridas o validación
  cruzada) con contraste; 4 añade incertidumbre formal (IC o prueba) sobre corridas independientes; 1 si hay indicios
  de que las configuraciones se eligieron sobre el mismo conjunto que se reporta (sin conjunto de selección aparte).
- Viabilidad: 4 si se ejecutó un piloto dentro del presupuesto (las siete seleccionadas: benchmark de 5 semillas
  completado); 3 configuración pública plausible sin piloto. Esto favorece a las ya ejecutadas: por eso se reporta la
  sensibilidad sin este componente.
- Recencia: año de la versión cuyo contenido se evaluó.
- Citación: citas/año (OpenAlex, misma fecha) convertidas a percentil entre las nueve (scipy percentileofscore,
  kind='mean') × 10.
Primeros cinco componentes: peso × puntuación / 4.
"""
import itertools
import json
from pathlib import Path

from scipy.stats import percentileofscore

ROOT = Path(__file__).resolve().parent
PESOS = {'pertinencia': 25, 'reproducibilidad': 25, 'evidencia': 20, 'viabilidad': 10, 'recencia': 10}

# red: familia, {componente: (nivel 0-4, justificación con el campo de la ficha que la respalda)}
PUNTOS = {
    'DeepLabv3+': ('CNN', {
        'pertinencia': (3, 'Experimento trimap: mIoU en bandas alrededor del borde; la ganancia del decoder crece al estrechar la banda (ficha: dificultad compartida)'),
        'reproducibilidad': (4, 'Código oficial en tensorflow/models con model_zoo.md de checkpoints entrenados'),
        'evidencia': (2, 'Ablaciones en val de VOC; la mejor variante se reentrena para test. Sin repeticiones'),
        'viabilidad': (4, 'Entrenada en el benchmark (SMP, ResNet-18), 5 semillas dentro del presupuesto'),
        'recencia': (0, '2018; excepción fundacional documentada en el protocolo')}),
    'UNet++': ('CNN', {
        'pertinencia': (3, 'Análisis estratificado por tamaño de tumor en siete grupos con prueba t (Fig. 7)'),
        'reproducibilidad': (3, 'Código de autores; el README no publica pesos entrenados'),
        'evidencia': (4, '20 ensayos independientes, media ± DE, prueba t contra U-Net e IC95 (Fig. 4); parada temprana en validación'),
        'viabilidad': (4, 'Entrenada en el benchmark, 5 semillas'),
        'recencia': (1, 'IEEE TMI 2020')}),
    'PIDNet-S': ('CNN', {
        'pertinencia': (2, 'Bordes y objetos pequeños como motivación y en figuras cualitativas; la ablación de pérdida de frontera se mide en mIoU global, sin métrica de borde'),
        'reproducibilidad': (3, 'Código de autores (MIT); el autor declara rotos los enlaces de pesos y la alternativa no se comprobó'),
        'evidencia': (2, 'Ablaciones en val de Cityscapes; test tras reentrenar con train+val. Sin repeticiones'),
        'viabilidad': (4, 'Entrenada en el benchmark, 5 semillas'),
        'recencia': (3, 'CVPR 2023')}),
    'SegNeXt-T': ('CNN', {
        'pertinencia': (2, 'Escala como motivación del diseño multiescala; solo figura cualitativa de detalles'),
        'reproducibilidad': (4, 'Código de autores (Apache-2.0) con tabla de modelos entrenados y pesos de ImageNet'),
        'evidencia': (2, 'Comparaciones sobre validación de ADE20K y Cityscapes como reporte final. Sin repeticiones'),
        'viabilidad': (4, 'Entrenada en el benchmark, 5 semillas'),
        'recencia': (2, 'NeurIPS 2022')}),
    'SegFormer-B0': ('Transformer', {
        'pertinencia': (3, 'Robustez a 16 corrupciones (Cityscapes-C: ruido, desenfoque, clima) frente a DeepLabv3+ y otros'),
        'reproducibilidad': (4, 'Código y pesos entrenados; además, la réplica de evaluación de este trabajo reprodujo 37,43 mIoU con el checkpoint oficial'),
        'evidencia': (2, 'Tablas de validación como reporte final; test de Cityscapes con val fusionado a train. Sin repeticiones'),
        'viabilidad': (4, 'Entrenada en el benchmark, 5 semillas'),
        'recencia': (2, 'NeurIPS 2021')}),
    'TransUNet': ('Híbrido CNN-Transformer', {
        'pertinencia': (3, 'Ablación de conexiones con ganancia desagregada en órganos pequeños (aorta, vesícula)'),
        'reproducibilidad': (3, 'Código de autores (Apache-2.0); solo pesos de ViT de ImageNet, no del modelo entrenado'),
        'evidencia': (1, 'Synapse sin conjunto de selección aparte: las configuraciones se eligen sobre los 12 casos que se reportan'),
        'viabilidad': (4, 'Entrenada en el benchmark, 5 semillas'),
        'recencia': (2, 'Preprint 2021')}),
    'VM-UNet': ('SSM', {
        'pertinencia': (2, 'Objetos pequeños y bordes complejos solo en figuras cualitativas'),
        'reproducibilidad': (3, 'Código de autores (Apache-2.0); el README no publica pesos del modelo entrenado'),
        'evidencia': (1, 'Partición train/test sin validación; el dropout se elige comparando sobre el mismo test'),
        'viabilidad': (4, 'Entrenada en el benchmark, 5 semillas'),
        'recencia': (3, 'Preprint 2024 (versión editorial ACM TOMM 2025)')}),
    'Attention U-Net': ('CNN', {
        'pertinencia': (2, 'Páncreas (objeto pequeño y variable) como motivación; resultados por órgano frente a U-Net sin análisis dedicado'),
        'reproducibilidad': (3, 'Código de autores (MIT); sin pesos entrenados en el README'),
        'evidencia': (3, 'Validación cruzada de 5 particiones en CT-82, valores p frente a U-Net y evaluación externa CT-150 → CT-82; sin conjunto de validación'),
        'viabilidad': (3, 'Configuración pública plausible; sin piloto'),
        'recencia': (0, '2018; sin excepción fundacional')}),
    'Swin-Unet': ('Transformer', {
        'pertinencia': (2, 'Bordes afirmados con HD y figura cualitativa; sin análisis dedicado'),
        'reproducibilidad': (3, 'Código de autores y pesos de Swin-T de ImageNet; el repositorio no declara licencia'),
        'evidencia': (1, 'Ablaciones sobre el test de Synapse y configuración elegida con esos resultados'),
        'viabilidad': (3, 'Configuración pública plausible (entrada 224); sin piloto'),
        'recencia': (2, 'ECCV Workshops 2022')}),
}
SELECCIONADAS = ['DeepLabv3+', 'UNet++', 'PIDNet-S', 'SegNeXt-T', 'SegFormer-B0', 'TransUNet', 'VM-UNet']
# Rango de la sede (no es cuartil: el protocolo prohíbe asignar cuartiles a conferencias).
SEDE = {
    'DeepLabv3+': 'ECCV, CORE A* (ICORE2026)', 'UNet++': 'IEEE TMI (revista); cuartil no verificado',
    'PIDNet-S': 'CVPR, CORE A* (ICORE2026)', 'SegNeXt-T': 'NeurIPS, CORE A* (ICORE2026)',
    'SegFormer-B0': 'NeurIPS, CORE A* (ICORE2026)', 'TransUNet': 'Preprint arXiv; no aplica',
    'VM-UNet': 'ACM TOMM (revista, 2025); cuartil no verificado', 'Attention U-Net': 'MIDL 2018; sin rango CORE',
    'Swin-Unet': 'Taller de ECCV 2022; el rango de ECCV no se extiende a sus talleres',
}


def main():
    biblio = json.loads((ROOT / 'bibliometria_candidatos.json').read_text(encoding='utf-8'))
    fichas = {f['red'] for f in json.loads((ROOT / 'fichas_extraccion.json').read_text(encoding='utf-8'))}
    assert fichas == set(PUNTOS), 'Cada candidata puntuada necesita su ficha cotejada'
    cpa = {r: biblio['candidatas'][r]['citas_por_anio'] for r in PUNTOS}
    tabla = {}
    for red, (familia, comp) in PUNTOS.items():
        base = {k: PESOS[k] * comp[k][0] / 4 for k in PESOS}
        cit = percentileofscore(list(cpa.values()), cpa[red], kind='mean') / 100 * 10
        sin_cit = sum(base.values())
        tabla[red] = {'familia': familia, 'niveles': {k: comp[k][0] for k in PESOS},
                      'justificacion': {k: comp[k][1] for k in PESOS}, 'puntos': base,
                      'citas': biblio['candidatas'][red]['citas_max'], 'citas_por_anio': cpa[red],
                      'citacion_10': round(cit, 2), 'total_100': round(sin_cit + cit, 2),
                      'sin_citacion_90_a_100': round(sin_cit / 90 * 100, 2),
                      'sin_viabilidad_90_a_100': round((sin_cit - base['viabilidad'] + cit) / 90 * 100, 2),
                      'licencia': biblio['candidatas'][red]['codigo'].get('spdx'), 'sede': SEDE[red],
                      'seleccionada': red in SELECCIONADAS}

    familias = {'CNN', 'Transformer', 'Híbrido CNN-Transformer', 'SSM'}

    def mejor(clave):
        opciones = [c for c in itertools.combinations(tabla, 7) if {tabla[r]['familia'] for r in c} >= familias]
        return sorted(max(opciones, key=lambda c: (sum(tabla[r][clave] for r in c),
                                                   sum(tabla[r]['niveles']['reproducibilidad'] for r in c))))

    optimos = {k: mejor(k) for k in ('total_100', 'sin_citacion_90_a_100', 'sin_viabilidad_90_a_100')}
    salida = {'fecha_citas': biblio['fecha_consulta'], 'candidatas': tabla, 'optimo_con_diversidad': optimos,
              'seleccion_operativa': sorted(SELECCIONADAS)}
    (ROOT / 'puntuacion_definitiva.json').write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding='utf-8')

    orden = sorted(tabla, key=lambda r: -tabla[r]['total_100'])
    md = ['# Puntuación definitiva de las candidatas', '',
          f"**Generado por `puntuar_candidatos.py`.** Rúbrica de `PROTOCOLO_BUSQUEDA.md` con sus seis componentes. "
          f"Evidencia de cada nivel: `FICHAS_COTEJADAS.md` (fragmento y página del PDF). Citas de OpenAlex consultadas el "
          f"{biblio['fecha_consulta']} (máximo entre las obras de cada artículo; detalle en `bibliometria_candidatos.json`). "
          'Las reglas operativas de cada nivel están en el encabezado del script. Sustituye a la tabla de '
          '`PUNTUACION_Y_EXCLUSIONES.md`, que no tenía citación ni cotejo con el PDF.', '',
          '| Candidata | Familia | Pert. (25) | Repr. (25) | Evid. (20) | Viab. (10) | Rec. (10) | Citas/año | Citación (10) | **Total** | Sin citación | Sin viabilidad | Estado |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    num = lambda x: f'{x:.2f}'.replace('.', ',')
    for r in orden:
        t = tabla[r]
        p = t['puntos']
        md.append(f"| {r} | {t['familia']} | {num(p['pertinencia'])} | {num(p['reproducibilidad'])} | {num(p['evidencia'])} | "
                  f"{num(p['viabilidad'])} | {num(p['recencia'])} | {str(t['citas_por_anio']).replace('.', ',')} | "
                  f"{num(t['citacion_10'])} | **{num(t['total_100'])}** | {num(t['sin_citacion_90_a_100'])} | "
                  f"{num(t['sin_viabilidad_90_a_100'])} | {'seleccionada' if t['seleccionada'] else 'excluida'} |")
    md += ['', '"Sin citación" y "sin viabilidad" reescalan a 0-100 los 90 puntos restantes (sensibilidad que pide el '
           'protocolo y control del sesgo a favor de las redes ya ejecutadas).', '', '## Selección óptima con la restricción de cuatro familias', '',
           'Siete de nueve, con al menos una CNN, un Transformer, un híbrido y un SSM; desempate por reproducibilidad.', '',
           '| Criterio | Conjunto óptimo | ¿Coincide con la selección operativa? | Cambio |', '|---|---|---|---|']
    for k, nombre in (('total_100', 'Total'), ('sin_citacion_90_a_100', 'Sin citación'), ('sin_viabilidad_90_a_100', 'Sin viabilidad')):
        o = optimos[k]
        entra = sorted(set(o) - set(SELECCIONADAS))
        sale = sorted(set(SELECCIONADAS) - set(o))
        md.append(f"| {nombre} | {', '.join(o)} | {'sí' if not entra else 'no'} | "
                  f"{'—' if not entra else 'entra ' + ', '.join(entra) + '; sale ' + ', '.join(sale)} |")
    md += ['', '## Justificación por componente', '']
    for r in orden:
        t = tabla[r]
        md += [f"**{r}** ({t['sede']}; licencia del código: {t['licencia'] or 'sin licencia declarada'}; "
               f"{t['citas']} citas en OpenAlex).", '']
        md += [f"- {k.capitalize()} {t['niveles'][k]}/4: {t['justificacion'][k]}." for k in PESOS]
        md.append('')
    md += ['## Cuartiles', '',
           'El protocolo pide fuente, categoría y año del cuartil y prohíbe asignarlo a conferencias. Cinco candidatas son '
           'de conferencia (rango CORE en la tabla de justificación, consultado en portal.core.edu.au, ICORE2026), una es '
           'preprint y una de taller. Las dos de revista (IEEE TMI para UNet++ y ACM TOMM para VM-UNet) quedan sin cuartil: '
           'SCImago exige una verificación de navegador que no se automatiza. El cuartil no entra en la rúbrica, de modo '
           'que el total no depende de él.', '']
    (ROOT / 'PUNTUACION_DEFINITIVA.md').write_text('\n'.join(md), encoding='utf-8')
    for r in orden:
        print(f"{r:16s} {tabla[r]['total_100']:6.2f} sin_cit {tabla[r]['sin_citacion_90_a_100']:6.2f} sin_viab {tabla[r]['sin_viabilidad_90_a_100']:6.2f}")
    print(json.dumps(optimos, ensure_ascii=False))


if __name__ == '__main__':
    main()
