"""Genera el informe ejecutivo en PDF usando matplotlib.backends.backend_pdf."""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Configurar directorio temporal para matplotlib
os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', 'C:/Users/luisa/AppData/Local/Temp')

PDF_PATH = Path(r"C:\Users\luisa\Downloads\DEEP LEARNING\tesis_imo_para_codex\docs\revision_entregable2\Informe_Estado_Entregable2.pdf")

def create_page():
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150) # A4 en pulgadas
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    return fig, ax

def generate_pdf():
    with PdfPages(PDF_PATH) as pdf:
        # ==========================================
        # PÁGINA 1: Introducción, E1 y Búsqueda PRISMA
        # ==========================================
        fig, ax = create_page()
        
        # Encabezado
        ax.text(0.08, 0.94, "INFORME DE ESTADO Y HOJA DE RUTA — ENTREGABLE 2", 
                fontsize=15, weight='bold', color='#1a365d')
        ax.plot([0.08, 0.92], [0.93, 0.93], color='#2b6cb0', lw=2)
        
        meta = ("Proyecto: Segmentación Semántica Binaria HAZMAT/IMDG | Maestría en Estadística Aplicada\n"
                "Fecha de corte: 18 de noviembre de 2026 | Propósito: Puesta al día para Daniel y el equipo")
        ax.text(0.08, 0.905, meta, fontsize=8.5, color='#4a5568')
        
        # Caja resumen
        rect = plt.Rectangle((0.08, 0.82), 0.84, 0.07, facecolor='#ebf8ff', edgecolor='#3182ce', lw=1.2)
        ax.add_patch(rect)
        resumen_txt = (
            "RESUMEN EJECUTIVO: Se completó la revisión bibliográfica sistemática bajo PRISMA 2020\n"
            "(9.646 registros importados de Scopus/WoS, 6.325 únicos). Se auditaron 7 nuevas arquitecturas\n"
            "para cubrir 4 familias distintas (10 modelos totales). Se compiló el Jupyter Book y se preparó\n"
            "el entorno de entrenamiento en Colab/VSCode con checkpoints atómicos a prueba de desconexiones."
        )
        ax.text(0.10, 0.835, resumen_txt, fontsize=8, color='#1a365d', weight='bold')

        # Sección 1
        ax.text(0.08, 0.785, "1. ¿Qué está hecho? (Línea Base y Entregable 1)", fontsize=11, weight='bold', color='#2b6cb0')
        ax.plot([0.08, 0.92], [0.78, 0.78], color='#e2e8f0', lw=1)
        
        e1_txt = (
            "• Auditoría de datos no destructiva: 1.685 fotografías de HAZMAT VOC2012 analizadas;\n"
            "  1.451 pares imagen/máscara técnicamente válidos aislados tras control de calidad riguroso.\n"
            "• Partición agrupada anti-fuga: Detección de 748 pares de alta similitud perceptual;\n"
            "  imágenes agrupadas en clusters para evitar fuga entre splits (1.015 train, 218 val, 218 test).\n"
            "• Línea base E1 ejecutada: 15 corridas históricas (FCN, SegNet, U-Net) con 5 semillas comunes\n"
            "  (11, 23, 37, 53, 71) a 256×256 con Adam (lr=0,001). FCN obtuvo el mejor rendimiento inicial.\n"
            "• Test reservado congelado: Selección de modelos exclusivamente por validación; test intacto."
        )
        ax.text(0.09, 0.68, e1_txt, fontsize=8.5, color='#2d3748', linespacing=1.3)

        # Sección 2
        ax.text(0.08, 0.645, "2. ¿Qué se adelantó hoy? (Revisión Sistemática y Conciliación)", fontsize=11, weight='bold', color='#2b6cb0')
        ax.plot([0.08, 0.92], [0.64, 0.64], color='#e2e8f0', lw=1)
        
        prisma_txt = (
            "• Búsqueda sistemática indexada (Q0–Q4): 9.646 registros totales importados\n"
            "  (6.399 Scopus + 3.247 Web of Science Core Collection). Consultas transversales Q5 y Q6\n"
            "  omitidas por decisión explícita tras incidencias de sesión institucional.\n"
            "• Deduplicación y conciliación: 3.321 duplicados consolidados, resultando en 6.325 registros\n"
            "  únicos preservando identificadores estables (E2-XXXX).\n"
            "• Resolución de 3 discrepancias DOI en Crossref:\n"
            "  - MCV-UNet: Se adoptó 10.7717/peerj-cs.2146 (oficial con guión).\n"
            "  - Multi-level spatial attention: Se adoptó 10.1504/ijes.2021.116134 (descarte ID interno Inderscience).\n"
            "  - OCT Inpainting: Se adoptó 10.3788/cjl202148.1507004 (formato correcto Chinese Journal of Lasers).\n"
            "• Cribado automatizado por título/resumen (6.325 registros):\n"
            "  - 2.186 retenidos: 20 dominio HAZMAT (RQ-L1), 507 arquitecturas (RQ-L2), 1.659 mecanismos (RQ-L3).\n"
            "  - 2.052 excluidos: 1.732 dominios no transferibles, 248 sólo detección de cajas, 72 no experimentales.\n"
            "  - 2.087 en zona de duda (inspección secundaria dirigida si se requiere)."
        )
        ax.text(0.09, 0.455, prisma_txt, fontsize=8.5, color='#2d3748', linespacing=1.25)

        # Diagrama de flujo conceptual en texto
        ax.text(0.08, 0.42, "Flujo Metodológico PRISMA 2020 Consolidado", fontsize=9.5, weight='bold', color='#1a365d')
        rect_flow = plt.Rectangle((0.08, 0.10), 0.84, 0.30, facecolor='#f7fafc', edgecolor='#cbd5e0', lw=1)
        ax.add_patch(rect_flow)
        
        flow_txt = (
            "  IDENTIFICACIÓN: Scopus (6.399) + WoS (3.247) ──► 9.646 registros brutos\n"
            "       │\n"
            "       ▼ Deduplicación técnica y resolución de 3 conflictos DOI\n"
            "  CORPUS ÚNICO CONCILIADO: 6.325 registros únicos evaluables\n"
            "       │\n"
            "       ├─────────────────────────────────────────┐\n"
            "       ▼                                         ▼\n"
            "  EXCLUIDOS: 2.052 registros                RETENIDOS / ELEGIBLES: 2.186 registros\n"
            "    • Dominio no transferible: 1.732          • Dominio HAZMAT directo (RQ-L1): 20\n"
            "    • Detección sin máscaras: 248             • Arquitecturas directas (RQ-L2): 507\n"
            "    • Actas / tipo no exp.: 72                • Mecanismos frontera (RQ-L3): 1.659\n"
            "                                              (Zona de duda reservada: 2.087)\n"
            "                                                 │\n"
            "                                                 ▼ Auditoría técnica documental completa\n"
            "  INCLUSIÓN EN BENCHMARK: 10 modelos representativos de 4 familias distintas"
        )
        ax.text(0.10, 0.12, flow_txt, fontsize=7.5, family='monospace', color='#2d3748')
        
        ax.text(0.50, 0.04, "Página 1 de 2", fontsize=8, color='#718096', ha='center')
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================
        # PÁGINA 2: Las 10 Redes, Experimento y Próximos Pasos
        # ==========================================
        fig, ax = create_page()
        
        ax.text(0.08, 0.94, "BENCHMARK DE 10 MODELOS Y HOJA DE RUTA", fontsize=15, weight='bold', color='#1a365d')
        ax.plot([0.08, 0.92], [0.93, 0.93], color='#2b6cb0', lw=2)
        
        ax.text(0.08, 0.90, "2.2 Cobertura de 4 Familias Arquitectónicas (10 Modelos Totales)", fontsize=11, weight='bold', color='#2b6cb0')
        ax.plot([0.08, 0.92], [0.895, 0.895], color='#e2e8f0', lw=1)
        
        # Tabla de modelos
        table_data = [
            ["Familia", "Modelo", "Mecanismo Principal", "Puntaje", "Estado"],
            ["1. CNN Clásica", "FCN / SegNet / U-Net", "Líneas base históricas de referencia E1", "Ref.", "Ejecutado E1"],
            ["1. CNN Multiescala", "DeepLabv3+", "Pirámide dilatada ASPP + decoder contornos", "73.61", "Auditado SMP"],
            ["1. CNN Anidada", "UNet++", "Skip connections anidados y densos (ResNet-18)", "76.39", "Auditado SMP"],
            ["1. CNN Multirrama", "PIDNet-S", "3 ramas: Detalle (P), Contexto (I), Borde (D)", "88.89", "Auditado Oficial"],
            ["1. CNN Atención", "SegNeXt-T", "Atención convolucional multiescala (MSCAN)", "69.44", "Auditado Oficial"],
            ["2. Transformer", "SegFormer-B0", "MiT-B0 jerárquico + MLP decoder multiescala", "79.17", "Paper Guía"],
            ["3. Híbrido", "TransUNet 2D", "ResNet-50 + ViT-B/16 + decoder U-Net", "61.11", "Condic. GPU"],
            ["4. SSM (Mamba)", "VM-UNet", "Modelos de espacio de estados 2D (VSSM)", "63.89", "Condic. Linux"]
        ]
        
        # Dibujar tabla estilizada con matplotlib
        y_top = 0.87
        row_h = 0.032
        col_x = [0.08, 0.25, 0.42, 0.74, 0.82, 0.92]
        
        for r_idx, row in enumerate(table_data):
            y_curr = y_top - r_idx * row_h
            bg_color = '#edf2f7' if r_idx == 0 else ('#f7fafc' if r_idx % 2 == 1 else '#ffffff')
            weight = 'bold' if r_idx == 0 else 'normal'
            rect_row = plt.Rectangle((0.08, y_curr - row_h + 0.005), 0.84, row_h, facecolor=bg_color, edgecolor='#cbd5e0', lw=0.5)
            ax.add_patch(rect_row)
            
            for c_idx in range(len(row)):
                ax.text(col_x[c_idx] + 0.01, y_curr - 0.015, row[c_idx], fontsize=7.2, weight=weight, color='#2d3748')

        # Paper guía y Jupyter book
        ax.text(0.08, 0.54, "2.3 Paper Guía y Cierre Documental", fontsize=10, weight='bold', color='#1a365d')
        guia_txt = (
            "• Paper Guía seleccionado: SegFormer-B0 (Xie et al., NeurIPS 2021). Meta de reproducción:\n"
            "  37,4 % mIoU en ADE20K (inferencia a escala única). Adaptación a HAZMAT es benchmark independiente.\n"
            "• Jupyter Book compilado: Capítulo libro/literatura.md actualizado con PRISMA y los 10 modelos.\n"
            "  Compilación HTML exitosa en libro/_build/html/index.html."
        )
        ax.text(0.09, 0.465, guia_txt, fontsize=8, color='#2d3748', linespacing=1.2)

        # Preparación Colab
        ax.text(0.08, 0.43, "2.4 Preparación de Entrenamiento en Colab/VS Code con Persistencia", fontsize=10, weight='bold', color='#1a365d')
        colab_txt = (
            "• Adaptadores verificados (scripts/modelos_e2.py): 10 redes aceptan tensores (B, 3, 256, 256)\n"
            "  y entregan un logit binario sin activación sigmoid final. Probados en forward pass sintético.\n"
            "• Runner resiliente (scripts/experimento_e2_colab.py) y Cuaderno (notebooks/benchmark_e2_colab.ipynb):\n"
            "  1. Guardado atómico por época (last.tmp -> last.pt) para tolerar desconexiones de Colab.\n"
            "  2. Reanudación automática: al relanzar detecta la época pendiente sin duplicar cómputo.\n"
            "  3. Selección estricta por validación agrupada (best.pt con Dice por grupo).\n"
            "  4. Almacenamiento persistente en Google Drive (/content/drive/MyDrive/tesis_imo_experimentos_e2/)."
        )
        ax.text(0.09, 0.305, colab_txt, fontsize=8, color='#2d3748', linespacing=1.2)

        # Sección 3
        ax.text(0.08, 0.275, "3. ¿Qué se va a hacer? (Hoja de Ruta y Próximos Pasos)", fontsize=11, weight='bold', color='#2b6cb0')
        ax.plot([0.08, 0.92], [0.27, 0.27], color='#e2e8f0', lw=1)
        
        todo_txt = (
            "1. Conectar VS Code a GPU Colab y ejecutar el benchmark de 10 modelos × 5 semillas (50 corridas).\n"
            "2. Monitorear curvas de aprendizaje y early stopping (paciencia 5 épocas sobre validación).\n"
            "3. Evaluar los modelos ganadores una única vez sobre el test reservado con pruebas pareadas de Holm.\n"
            "4. Coordinación GitHub: Revisar primero los cambios de Daniel en el remoto antes de hacer push."
        )
        ax.text(0.09, 0.165, todo_txt, fontsize=8.5, color='#2d3748', linespacing=1.3)

        # Caja de advertencia
        rect_warn = plt.Rectangle((0.08, 0.08), 0.84, 0.055, facecolor='#fffaf0', edgecolor='#dd6b20', lw=1)
        ax.add_patch(rect_warn)
        warn_txt = (
            "REGLA DE INTEGRIDAD: El split de test permanece congelado. No se altera la base de datos\n"
            "ni se usa test para detener épocas. Si la GPU se desconecta, el runner reanuda automáticamente."
        )
        ax.text(0.10, 0.095, warn_txt, fontsize=7.8, color='#744210', weight='bold')

        ax.text(0.50, 0.04, "Página 2 de 2", fontsize=8, color='#718096', ha='center')
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF generado exitosamente en: {PDF_PATH}")

if __name__ == '__main__':
    generate_pdf()
