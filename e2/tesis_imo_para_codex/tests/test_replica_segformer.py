"""Pruebas de la réplica de SegFormer-B0 con un modelo y un ADE20K simulados.

No descargan nada: validan preprocesamiento, etiquetas, métrica, bootstrap y
criterio. La carga del checkpoint de Hugging Face solo se ejecuta en Colab.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import replica_segformer_ade as r  # noqa: E402


class PreprocesamientoTests(unittest.TestCase):
    def test_reescalado_como_mmcv(self):
        self.assertEqual(r.tamano_reescalado(512, 683), (512, 683))
        self.assertEqual(r.tamano_reescalado(256, 256), (512, 512))
        # lado corto manda: 512/1000 < 2048/3000
        self.assertEqual(r.tamano_reescalado(1000, 3000), (512, 1536))
        # lado largo manda: 2048/4000 < 512/600
        self.assertEqual(r.tamano_reescalado(600, 4000), (307, 2048))

    def test_alineado_a_32(self):
        self.assertEqual(r.tamano_alineado(512, 683), (512, 704))
        self.assertEqual(r.tamano_alineado(307, 2048), (320, 2048))

    def test_preprocesos_forma_y_normalizacion(self):
        rgb = np.full((300, 400, 3), 128, dtype=np.uint8)
        x, tam = r.preprocesar_primario(rgb)
        self.assertEqual(tam, (512, 704))  # 300x400 -> 512x683 -> 512x704
        self.assertEqual(x.shape, (3, 512, 704))
        np.testing.assert_allclose(x[:, 0, 0], (128 - r.MEDIA_0_255) / r.DE_0_255, rtol=1e-5)
        y, tam2 = r.preprocesar_hf(rgb)
        self.assertEqual((y.shape, tam2), ((3, 512, 512), (512, 512)))
        np.testing.assert_allclose(y[:, 0, 0], (128 / 255 - r.MEDIA_IMAGENET) / r.DE_IMAGENET, rtol=1e-5)

    def test_reducir_etiqueta_cero(self):
        e = np.array([[0, 1, 150, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(r.reducir_etiqueta_cero(e), [[255, 0, 149, 255]])


class MetricaTests(unittest.TestCase):
    def test_interseccion_union_contra_fuerza_bruta(self):
        rng = np.random.default_rng(0)
        gt = rng.integers(0, 6, (40, 50))
        gt[0, :10] = 255
        pred = rng.integers(0, 6, (40, 50))
        inter, union, area = r.interseccion_union(pred, gt, num_clases=6)
        v = gt != 255
        for k in range(6):
            p, g = (pred == k) & v, (gt == k) & v
            self.assertEqual(inter[k], (p & g).sum())
            self.assertEqual(union[k], (p | g).sum())
            self.assertEqual(area[k], g.sum())

    def test_miou_acumulado_ignora_clases_sin_union(self):
        inter = np.array([[1, 0, 0], [1, 0, 0]])
        union = np.array([[2, 0, 4], [2, 0, 0]])
        miou, iou = r.miou_acumulado(inter, union)
        self.assertTrue(np.isnan(iou[1]))
        self.assertAlmostEqual(miou, (0.5 + 0.0) / 2)

    def test_acumulado_no_es_media_por_imagen(self):
        inter = np.array([[9, 0], [0, 1]])
        union = np.array([[10, 0], [0, 10]])
        self.assertAlmostEqual(r.miou_acumulado(inter, union)[0], (0.9 + 0.1) / 2)
        self.assertAlmostEqual(r.miou_medio_por_imagen(inter, union), (0.9 + 0.1) / 2)
        inter = np.array([[9, 1], [0, 1]])
        union = np.array([[10, 10], [0, 1]])
        self.assertAlmostEqual(r.miou_acumulado(inter, union)[0], (0.9 + 2 / 11) / 2)
        self.assertAlmostEqual(r.miou_medio_por_imagen(inter, union), ((0.9 + 0.1) / 2 + 1.0) / 2)

    def test_bootstrap_reproducible_y_centrado(self):
        rng = np.random.default_rng(1)
        union = rng.integers(50, 100, (300, 5))
        inter = (union * rng.uniform(0.3, 0.7, (300, 5))).astype(int)
        a = r.bootstrap_miou(inter, union, replicas=500, semilla=7)
        b = r.bootstrap_miou(inter, union, replicas=500, semilla=7)
        np.testing.assert_array_equal(a, b)
        punto = r.miou_acumulado(inter, union)[0]
        lo, hi = np.percentile(a, [2.5, 97.5])
        self.assertTrue(lo < punto < hi)
        self.assertLess(hi - lo, 0.05)

    def test_veredicto(self):
        self.assertEqual(r.veredicto(37.2, [36.9, 37.5])['veredicto'], 'replicado')
        self.assertEqual(r.veredicto(37.0, [36.2, 37.8])['veredicto'], 'compatible')
        self.assertEqual(r.veredicto(35.9, [35.5, 36.3])['veredicto'], 'no_replicado')
        self.assertTrue(r.veredicto(37.2, [36.9, 37.5])['reportado_dentro_IC95'])


class EvaluacionSimuladaTests(unittest.TestCase):
    """ADE20K de mentira: el color de cada píxel codifica su clase; el modelo la lee a 1/4."""

    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        raiz = Path(self.tmp.name)
        (raiz / 'images/validation').mkdir(parents=True)
        (raiz / 'annotations/validation').mkdir(parents=True)
        rng = np.random.default_rng(3)
        for i, (h, w) in enumerate([(240, 320), (300, 200), (256, 256)]):
            etiqueta = np.zeros((h, w), dtype=np.uint8)  # 0 = ignorado tras reducir
            etiqueta[:, : w // 2] = 1 + i                 # clase i (tras reducir)
            etiqueta[: h // 2, w // 2:] = 10              # clase 9
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[..., 0] = etiqueta * 20                   # canal R codifica la etiqueta
            rgb[..., 1] = rng.integers(0, 5, (h, w))      # ruido leve
            Image.fromarray(rgb).save(raiz / f'images/validation/ADE_val_{i:08d}.jpg', quality=100)
            Image.fromarray(etiqueta).save(raiz / f'annotations/validation/ADE_val_{i:08d}.png')
        self.raiz = raiz

    def tearDown(self):
        self.tmp.cleanup()

    def modelo_lector(self, x):
        """Lee el canal R desnormalizado y produce logits one-hot a 1/4 de resolución."""
        import torch
        import torch.nn.functional as F
        r0 = x[:, 0] * float(r.DE_0_255[0]) + float(r.MEDIA_0_255[0])
        clase = torch.clamp(torch.round(r0 / 20) - 1, 0, r.NUM_CLASES - 1).long()
        clase = clase[:, ::4, ::4]
        return F.one_hot(clase, r.NUM_CLASES).permute(0, 3, 1, 2).float() * 10

    def test_extremo_a_extremo(self):
        import torch
        pares = r.pares_validacion(self.raiz)
        self.assertEqual(len(pares), 3)
        areas = r.evaluar(self.modelo_lector, pares, r.preprocesar_primario, torch.device('cpu'), avisar=0)
        self.assertEqual(areas['inter'].shape, (3, r.NUM_CLASES))
        miou, iou = r.miou_acumulado(areas['inter'], areas['union'])
        # Las clases reales se recuperan casi completas; el lector a 1/4 solo falla
        # en los bordes, donde la interpolación de colores crea clases espurias con IoU 0.
        self.assertTrue(all(iou[k] > 0.9 for k in [0, 1, 2, 9]))
        self.assertLess(miou, 1.0)
        protocolo = json.loads((PROJECT / 'config/replica_segformer.json').read_text(encoding='utf-8'))
        protocolo['incertidumbre']['replicas'] = 200
        resumen = r.resumir(areas, protocolo, 'primaria', {'modelo': 'simulado'})
        self.assertEqual(resumen['imagenes'], 3)
        self.assertIn(resumen['criterio']['veredicto'], {'replicado', 'compatible', 'no_replicado'})
        self.assertLessEqual(resumen['IC95_pct'][0], resumen['mIoU_pct'] + 1e-9)
        json.dumps(resumen)  # serializable

    def test_faltante_de_anotacion_detiene(self):
        next((self.raiz / 'annotations/validation').glob('*.png')).unlink()
        with self.assertRaises(FileNotFoundError):
            r.pares_validacion(self.raiz)


class ProtocoloTests(unittest.TestCase):
    def test_protocolo_coherente_con_script(self):
        p = json.loads((PROJECT / 'config/replica_segformer.json').read_text(encoding='utf-8'))
        self.assertEqual(p['paper_guia']['valor_reportado_miou'], 37.4)
        self.assertEqual(p['criterio_a_priori']['delta_puntos_miou'], 1.0)
        self.assertEqual(p['checkpoint']['sha256_lfs'], r.HF_SHA256)
        self.assertIn(r.HF_REVISION, p['checkpoint']['fuente_principal'])


if __name__ == '__main__':
    unittest.main()
