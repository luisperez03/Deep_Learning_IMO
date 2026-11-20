"""Pruebas del análisis de la limitación por área con datos simulados (ítem 33)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import limitacion_area_segformer as L  # noqa: E402

EDA = json.loads(L.EDA.read_text(encoding='utf-8'))
PROTOCOLO = json.loads(L.PROTOCOLO.read_text(encoding='utf-8'))


def correr(escenario, replicas=1000):
    protocolo = json.loads(json.dumps(PROTOCOLO))
    protocolo['incertidumbre']['replicas'] = replicas
    return L.ejecutar(L.simular(EDA, escenario), EDA, protocolo, f'SIMULADO ({escenario})', 'x' * 64)


class TercilesTests(unittest.TestCase):
    def test_cada_grupo_en_un_solo_tercil(self):
        orden, area, mapa = L.grupos_validacion(EDA)
        tercil, cortes = L.terciles(area)
        self.assertEqual(len(orden), 122)
        self.assertEqual(len(mapa), 218)
        self.assertEqual(len(tercil), len(orden))
        self.assertTrue(set(np.bincount(tercil)) <= {40, 41, 42})
        self.assertTrue(area[tercil == 0].max() <= cortes[0] < area[tercil == 2].min())

    def test_holm(self):
        self.assertEqual(L.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class VeredictoTests(unittest.TestCase):
    def test_especifica(self):
        res = correr('especifica')
        p = res['primario']
        self.assertEqual(p['veredicto'], 'especifica')
        self.assertAlmostEqual(p['redes']['segformer_b0']['Delta_T3_menos_T1'], 0.15, delta=0.03)
        for c in PROTOCOLO['redes']['piloto']:
            self.assertGreater(p['contrastes_paper_guia'][c]['IC95_C'][0], 0)

    def test_compartida(self):
        p = correr('compartida')['primario']
        self.assertEqual(p['veredicto'], 'compartida')
        self.assertTrue(p['L_existe'])

    def test_nula(self):
        self.assertEqual(correr('nula')['primario']['veredicto'], 'no_confirmada')

    def test_ic_contiene_punto_y_es_reproducible(self):
        a, b = correr('compartida', 300), correr('compartida', 300)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        r = a['primario']['redes']['unet']
        self.assertLess(r['IC95_Delta'][0], r['Delta_T3_menos_T1'])
        self.assertGreater(r['IC95_Delta'][1], r['Delta_T3_menos_T1'])
        self.assertIn('SIMULADO', L.resumen_markdown(a))


class EntradasTests(unittest.TestCase):
    def test_faltante_detiene(self):
        corridas = L.simular(EDA, 'nula')
        del corridas[('unet', 37)]
        with self.assertRaises(ValueError):
            L.ejecutar(corridas, EDA, PROTOCOLO, 'x', 'x')

    def test_grupo_incompleto_detiene(self):
        corridas = L.simular(EDA, 'nula')
        corridas[('segformer_b0', 11)] = corridas[('segformer_b0', 11)][:-5]
        with self.assertRaises(ValueError):
            L.ejecutar(corridas, EDA, PROTOCOLO, 'x', 'x')

    def test_lectura_desde_carpeta(self):
        corridas = L.simular(EDA, 'nula')
        with tempfile.TemporaryDirectory() as tmp:
            for (m, s), filas in corridas.items():
                Path(tmp, f'{m}_seed{s}_val_por_imagen.json').write_text(
                    json.dumps({'model': m, 'seed': s, 'split': 'val', 'detail': filas}), encoding='utf-8')
            leidas = L.cargar_por_imagen(tmp)
        self.assertEqual(len(leidas), 50)


class ExtractorTests(unittest.TestCase):
    def test_extraer_desde_checkpoint_simulado(self):
        import torch
        import extraer_validacion_por_imagen as X
        detalle = [{'id': 'a', 'group_id': 'g1', 'Dice': 0.8, 'IoU': 0.7, 'mIoU': 0.8, 'accuracy': 0.9},
                   {'id': 'b', 'group_id': 'g1', 'Dice': 0.6, 'IoU': 0.5, 'mIoU': 0.7, 'accuracy': 0.9},
                   {'id': 'c', 'group_id': 'g2', 'Dice': 0.9, 'IoU': 0.8, 'mIoU': 0.9, 'accuracy': 0.9}]
        with tempfile.TemporaryDirectory() as tmp:
            corrida = Path(tmp, 'unet_seed11')
            corrida.mkdir()
            torch.save({'model': {}, 'epoch': 4, 'fingerprint': 'f', 'validation': detalle}, corrida / 'best_epoch_005.pt')
            sha = hashlib.sha256((corrida / 'best_epoch_005.pt').read_bytes()).hexdigest()
            resultado = {'completed': True, 'model': 'unet', 'seed': 11, 'best_file': 'best_epoch_005.pt',
                         'best_sha256': sha, 'best_val_group_Dice': (0.7 + 0.9) / 2}
            (corrida / 'result.json').write_text(json.dumps(resultado), encoding='utf-8')
            datos = X.extraer(corrida)
            self.assertEqual((datos['epoch'], len(datos['detail'])), (5, 3))
            resultado['best_val_group_Dice'] = 0.5
            (corrida / 'result.json').write_text(json.dumps(resultado), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                X.extraer(corrida)


if __name__ == '__main__':
    unittest.main()
