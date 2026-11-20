"""El cuaderno de la réplica compila, está al día con su generador y corre con datos simulados.

Se ejecutan las celdas 1, 3, 6 y 7 con el ADE20K y el modelo simulados de
test_replica_segformer. Las celdas 2 (entorno de Colab), 4 (descarga de ADE20K),
5 (checkpoint de Hugging Face) y 8 (Drive) solo se compilan: no se ejecutan aquí.
"""
import ast
import json
import pathlib
import sys
import tempfile
import unittest

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(PROJECT / 'tests'))
CUADERNO = PROJECT / 'colab/Replica_SegFormer_ADE20K.ipynb'


class CuadernoTests(unittest.TestCase):
    def test_al_dia_con_el_generador(self):
        import crear_colab_replica as G
        esperado = {'cells': G.celdas()}
        real = json.loads(CUADERNO.read_text(encoding='utf-8'))
        self.assertEqual(real['cells'], esperado['cells'], 'Regenerar con scripts/crear_colab_replica.py')

    def test_celdas_simuladas(self):
        import torch
        import test_replica_segformer as T
        celdas = [''.join(c['source']) for c in json.loads(CUADERNO.read_text(encoding='utf-8'))['cells']
                  if c['cell_type'] == 'code']
        self.assertEqual(len(celdas), 8)
        for c in celdas:
            ast.parse(c)
        caso = T.EvaluacionSimuladaTests()
        caso.setUp()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                g = {}
                exec(celdas[0].replace("RAIZ = '/content/replica'", f'RAIZ = r"{tmp}"'), g)
                exec(celdas[2], g)
                R = g['R']
                g.update(PARES=R.pares_validacion(caso.raiz), MODELO_FN=caso.modelo_lector,
                         INFO={'modelo': 'simulado', 'id2label': {str(k): f'c{k}' for k in range(150)}},
                         DISPOSITIVO=torch.device('cpu'), REGISTRO_DATOS={'simulado': True})
                exec(celdas[5].replace("pathlib.Path('/content/replica_segformer')",
                                       f'pathlib.Path(r"{tmp}") / "salida"'), g)
                exec(celdas[6], g)
                resumen = json.loads((pathlib.Path(tmp) / 'salida/resumen_replica.json').read_text(encoding='utf-8'))
                self.assertIn(resumen['veredicto'], {'replicado', 'compatible', 'no_replicado'})
                self.assertEqual(resumen['primaria']['imagenes'], 3)
                self.assertIn('diferencia_con_primaria', resumen['D1_preprocesamiento_hf_512'])
        finally:
            caso.tearDown()


if __name__ == '__main__':
    unittest.main()
