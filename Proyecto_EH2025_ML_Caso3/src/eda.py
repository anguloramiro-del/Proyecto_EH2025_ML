"""Perfila los nueve departamentos a partir de las estimaciones publicadas."""
from pathlib import Path
import argparse
import pandas as pd
from ydata_profiling import ProfileReport


def generar(tablas, salida):
    tasas = pd.read_csv(tablas/'tasas.csv')
    departamentos = tasas.loc[tasas.dimension.eq('Departamento')]
    datos = departamentos.pivot(index='grupo', columns='indicador',
                                 values=['n','positivos','excluidos','tasa','inferior','superior'])
    datos.columns = [f'{medida}_{indicador}' for medida, indicador in datos.columns]
    grupos = pd.read_csv(tablas/'regiones.csv').set_index('grupo')['perfil']
    datos = datos.join(grupos, validate='one_to_one').rename_axis('Departamento').reset_index()
    assert len(datos) == 9 and datos.perfil.notna().all()
    salida.mkdir(parents=True, exist_ok=True)
    datos.to_csv(salida/'dataset_procesado.csv', index=False)
    variables = ['Departamento','perfil'] + [c for c in datos if c.startswith(('n_', 'tasa_'))]
    reporte = ProfileReport(datos[variables], title='EH2025 · Caso 3 · Nueve departamentos',
        minimal=True, dataset={'description':'INE Bolivia, EH2025, acceso 15/11/2026. Unidad: departamento. Las tasas de origen usan ponderación. El EDA resume nueve departamentos con igual peso.'},
        progress_bar=False, samples={'head':0,'tail':0,'random':0}, duplicates={'head':0})
    reporte.to_file(salida/'eda_report.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tablas', type=Path, default=Path('outputs'))
    parser.add_argument('--salida', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    generar(args.tablas, args.salida)
