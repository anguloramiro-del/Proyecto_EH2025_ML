"""Genera el EDA muestral y el dataset analítico agregado de la EH2025."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport


def preparar_grupos(carpeta, salida):
    from .analisis import preparar
    _, a, _ = preparar(carpeta)
    from .analisis import resumen
    a = a.assign(Area=a.area.map({1:'Urbana',2:'Rural'}), Sexo=a.s01a_02.map({1:'Hombre',2:'Mujer'}))
    registros = [dict(Dominio=k[0], Area=k[1], Sexo=k[2], **resumen(g))
                 for k,g in a.groupby(['region','Area','Sexo'], observed=True)]
    salida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(salida/'dataset_procesado.csv', index=False)

def generar(carpeta, salida):
    if carpeta is not None:
        preparar_grupos(carpeta, salida)
    perfil = pd.read_csv(salida/'dataset_procesado.csv')
    assert len(perfil) == 32 and perfil['n'].sum() == 19369
    informe = ProfileReport(perfil, title='EH2025 · Caso 2 · Indicadores de 32 grupos laborales',
        minimal=True, dataset={'description':'INE Bolivia, EH2025, acceso 15/11/2026. Cada fila es un grupo por dominio, área y sexo. Los porcentajes y los ingresos del grupo son ponderados. El EDA describe los 32 grupos con igual peso.'},
        samples={'head':0,'tail':0,'random':0}, duplicates={'head':0}, progress_bar=False)
    informe.to_file(salida/'eda_report.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('carpeta', type=Path, nargs='?', help='Carpeta con los módulos originales para recalcular los grupos.')
    parser.add_argument('--salida', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    generar(args.carpeta, args.salida)
