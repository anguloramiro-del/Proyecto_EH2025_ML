"""Verifica la Parte 1 del Caso 1. Ejecutar en una copia preparada del proyecto."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import duckdb
import pandas as pd
import pyarrow as pa
import pyreadstat

PERSONA = ['folio', 'nro', 'depto', 'area', 's01a_02', 's01a_03', 'cmasi',
           'upm', 'estrato', 'factor', 'yhogpc', 's03a_04', 's03b_10', 's03b_11',
           's03b_12', 'niv_ed']
VIVIENDA = ['folio', 'totper', 's06a_07', 's06a_09', 's06a_10', 's06a_12']

def huella(path):
    with path.open('rb') as f:
        h = sha256()
        for bloque in iter(lambda: f.read(1024*1024), b''):
            h.update(bloque)
    return h.hexdigest()

def verificar(root):
    manifest = json.loads((root/'data/source_manifest.json').read_text())
    privados = root/'outputs/private'
    fuentes = {}
    with TemporaryDirectory(prefix='eh2025_parte1_') as tmp, ExitStack() as stack:
        con = stack.enter_context(duckdb.connect())
        con.execute("SET memory_limit='512MB'")
        con.execute("SET threads=2")
        con.execute("SET temp_directory=?", [tmp])
        for tabla, columnas in [('Persona', PERSONA), ('Vivienda', VIVIENDA)]:
            modulo = 'EH2025_' + tabla
            original = root / manifest['raw_module_files'][modulo]
            meta = json.loads((privados/f'{modulo}_source.json').read_text())
            digest = huella(original)
            if digest != meta['source_sha256']:
                raise ValueError(f'{modulo}: cambió la huella de la fuente')
            stata = original
            if original.suffix.lower()=='.sav':
                # Leer únicamente las columnas analíticas de un módulo a la vez.
                df, _ = pyreadstat.read_sav(str(original), usecols=columnas,
                    apply_value_formats=False, user_missing=False, disable_datetime_conversion=True)
                stata = Path(tmp)/f'{modulo}.dta'
                df[columnas].to_stata(stata, write_index=False, version=118)
                del df
            reader = stack.enter_context(pd.read_stata(stata, columns=columnas, chunksize=5000,
                iterator=True, convert_categoricals=False, convert_dates=False))
            # Cada escaneo consume el archivo DTA una sola vez. Los RecordBatch
            # se obtienen por demanda durante la ejecución de la consulta DuckDB.
            primero = pa.RecordBatch.from_pandas(next(reader), preserve_index=False)
            def lotes(inicial, lector):
                yield inicial
                for bloque in lector:
                    yield pa.RecordBatch.from_pandas(bloque, schema=inicial.schema, preserve_index=False)
            lotes_arrow = pa.RecordBatchReader.from_batches(primero.schema, lotes(primero, reader))
            con.register(tabla,lotes_arrow)
            fuentes[tabla] = dict(original=original.name, sha256_original=digest, filas_originales=meta['rows'],
                entrada_cruce=stata.name, dta_derivado=(stata!=original), columnas=len(columnas))
        proyeccion = ', '.join(['p.'+c for c in PERSONA]+['h.'+c+' AS hh_'+c for c in VIVIENDA[1:]])
        con.execute(f'CREATE TEMP TABLE cruce AS SELECT {proyeccion} FROM Persona p LEFT JOIN Vivienda h ON p.folio=h.folio')
        con.read_parquet(str(privados/'joined.parquet')).create_view('referencia')
        def escalar(sql):
            return con.execute(sql).fetchone()[0]
        n = escalar('SELECT count(*) FROM cruce')
        dif = [escalar(f'SELECT count(*) FROM (SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b})')
               for a,b in [('cruce','referencia'),('referencia','cruce')]]
        base = json.loads((root/'outputs/join_audit.json').read_text())
        assert n == fuentes['Persona']['filas_originales'] == base['joined_rows'] and dif == [0,0]
        assert fuentes['Vivienda']['filas_originales'] == base['housing_rows']
        assert base['unmatched_person_rows'] == base['duplicate_household_key_groups'] == 0
        peso = escalar('SELECT sum(factor) FROM cruce')
        assert abs(peso-base['source_weight_sum']) < 1e-5
        eda = root/'outputs/eda_report.html'
        html = eda.read_text()
        assert all(t in html for t in ['EDA EH2025 Caso 1','inasistencia','4731','data-bs-toggle'])
        return dict(catalogo=manifest['catalog_url'],fecha_descarga_declarada=manifest['microdata_download_date'],
            verificacion_utc=datetime.now(timezone.utc).isoformat(),fuentes=fuentes,duckdb=duckdb.__version__,
            filas_cruzadas=n,columnas_cruzadas=len(con.table('cruce').columns),
            diferencias_dta_menos_referencia=dif[0],diferencias_referencia_menos_dta=dif[1],
            suma_factor=peso,clave='LEFT JOIN Persona.folio = Vivienda.folio',
            lectura_dta_lote=5000,limite_duckdb='512MB',hilos=2,
            eda=dict(archivo='outputs/eda_report.html',sha256=huella(eda),bytes=eda.stat().st_size))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--proyecto',type=Path,default=Path.cwd())
    parser.add_argument('--salida',type=Path)
    args=parser.parse_args()
    texto=json.dumps(verificar(args.proyecto.resolve()),ensure_ascii=False,indent=2)
    if args.salida:
        args.salida.write_text(texto+'\n')
    print(texto)
