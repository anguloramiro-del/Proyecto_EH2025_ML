"""Comprueba los módulos EH2025 disponibles en una carpeta local."""
from pathlib import Path
import argparse

CATALOGO = "https://anda.ine.gob.bo/index.php/catalog/256"
MODULOS = ['Persona', 'Discriminacion']

def conectar(carpeta):
    carpeta = Path(carpeta).expanduser()
    rutas = [carpeta / f"EH2025_{nombre}.sav" for nombre in MODULOS]
    faltan = [ruta.name for ruta in rutas if not ruta.is_file()]
    if faltan:
        raise FileNotFoundError(f"Faltan {', '.join(faltan)}. Fuente: {CATALOGO}")
    return rutas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carpeta", type=Path)
    for ruta in conectar(parser.parse_args().carpeta):
        print(ruta.name)
