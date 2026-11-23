# Discriminación y seguridad ciudadana en Bolivia · EH2025

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anguloramiro-del/Proyecto_EH2025_ML/blob/main/Proyecto_EH2025_ML_Caso3/notebooks/Caso3_Discriminacion_Seguridad.ipynb)
![Python](https://img.shields.io/badge/Python-3.12-306998)

Abra `docs/Informe_Caso3.html` para ver el informe completo sin conexión. El informe contiene cuatro gráficos exploratorios y tres figuras de evaluación. El cuaderno de Colab en `notebooks/` incorpora las mismas cifras y el código para recalcularlas.

## Parte 3 · Modelos supervisados

Se entrenaron Random Forest, XGBoost y una regresión logística por separado para discriminación declarada y victimización. La elección de candidato usa la precisión promedio ponderada (AP) en cuatro pliegues estratificados por clase y agrupados por unidad primaria de muestreo. Un pliegue externo independiente se reservó para la evaluación final. Las celdas de las matrices son conteos sin ponderar. Las métricas usan el ponderador de la encuesta.

| Archivo en outputs/ | Contenido |
| --- | --- |
| modelos.csv | AP y AUC medias y desviaciones estándar de validación, métricas de prueba, umbral de F1, tamaño de muestra y candidato elegido |
| validacion_pliegues.csv | AP, AUC, prevalencia, número de participantes, positivos y UPM por pliegue |
| umbrales_matrices.csv | Métricas y conteos tn, fp, fn, tp al umbral elegido y al corte de 0,5 |
| curvas_modelos.csv | Coordenadas de curvas ROC y precisión–sensibilidad ponderadas |
| importancia.csv | Pérdida media de AP y desviación estándar tras diez permutaciones por predictor y modelo |

Para reproducir con Python 3.12 instale las dependencias de `requirements.txt` y ejecute el cuaderno indicando la carpeta con los dos SAV originales. Se requieren EH2025_Discriminacion.sav y EH2025_Persona.sav. La variable Percepcion_Discriminacion registra experiencia personal declarada. Victimizacion registra exposición como víctima. El informe contiene los vínculos al cuestionario y los tabulados del INE. Las salidas distribuidas contienen resultados agregados.

## Instalación y reproducción

Desde la carpeta de este caso, utilice Python 3.12 e instale las dependencias. Las tablas y los informes publicados se pueden explorar directamente después de descargar el repositorio. Google Colab incluye el entorno Jupyter e IPython.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data/conectar.py /ruta/a/modulos
python -m src.eda
```

La última orden funciona con las tablas publicadas. Reconstruye el EDA y el dataset de nueve departamentos, con tamaños de muestra, positivos, exclusiones, tasas porcentuales ponderadas, intervalos aproximados del 95 % y grupo territorial. El EDA da igual peso a cada departamento. Para recalcular las estimaciones individuales y reentrenar, ejecute el cuaderno con los dos módulos SAV.

Las curvas se publican en una malla de 51 puntos por modelo, objetivo y tipo de curva. Las métricas se calcularon con la resolución original.
