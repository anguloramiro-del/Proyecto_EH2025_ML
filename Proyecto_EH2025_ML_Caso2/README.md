# Caso 2 · Pobreza multidimensional y empleo en Bolivia

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anguloramiro-del/Proyecto_EH2025_ML/blob/main/Proyecto_EH2025_ML_Caso2/notebooks/Caso2_Pobreza_Multidimensional_Empleo.ipynb)
![Python](https://img.shields.io/badge/Python-3.12-306998)

Abra docs/Caso2_Pobreza_y_Empleo.html en un navegador, incluso sin conexión. El menú del mapa permite consultar ocho dominios. El cuaderno notebooks/Caso2_Pobreza_Multidimensional_Empleo.ipynb contiene el informe y el código de reproducción.

## Repetir el ajuste

Instale requirements.txt en Python 3.12. En el cuaderno indique CARPETA_DATOS con los módulos completos Persona y Vivienda de la EH2025, en formato SAV o Parquet con códigos originales. Ejecute las celdas. Los resultados agregados se escriben en outputs_caso2/. El HTML incluido documenta la ejecución original y no se actualiza automáticamente al cambiar los datos.

Los CSV incluyen métricas en prueba, importancia por permutación, tasas por ámbito y centroides de los perfiles. En resumen.json las matrices de confusión se leen por fila observada y columna predicha (0 = cotiza, 1 = sin aporte). Los microdatos no se distribuyen. Fuente: INE, EH2025, https://anda.ine.gob.bo/index.php/catalog/256, descarga documentada el 15/11/2026.

## Parte 2 · Visualizaciones interactivas

Abra docs/Parte2_Visualizaciones.html en el navegador. El archivo incorpora Plotly.js y funciona sin conexión. Contiene un mapa con tres capas, barras agrupadas, una dispersión territorial multivariada, la proyección de los hogares por K-Means y una matriz de carencias por perfil. El cuaderno también permite ejecutar las cinco figuras.

Para reconstruir el panel desde los archivos agregados de outputs/ y los contornos de data/, ejecute `python -m src.visualizaciones` desde la raíz. Los patrones de hogares se regeneran en la etapa de agrupamiento con acceso a los microdatos. La proyección utiliza PCA centrada con pesos de expansión sobre las cuatro carencias binarias, en las mismas unidades del ajuste K-Means. Los puntos idénticos se agrupan por combinación. Los seis grupos se contrastaron contra la tabla de perfiles anterior.

## Instalación y reproducción

Desde la carpeta de este caso, utilice Python 3.12 e instale las dependencias. Las tablas y los informes publicados se pueden explorar directamente después de descargar el repositorio. Google Colab incluye el entorno Jupyter e IPython.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data/conectar.py /ruta/a/modulos
python -m src.eda /ruta/a/modulos
```

El EDA público describe 32 grupos por dominio, área y sexo que reúnen a 19.369 personas ocupadas. Incluye porcentajes ponderados, ingreso medio y mediano, y tamaño de cada grupo. Para explorar los agregados sin microdatos, ejecute `python -m src.eda` desde la carpeta del caso.
