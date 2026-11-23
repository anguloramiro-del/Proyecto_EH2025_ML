# Bolivia · Encuesta de Hogares 2025 y machine learning

![Python 3.12](https://img.shields.io/badge/Python-3.12-306998)
![Fuente INE Bolivia](https://img.shields.io/badge/Fuente-INE%20Bolivia-19674a)
![Casos de estudio](https://img.shields.io/badge/Casos-3-67478e)

Tres preguntas sobre educación, empleo y seguridad ciudadana en Bolivia. Cada carpeta reúne el código reproducible, el cuaderno, las salidas agregadas, el reporte exploratorio y el informe final de su caso.

| Caso | Pregunta y alcance | Entregables principales |
| --- | --- | --- |
| [1 · Asistencia escolar](Proyecto_EH2025_ML_Caso1/) | ¿Qué características se asocian con la no asistencia de adolescentes de 12 a 17 años? La EH2025 mide la asistencia en un momento dado. El Censo 2024 aporta el componente municipal. | [Informe PDF](Proyecto_EH2025_ML_Caso1/docs/Informe_Caso1_EH2025_Final.pdf) · [Cuaderno](Proyecto_EH2025_ML_Caso1/notebooks/03_Caso1_Completo.ipynb) · [EDA](Proyecto_EH2025_ML_Caso1/outputs/eda_report.html) |
| [2 · Pobreza y empleo](Proyecto_EH2025_ML_Caso2/) | ¿Cómo varían la cobertura contributiva, los ingresos y la jornada laboral con las carencias del hogar? | [Informe HTML](Proyecto_EH2025_ML_Caso2/docs/Caso2_Pobreza_y_Empleo.html) · [Panel de cinco gráficos](Proyecto_EH2025_ML_Caso2/docs/Parte2_Visualizaciones.html) · [Cuaderno](Proyecto_EH2025_ML_Caso2/notebooks/Caso2_Pobreza_Multidimensional_Empleo.ipynb) · [EDA](Proyecto_EH2025_ML_Caso2/outputs/eda_report.html) |
| [3 · Discriminación y seguridad](Proyecto_EH2025_ML_Caso3/) | ¿Qué factores se asocian con la discriminación declarada y con haber sufrido un delito? | [Informe HTML](Proyecto_EH2025_ML_Caso3/docs/Informe_Caso3.html) · [Cuaderno](Proyecto_EH2025_ML_Caso3/notebooks/Caso3_Discriminacion_Seguridad.ipynb) · [EDA](Proyecto_EH2025_ML_Caso3/outputs/eda_report.html) |

## Cómo explorar la entrega

1. Para leer los informes HTML con sus mapas, curvas y filtros, descargue el repositorio con **Code → Download ZIP**, descomprímalo y abra el HTML con su navegador. GitHub permite consultar directamente el PDF y el código.
2. Cada carpeta tiene instrucciones de ejecución, dependencias con versiones y un directorio `outputs/` con los resultados ya calculados. Los cuadernos pueden abrirse desde Google Colab.
3. Para repetir el ajuste con datos individuales, obtenga los módulos de la encuesta desde el [catálogo ANDA del INE](https://anda.ine.gob.bo/index.php/catalog/256) y siga el README del caso. La descarga pública del repositorio contiene datos procesados agregados.

## Organización

| Carpeta de cada caso | Contenido |
| --- | --- |
| `data/` | Procedencia, preparación o conexión de insumos y recursos geográficos |
| `notebooks/` | Cuadernos Jupyter |
| `src/` | Preparación de variables, modelos y visualizaciones |
| `outputs/` | EDA interactivo, datasets agregados y métricas |
| `docs/` | Informe final y materiales de interpretación |

El Caso 1 conserva además sus controles de calidad y pruebas. Los tres estudios usan poblaciones y ponderadores propios. El índice del Caso 2 resume cuatro carencias definidas para el ejercicio. El Caso 3 estudia experiencias declaradas.

Fuente: Instituto Nacional de Estadística, Encuesta de Hogares 2025 - https://anda.ine.gob.bo/index.php/catalog/256. Fecha de acceso documentada: 15 de noviembre de 2026.
