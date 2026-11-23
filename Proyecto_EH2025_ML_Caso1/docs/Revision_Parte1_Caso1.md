# Caso 1 · Verificación de la Parte 1

## Dictamen

La entrega revisada contiene los módulos documentados, el cruce muchos-a-uno con DuckDB y el EDA interactivo. La ejecución original utilizó archivos SPSS. La verificación complementaria reproduce el cruce leyendo dos copias de trabajo Stata y obtiene exactamente las mismas filas y valores en las 21 columnas del cruce.

| Requisito | Evidencia | Resultado |
|---|---|---|
| Acceso a la EH2025 | Catálogo BOL-INE-EH-2025, manifiesto del proyecto y huellas SHA-256 de Persona y Vivienda | La fecha y procedencia de descarga que declaró el usuario son 15/11/2026 y ANDA-INE. La identidad del estudio y las huellas de los originales se verificaron. Se reutilizaron los módulos recibidos. |
| Cruce en DuckDB con insumos Stata | src/checks/verificar_parte1_dta.py y outputs/verificacion_parte1_dta.json | El script genera temporalmente dos archivos DTA con las columnas de trabajo de las fuentes SAV. Lee los DTA en lotes de 5.000 filas y entrega esos lotes a DuckDB mediante Arrow. El SQL hace un LEFT JOIN por folio con dos hilos y límite de memoria de DuckDB de 512 MB. |
| EDA automatizado en outputs/ | outputs/eda_report.html y src/workflow.py en el proyecto principal | El HTML entregado identifica YData-Profiling 4.18.4 y presenta 4.731 observaciones, 11 variables y 1.275 celdas faltantes (2,4 %). Los controles interactivos y la suma de verificación del archivo están presentes. |

## Resultado del cruce

Persona aporta 39.485 filas y Vivienda aporta 12.726. El resultado conserva las 39.485 personas. La auditoría original controla claves vacías, duplicadas y hogares sin correspondencia. Dos consultas EXCEPT ALL, una en cada dirección, devolvieron cero diferencias entre el cruce desde Stata y la referencia. Esta comparación conserva las multiplicidades y cubre las 21 columnas. Los factores de expansión suman aproximadamente 11.980.605 antes y después. Este total corresponde al módulo completo. El EDA posterior describe la muestra de adolescentes de 12 a 17 años sin ponderar.

Dask es opcional en la consigna. El cruce terminó correctamente con DuckDB y lectura por lotes. Los 512 MB son un ajuste del motor SQL; la verificación no midió el consumo máximo de todo el proceso Python.

## Cómo revisar o reproducir

1. Abrir outputs/eda_report.html en un navegador para explorar las variables.
2. Para repetir el proceso privado, integrar el complemento en la raíz del proyecto original, instalar sus dependencias y colocar los módulos autorizados en las rutas registradas en data/source_manifest.json.
3. Ejecutar los comandos de abajo. La primera etapa genera los Parquet privados y la auditoría base. La segunda regenera el EDA. La tercera repite el cruce desde Stata y contrasta sus resultados.
4. En Colab, el cuaderno 03_Caso1_Completo contiene las etapas prepare y profile. El cuaderno 04 es una continuación de resultados. El script complementario se ejecuta desde la raíz del proyecto preparado.

```bash
python src/run.py prepare
python src/run.py profile
python src/checks/verificar_parte1_dta.py --salida outputs/verificacion_parte1_dta.json
```

El complemento contiene la verificación y las evidencias agregadas. Los DTA de trabajo se eliminan al concluir la ejecución y pueden regenerarse desde los módulos autorizados. El repositorio publicado anteriormente conserva el flujo original. Si el docente exige originales distribuidos por ANDA directamente en formato Stata, la evidencia disponible acredita la descarga declarada en SAV y el uso posterior de DTA derivados.

Fuente: Instituto Nacional de Estadística, Encuesta de Hogares 2025, https://anda.ine.gob.bo/index.php/catalog/256. Fecha de acceso registrada en el proyecto: 15/11/2026.
