"""Figuras y lectura de la evaluación a partir de tablas agregadas."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NOMBRES = {'Percepcion_Discriminacion': 'Discriminación declarada', 'Victimizacion': 'Victimización'}
COLORES = {'Regresión logística': '#6b5194', 'Random Forest': '#06786c', 'XGBoost': '#b65322'}


def decimal(valor):
    return f'{valor:.3f}'.replace('.', ',')


def resumen_modelos(salida):
    salida = Path(salida)
    metricas = pd.read_csv(salida/'modelos.csv')
    imp = pd.read_csv(salida/'importancia.csv')
    sel = metricas.loc[metricas.seleccionado_cv].set_index('objetivo')
    texto = ['<h2>Parte 3 · Modelos y explicabilidad</h2>',
             '<p>Se compararon Random Forest y XGBoost con una regresión logística de referencia. La unidad de análisis es la persona de 15 años o más seleccionada mediante Kish. La clase positiva indica que declaró discriminación por al menos un motivo o haber sufrido al menos un delito en los últimos doce meses, según el desenlace. Cada modelo se entrenó y evaluó por separado para cada desenlace.</p>',
             '<p>Se reservó uno de cinco pliegues estratificados para prueba. En el resto se usaron cuatro pliegues estratificados para comparar modelos. Las unidades primarias de muestreo (UPM) se mantuvieron completas en cada partición. Se eligió el candidato con mayor precisión promedio (AP) media en validación. Para cada candidato se eligió el umbral que maximizó F1 en las predicciones fuera de pliegue del entrenamiento. El modelo final se ajustó con todo el entrenamiento. Las métricas usan el ponderador de la encuesta. Las matrices muestran personas observadas.</p>',
             '<p>AP resume la curva precisión–sensibilidad. Su referencia es la prevalencia ponderada en la prueba. La precisión es la proporción de positivos predichos que declaró el hecho y la sensibilidad es la fracción de casos declarados detectada. F1 combina ambas al umbral elegido. El AUC-ROC mide la capacidad para ordenar los casos positivos por encima de los negativos.</p>']
    encabezados = ['Objetivo', 'Modelo', 'AP validación', 'AUC-ROC prueba', 'AP prueba', 'Umbral', 'Precisión', 'Sensibilidad', 'F1']
    tabla = []
    for fila in metricas.itertuples():
        tabla.append([NOMBRES[fila.objetivo], fila.modelo] + [decimal(x) for x in (
            fila.AP_cv, fila.AUC, fila.AP, fila.umbral, fila.precision, fila.sensibilidad, fila.F1)])
    texto.append(pd.DataFrame(tabla, columns=encabezados).to_html(index=False, border=0))
    for obj, nombre in NOMBRES.items():
        m = sel.loc[obj]
        ranking = imp[(imp.objetivo == obj) & (imp.modelo == m.modelo)].nlargest(2, 'caida_AP')
        primero, segundo = list(ranking.itertuples())
        texto.append(f'<h3>{nombre}</h3><p>{m.modelo} obtuvo la AP media más alta en validación '
                     f'({decimal(m.AP_cv)}, desviación estándar entre pliegues {decimal(m.DE_AP_cv)}). '
                     f'La prueba independiente por UPM reunió {m.n_prueba:,} personas. La AP fue {decimal(m.AP)} '
                     f'frente a una prevalencia de {decimal(m.AP_referencia)}. Al umbral {decimal(m.umbral)}, '
                     f'identificó {m.tp} de {m.tp+m.fn} casos observados y produjo {m.fp} falsos positivos.</p>'
                     .replace(f'{m.n_prueba:,}', f'{m.n_prueba:,}'.replace(',', '.')))
        texto.append(f'<p>Al permutar una variable en la prueba, la mayor pérdida media de AP correspondió a '
                     f'{primero.variable.lower()} ({decimal(primero.caida_AP)}), seguida de '
                     f'{segundo.variable.lower()} ({decimal(segundo.caida_AP)}). '
                     'Las diez repeticiones estiman cuánto depende el modelo de la información de cada variable. '
                     'Las variables correlacionadas pueden compartir importancia.</p>')
    texto.append('<p>En la EH2025, la discriminación declarada alcanza 25,05 % en La Paz y 15,50 % en el país. '
                 'La victimización declarada alcanza 14,82 % en el área urbana y 10,09 % en el área rural. '
                 'Estas diferencias agregadas contextualizan el aporte de la geografía. La encuesta transversal '
                 'mide experiencias declaradas y el análisis no permite atribuir las diferencias a causas específicas. '
                 'La baja precisión, especialmente para victimización, limita su utilidad para decisiones sobre personas. '
                 'El análisis admite comparaciones territoriales y formulación de preguntas para investigación posterior.</p>')
    return '\n'.join(texto)


def figuras_modelos(salida):
    salida = Path(salida)
    curvas = pd.read_csv(salida/'curvas_modelos.csv')
    metricas = pd.read_csv(salida/'modelos.csv')
    cortes = pd.read_csv(salida/'umbrales_matrices.csv')
    imp = pd.read_csv(salida/'importancia.csv')
    curva = make_subplots(rows=2, cols=2, horizontal_spacing=.12, vertical_spacing=.19,
                          subplot_titles=('Discriminación · ROC', 'Discriminación · precisión–sensibilidad',
                                          'Victimización · ROC', 'Victimización · precisión–sensibilidad'))
    for fila, obj in enumerate(NOMBRES, 1):
        sub = curvas[curvas.objetivo == obj]
        base = metricas.loc[metricas.objetivo == obj, 'AP_referencia'].iloc[0]
        for col, tipo in enumerate(['ROC', 'PR'], 1):
            for mod, color in COLORES.items():
                datos = sub[(sub.modelo == mod) & (sub.curva == tipo)]
                curva.add_trace(go.Scatter(x=datos.x.tolist(), y=datos.y.tolist(), mode='lines', name=mod,
                    legendgroup=mod, showlegend=fila == 1 and col == 1,
                    line=dict(color=color, width=1.8), hovertemplate='x = %{x:.3f}<br>y = %{y:.3f}<extra>%{fullData.name}</extra>'),
                    row=fila, col=col)
            curva.add_shape(type='line', x0=0, x1=1, y0=0 if col == 1 else base, y1=1 if col == 1 else base,
                            line=dict(color='#555', width=1.3, dash='dot'), row=fila, col=col)
            curva.update_xaxes(title_text='Tasa de falsos positivos' if col == 1 else 'Sensibilidad',
                               range=[0,1], row=fila, col=col)
            curva.update_yaxes(title_text='Sensibilidad' if col == 1 else 'Precisión', range=[0,1.02], row=fila, col=col)
    curva.update_layout(height=690, margin=dict(t=55,b=75), legend=dict(orientation='h',y=-.15))

    matriz = go.Figure()
    opciones = []
    for i, r in enumerate(cortes.itertuples()):
        etiqueta = f'{NOMBRES[r.objetivo]} · {r.modelo} · {r.regla}'
        matriz.add_trace(go.Heatmap(z=[[r.tn, r.fp], [r.fn, r.tp]], text=[[r.tn, r.fp], [r.fn, r.tp]],
            x=['Predice ausencia', 'Predice presencia'], y=['Declaró ausencia', 'Declaró presencia'],
            visible=i == 0, texttemplate='%{text}', textfont_size=22, colorscale='Blues', showscale=False,
            hovertemplate='%{y}<br>%{x}<br>%{z} personas<extra></extra>'))
        opciones.append(dict(label=etiqueta, method='update',
                             args=[{'visible':[j == i for j in range(len(cortes))]},
                                   {'title.text':f'Umbral {decimal(r.umbral)} · F1 ponderado {decimal(r.F1)}'}]))
    matriz.update_layout(title_text=f'Umbral {decimal(cortes.iloc[0].umbral)} · F1 ponderado {decimal(cortes.iloc[0].F1)}',
        updatemenus=[dict(buttons=opciones,x=0,y=1.25,xanchor='left',yanchor='bottom')],
        height=470, margin=dict(t=150,b=50), yaxis=dict(autorange='reversed',automargin=True))

    importancia = make_subplots(rows=2, cols=1, vertical_spacing=.22,
        subplot_titles=tuple(f'{NOMBRES[m.objetivo]} · {m.modelo}' for m in metricas[metricas.seleccionado_cv].itertuples()))
    for fila, r in enumerate(metricas[metricas.seleccionado_cv].itertuples(), 1):
        g = imp[(imp.objetivo == r.objetivo) & (imp.modelo == r.modelo)].sort_values('caida_AP')
        importancia.add_trace(go.Bar(x=g.caida_AP.tolist(), y=g.variable.tolist(), orientation='h',
            error_x=dict(type='data',array=g.DE.tolist(),color='#575757'), marker_color=COLORES[r.modelo],
            hovertemplate='%{y}<br>Caída de AP: %{x:.3f}<extra></extra>'), row=fila, col=1)
        importancia.update_xaxes(title_text='Pérdida de precisión promedio (AP)', row=fila, col=1)
    importancia.update_layout(height=700, showlegend=False, yaxis=dict(automargin=True), yaxis2=dict(automargin=True))
    figuras = [
        ('Curvas ROC y precisión–sensibilidad', 'La diagonal marca azar en ROC. En precisión–sensibilidad, la horizontal marca la prevalencia ponderada de la prueba. Pulse la leyenda para mostrar u ocultar un modelo.', curva),
        ('Matrices de confusión', 'El menú permite elegir el objetivo, el modelo y el umbral. Las filas son declaraciones observadas y las columnas son predicciones. Cada celda cuenta personas sin ponderar.', matriz),
        ('Importancia por permutación', 'Para el modelo seleccionado en cada objetivo se muestra la caída de AP ponderada después de permutar cada predictor. Las barras de error indican una desviación estándar entre diez permutaciones. Un valor cercano a cero indica escasa contribución medible en la prueba.', importancia)
    ]
    for _, _, fig in figuras:
        fig.update_layout(template='plotly_white', font=dict(family='Arial',size=13,color='#183d38'))
    return figuras
