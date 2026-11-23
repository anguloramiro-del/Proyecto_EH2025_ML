from pathlib import Path
import json
from html import escape
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs
import plotly.io as pio

ET={'Inseguridad':'Inseguridad nocturna','Victimizacion':'Victimización','Percepcion_Discriminacion':'Discriminación declarada'}
def figuras(root):
    t=pd.read_csv(root/'outputs/tasas.csv')
    reg=pd.read_csv(root/'outputs/regiones.csv').set_index('grupo')
    perfiles=pd.read_csv(root/'outputs/seleccion_k.csv')
    motivos=pd.read_csv(root/'outputs/motivos.csv').sort_values('porcentaje')
    geo=json.loads((root/'data/contornos_departamentales.json').read_text())
    mapas=go.Figure()
    for i,(c,label) in enumerate(ET.items()):
        for n,rings in geo.items():
            x=[];y=[]
            for ring in rings:
                for lon,lat in ring:
                    x.append(lon);y.append(lat)
                x.append(None);y.append(None)
            row=t[(t.dimension=='Departamento')&(t.grupo==n)&(t.indicador==c)].iloc[0]
            val=float(row.tasa); color=sample_colorscale('Teal',[min(val/60,1)])[0]
            titulo=f'{n} · {val:.1f}%'
            mapas.add_scatter(x=x,y=y,name=titulo,text=f'{titulo}<br>IC 95%: {row.inferior:.1f}–{row.superior:.1f}%<br>n={row.n}',
                mode='lines',line=dict(width=1,color='white'),fill='toself',fillcolor=color,
                hoveron='fills',hoverinfo='name+text',showlegend=False,visible=i==0)
    mapas.add_scatter(x=[None,None],y=[None,None],mode='markers',hoverinfo='skip',showlegend=False,
        marker=dict(color=[0,60],colorscale='Teal',cmin=0,cmax=60,showscale=True,colorbar=dict(title='Tasa (%)',thickness=13)))
    mapas.update_layout(updatemenus=[dict(x=0,y=1.12,xanchor='left',buttons=[
        dict(label=l,method='update',args=[{'visible':[j//9==i for j in range(27)]+[True]}]) for i,l in enumerate(ET.values())])],
        height=510, margin=dict(t=58,l=5,r=10,b=5))
    mapas.update_xaxes(visible=False);mapas.update_yaxes(visible=False,scaleanchor='x',scaleratio=1.04)
    barra=go.Figure()
    dims=['Área','Sexo','Edad','Autoidentificación NPIOC o afroboliviana','Pobreza monetaria','Nivel educativo']
    for j,dim in enumerate(dims):
        for (c,label),color in zip(ET.items(),['#28746b','#ae603b','#4f5b93']):
            s=t[(t.dimension==dim)&(t.indicador==c)]
            barra.add_bar(x=s.grupo.tolist(),y=s.tasa.tolist(),name=label,visible=j==0,marker_color=color,
                  error_y=dict(type='data',symmetric=False,array=(s.superior-s.tasa).tolist(),arrayminus=(s.tasa-s.inferior).tolist()),
                  hovertemplate='%{x} · %{y:.1f}%<extra>%{fullData.name}</extra>')
    barra.update_layout(barmode='group',yaxis_title='Porcentaje ponderado',height=450,
        updatemenus=[dict(x=0,y=1.21,xanchor='left',buttons=[dict(label=dim,method='update',args=[{'visible':[j//3==i for j in range(3*len(dims))]}]) for i,dim in enumerate(dims)])])
    f=go.Figure()
    for k,g in reg.groupby('perfil'):
        f.add_scatter(x=g.Victimizacion.tolist(),y=g.Inseguridad.tolist(),mode='markers',marker=dict(size=15),
            text=g.index.tolist(),name=f'Perfil {k}',hovertemplate='<b>%{text}</b><br>Victimización: %{x:.1f}%<br>Inseguridad: %{y:.1f}%<extra>%{fullData.name}</extra>')
    f.update_layout(xaxis_title='Victimización en los últimos 12 meses (%)',yaxis_title='Inseguridad nocturna (%)')
    m=go.Figure(go.Bar(x=motivos.porcentaje.tolist(),y=motivos.motivo.tolist(),orientation='h',marker_color='#536490',hovertemplate='%{y}: %{x:.2f}%<extra></extra>'))
    m.update_layout(xaxis_title='Porcentaje ponderado entre respuestas válidas al motivo',height=440,yaxis=dict(automargin=True))
    figs=[('Mapa departamental','Seleccione el indicador. Las tres capas comparten escala. Los intervalos y denominadores detallados se incluyen en tasas.csv. Cartografía referencial de GeoBolivia 2015, distribuida por geoBoundaries (CC BY 4.0).',mapas),
          ('Brechas sociodemográficas','Cambie la dimensión en el menú. Las barras muestran tasas ponderadas; los segmentos representan intervalos aproximados del 95 %.',barra),
          ('Perfiles de seguridad','Cada punto representa un departamento. K-Means agrupa las tasas estandarizadas de victimización e inseguridad, dando igual peso a cada departamento.',f),
          ('Motivos de discriminación','Una persona puede declarar varios motivos. Para cada uno, el denominador incluye quienes respondieron sí o no.',m)]
    for _,_,fig in figs: fig.update_layout(template='plotly_white',font=dict(family='Arial',size=13,color='#183d38'))
    return figs

def exportar(root,lectura,metodo,propuestas):
    from .panel_modelos import figuras_modelos, resumen_modelos
    figs=figuras(root)+figuras_modelos(root/"outputs");partes=[]
    for i,(t,nota,f) in enumerate(figs):
        if i == 4:
            partes.append('<section class="evaluacion">'+resumen_modelos(root/'outputs')+'</section>')
        partes.append('<section><h2>'+escape(t)+'</h2><p>'+escape(nota)+'</p>'+pio.to_html(f,include_plotlyjs=False,full_html=False,div_id=f'fig-{i}',config={'responsive':True,'displaylogo':False})+'</section>')
    html='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Discriminación y seguridad ciudadana · Bolivia</title><style>body{background:#eef4f1;color:#203d38;font:16px/1.55 system-ui;margin:0}main{max-width:1000px;background:white;margin:auto;padding:clamp(18px,5vw,55px)}h1{font-size:clamp(34px,5vw,50px);line-height:1.1}h2{border-top:2px solid #aac8bd;padding-top:22px}a{color:#16544b}p{max-width:86ch}li{padding-bottom:8px}.evaluacion{overflow-x:auto}table{border-collapse:collapse;font-size:14px;margin:1rem 0}th,td{padding:.4rem .55rem;text-align:right;border-bottom:1px solid #d5e0dd}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}thead{background:#edf4f1}</style><script>'+get_plotlyjs()+'</script><main><p>Bolivia · Encuesta de Hogares 2025</p><h1>Discriminación y seguridad ciudadana</h1>'+lectura+''.join(partes)+propuestas+metodo+'</main></html>'
    (root/'docs/Informe_Caso3.html').write_text(html)
    return figs
