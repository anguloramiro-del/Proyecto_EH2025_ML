"""Parte 2. Generar el panel desde los resultados agregados del Caso 2."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs

RAIZ = Path(__file__).resolve().parents[1]
DIM = ['educacion','piso','saneamiento','electricidad']
CAR = ['Educación','Piso de tierra','Saneamiento','Electricidad']
COLORES = ['#93502f','#007367','#7052a0','#326885','#a13c68','#545a2e']
PERFILES = ['Saneamiento y, en su mayoría, piso', 'Educación', 'Educación y saneamiento',
            'Piso de tierra', 'Educación y piso', 'Educación, piso y saneamiento']

def mapa(r, contornos):
    f = go.Figure()
    capas = [('informal_pct','Sin aporte (%)','Teal'),('ingreso_mediano','Ingreso mediano (Bs/mes)','Blues'),
             ('mas_48h_pct','Más de 48 h (%)','Oranges')]
    for i,(var,etiqueta,escala) in enumerate(capas):
        bajo,alto = r[var].min(),r[var].max()
        for nombre,poligonos in contornos.items():
            dominio='Beni–Pando' if nombre in ['Beni','Pando'] else nombre
            g=r.loc[dominio]
            x=[];y=[]
            for anillo in poligonos:
                for lon,lat in anillo:
                    x.append(lon); y.append(lat)
                x.append(None); y.append(None)
            color=sample_colorscale(escala,[(g[var]-bajo)/(alto-bajo) * .8 + .16])[0]
            texto=(f'<b>{dominio}</b><br>Sin aporte: {g.informal_pct:.1f}%'
                   f' (IC 95%: {g.ic_inf:.1f}–{g.ic_sup:.1f}%)'
                   f'<br>Ingreso mediano: Bs {g.ingreso_mediano:,.0f}'
                   f'<br>Más de 48 h: {g.mas_48h_pct:.1f}%<br>Muestra: {int(g.n):,} ocupados')
            f.add_scatter(x=x,y=y,fill='toself',mode='lines',line=dict(color='#fcfcfa',width=1),
                          fillcolor=color,hoveron='fills',text=texto,hoverinfo='text',
                          name=nombre,showlegend=False,visible=(i==0))
        stops=[[j/8,sample_colorscale(escala,[.16+.8*j/8])[0]] for j in range(9)]
        f.add_scatter(x=[None,None],y=[None,None],mode='markers',hoverinfo='skip',showlegend=False,
                      marker=dict(color=[bajo,alto],cmin=bajo,cmax=alto,colorscale=stops,showscale=True,
                                  colorbar=dict(title=etiqueta,len=.65,thickness=14)),visible=i==0)
    longitud=len(contornos)+1
    botones=[dict(label=tag,method='update',args=[dict(visible=[k//longitud==i for k in range(len(f.data))])])
             for i,(_,tag,_) in enumerate(capas)]
    f.update_layout(updatemenus=[dict(buttons=botones,x=0,y=1.12,xanchor='left')],
                    margin=dict(t=60,b=10,l=8,r=10),height=510)
    f.update_xaxes(visible=False,constrain='domain'); f.update_yaxes(visible=False,scaleanchor='x',scaleratio=1.04)
    return f

def figuras(root=RAIZ):
    t=pd.read_csv(root/'outputs/tasas.csv').set_index('ambito')
    p=pd.read_csv(root/'outputs/perfiles.csv')
    patrones=pd.read_csv(root/'outputs/patrones_kmeans.csv')
    regiones=t.drop(index=['Nacional','Desde el umbral','Bajo el umbral'])
    carto=json.loads((root/'data/contornos_departamentales.json').read_text())
    panel=[('Mapa de protección e ingreso','Seleccione una capa y sitúe el cursor sobre un departamento. Los nueve contornos representan ocho estimaciones: Beni y Pando comparten los valores del dominio conjunto. Cada capa tiene su propia escala. Cartografía referencial GeoBolivia 2015, distribuida por geoBoundaries (CC BY 4.0).',mapa(regiones,carto))]

    f=go.Figure()
    for nombre,rotulo,color in [('Bajo el umbral','IPM menor a ⅓','#326885'),('Desde el umbral','IPM de ⅓ o mayor','#a34728')]:
        g=t.loc[nombre]
        f.add_bar(name=rotulo,x=['Aporte actual','Jornada de más de 48 h'],y=[g.formal_pct,g.mas_48h_pct],
                  marker_color=color,text=[f'{x:.1f}%' for x in (g.formal_pct,g.mas_48h_pct)],textposition='outside',
                  hovertemplate='%{x}: %{y:.2f}%<extra>%{fullData.name}</extra>')
    f.update_layout(barmode='group',yaxis_title='Ocupados (%)',yaxis_range=[0,48],height=355)
    panel.append(('Protección y jornada según pobreza del hogar','La cobertura contributiva difiere en 15,83 puntos porcentuales entre grupos. En cambio, las jornadas de más de 48 horas alcanzan aproximadamente al 36 % en ambos. Pulse una categoría de la leyenda para aislarla.',f))

    f=go.Figure(go.Scatter(x=regiones.informal_pct.tolist(),y=regiones.ingreso_mediano.tolist(),
        mode='markers',text=regiones.index.tolist(),customdata=regiones[['mas_48h_pct','n','n_ingreso']].values.tolist(),
        marker=dict(size=regiones.n.tolist(),sizemode='area',sizeref=2*regiones.n.max()/43**2,sizemin=7,
                    color=regiones.mas_48h_pct.tolist(),colorscale='Viridis',showscale=True,
                    colorbar=dict(title='Más de 48 h (%)'),line=dict(width=1,color='white')),
        hovertemplate='<b>%{text}</b><br>Sin aporte: %{x:.1f}%<br>Ingreso mediano: Bs %{y:,.0f}<br>Más de 48 h: %{customdata[0]:.1f}%<br>Muestra: %{customdata[1]} · Con ingreso: %{customdata[2]}<extra></extra>'))
    f.update_layout(xaxis_title='Ocupados sin aporte actual (%)',yaxis_title='Ingreso mediano (Bs/mes)',height=420)
    panel.append(('Ingreso, aporte y jornada por territorio','Cada círculo representa un dominio. El tamaño indica el número de ocupados observados y el color, el porcentaje con jornadas extensas. La mediana considera solo ingresos observados. Las asociaciones territoriales no establecen una relación individual.',f))

    X=patrones[DIM].to_numpy()
    w=patrones.peso.to_numpy()
    Xc=X-np.average(X,axis=0,weights=w)
    vals,vec=np.linalg.eigh((Xc.T*w)@Xc/w.sum())
    orden=np.argsort(vals)[::-1]; vals=vals[orden]; ejes=vec[:,orden[:2]]
    for i in range(2):
        if ejes[np.abs(ejes[:,i]).argmax(),i]<0:ejes[:,i]*=-1
    patrones[['pc1','pc2']]=Xc@ejes
    var=100*vals/vals.sum()
    f=go.Figure()
    for n,g in patrones.groupby('grupo'):
        color=COLORES[int(n)-1]; titulo=f'Perfil {n}'
        privaciones=[' · '.join(label for label,activo in zip(CAR,fila) if activo) for fila in g[DIM].values]
        f.add_scatter(x=g.pc1.tolist(),y=g.pc2.tolist(),mode='markers',name=titulo,legendgroup=titulo,
             marker=dict(color=color,size=g.hogares.tolist(),sizemode='area',sizeref=2*patrones.hogares.max()/43**2,sizemin=5),
             text=privaciones,customdata=g.hogares.tolist(),
             hovertemplate='<b>%{fullData.name}</b><br>%{text}<br>%{customdata} hogares observados<extra></extra>')
        centro=np.average(g[['pc1','pc2']],axis=0,weights=g.peso)
        f.add_scatter(x=[centro[0]],y=[centro[1]],mode='markers',name=titulo,legendgroup=titulo,showlegend=False,
            marker=dict(symbol='x',color=color,size=13,line=dict(width=1,color='white')),
            hovertemplate='Centro ponderado · %{fullData.name}<extra></extra>')
    f.update_layout(xaxis_title=f'Componente 1 ({var[0]:.1f} %)',yaxis_title=f'Componente 2 ({var[1]:.1f} %)',height=460)
    panel.append(('Proyección de los seis perfiles de hogares',
        f'K-Means agrupó 2.684 hogares con IPM ≥ ⅓ según cuatro carencias binarias. Cada círculo reúne hogares con la misma combinación; su tamaño indica el recuento observado. Las cruces son los centros ponderados. Dos componentes principales conservan {var[:2].sum():.1f} % de la variación ponderada. La clasificación se ajustó con las cuatro variables originales.',f))

    f=go.Figure(go.Heatmap(z=p[DIM].round(1).values.tolist(),x=CAR,y=[f'Perfil {n}' for n in p.grupo],
        zmin=0,zmax=100,colorscale='Teal',colorbar=dict(title='% del perfil'),
        texttemplate='%{z:.1f}%',hovertemplate='%{y} · %{x}: %{z:.1f}%<extra></extra>'))
    f.update_layout(height=360,yaxis_autorange='reversed')
    panel.append(('Qué carencias distinguen a cada perfil','El perfil 2 reúne el 36,6 % ponderado de los hogares vulnerables y concentra la privación educativa. En el perfil 4 predomina el piso de tierra. En el perfil 6 coinciden educación, piso y saneamiento. Cada celda muestra la proporción ponderada dentro del perfil.',f))
    for _,_,fig in panel:
        fig.update_layout(template='plotly_white',font=dict(family='Arial, sans-serif',size=13,color='#25433e'),
                          hoverlabel=dict(align='left'),paper_bgcolor='#ffffff',plot_bgcolor='#ffffff')
    return panel

def exportar(panel,destino):
    from html import escape
    partes=[]
    for i,(titulo,nota,fig) in enumerate(panel):
        graf=pio.to_html(fig,include_plotlyjs=False,full_html=False,div_id=f'fig-{i+1}',
                         config=dict(responsive=True,displaylogo=False,scrollZoom=False))
        partes.append(f'<section><h2>{escape(titulo)}</h2><p>{escape(nota)}</p>{graf}</section>')
    hoja="""*{box-sizing:border-box}body{margin:0;background:#f0f4f1;color:#173d37;font:16px/1.55 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:clamp(18px,5vw,54px);background:white}h1{font-size:clamp(30px,5vw,48px);line-height:1.13;margin:12px 0}h2{line-height:1.2;margin:0}section{border-top:2px solid #b2cfc1;padding:24px 0 8px}p{max-width:85ch}footer{font-size:.875rem}"""
    destino.write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bolivia · Pobreza y empleo</title><style>'+hoja+'</style><script>'+get_plotlyjs()+'</script><main><header>Encuesta de Hogares 2025 · Bolivia<h1>Pobreza, empleo y territorio</h1><p>Exploración de 19.369 ocupados de 14 años o más y de los hogares que alcanzan el umbral de pobreza multidimensional. La ausencia de aporte actual a la Gestora se usa como aproximación a la informalidad laboral.</p></header>'+''.join(partes)+'<footer>Fuente: Instituto Nacional de Estadística, Encuesta de Hogares 2025, módulos Persona y Vivienda. Tasas y medianas ponderadas por factores de expansión. Datos agregados. <a href="https://anda.ine.gob.bo/index.php/catalog/256">Catálogo ANDA</a>.</footer></main></html>',encoding='utf-8')

if __name__=='__main__':
    exportar(figuras(),RAIZ/'docs/Parte2_Visualizaciones.html')
