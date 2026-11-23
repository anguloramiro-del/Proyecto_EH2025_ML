"""Caso 2. Todas las salidas son agregadas. Ejecutar desde la raíz del proyecto."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn import __version__ as sklearn_version

DIM = ['educacion', 'piso', 'saneamiento', 'electricidad']
NOMBRES = ['Chuquisaca','La Paz','Cochabamba','Oruro','Potosí','Tarija','Santa Cruz','Beni–Pando','Beni–Pando']
SEMILLA = 2026

def media(x, w):
    return float(np.average(x, weights=w))

def mediana(x, w):
    orden = np.argsort(x)
    x, w = np.asarray(x)[orden], np.asarray(w)[orden]
    return float(x[np.searchsorted(w.cumsum(), w.sum()/2)])

def leer(carpeta, modulo, columnas):
    ruta = carpeta / f'EH2025_{modulo}.parquet'
    if ruta.exists():
        return pd.read_parquet(ruta, columns=columnas)
    opciones = sorted(carpeta.glob(f'EH2025_{modulo}*.sav'))
    if len(opciones) != 1:
        raise FileNotFoundError(f'Se necesita el módulo {modulo} en SAV o Parquet en {carpeta}.')
    import pyreadstat
    return pyreadstat.read_sav(str(opciones[0]), usecols=columnas)[0]

def preparar(carpeta):
    p = leer(carpeta,'Persona',['folio','nro','upm','depto','area','factor','s01a_03','s01a_02',
        'aestudio','condact','s04f_35','s04f_36','ylab','tothrs','s04b_12','estrato'])
    h = leer(carpeta,'Vivienda',['folio','factor','s06a_06','s06a_09','s06a_12'])
    assert h.folio.is_unique and not p.duplicated(['folio','nro']).any()
    assert p.folio.isin(h.folio).all() and p.factor.gt(0).all() and h.factor.gt(0).all()
    adultos = p.loc[p.s01a_03.ge(18)].groupby('folio').aestudio.max()
    escolar = h.folio.map(adultos)
    h['educacion'] = escolar.lt(6).where(escolar.notna())
    for campo,col,privados,codigos in [('piso','s06a_06',[1],range(1,10)),
                               ('saneamiento','s06a_09',[3,5],range(1,6)),
                               ('electricidad','s06a_12',[2],[1,2])]:
        h[campo] = h[col].isin(privados).where(h[col].isin(codigos))
    h['ipm'] = (2*h.educacion + 2*h.piso + h.saneamiento + h.electricidad).astype(float)/6
    total_h = len(h)
    h = h.dropna(subset=['ipm']).copy()
    h[DIM] = h[DIM].astype(float)
    h['pobre'] = h.ipm.ge(1/3)
    a = p.loc[p.condact.eq(1) & p.s01a_03.ge(14)].copy()
    a['Empleo_Formal'] = np.select([a.s04f_35.eq(1)&a.s04f_36.eq(1),
                    a.s04f_35.eq(2)|(a.s04f_35.eq(1)&a.s04f_36.eq(2))],[1,0],default=np.nan)
    a['informal'] = 1-a.Empleo_Formal
    a['Ingreso_Laboral'] = a.ylab.where(a.ylab.ge(0))
    a['region'] = a.depto.map(dict(enumerate(NOMBRES,1)))
    assert a.region.notna().all()
    a['jornada_extensa'] = a.tothrs.gt(48).where(a.tothrs.notna())
    return h, a.merge(h[['folio','ipm','pobre']],on='folio',how='left',validate='many_to_one'), total_h

def intervalo(g, unidades):
    from scipy.stats import t
    g = g.dropna(subset=['informal']).copy()
    tasa = media(g.informal,g.factor)
    g['lineal'] = g.factor * (g.informal-tasa)/g.factor.sum()
    totales = g.groupby(['estrato','upm']).lineal.sum().reindex(unidades, fill_value=0)
    varianza = 0.
    for _, z in totales.groupby(level=0):
        if len(z)>1:
            varianza += len(z)*z.var(ddof=1)
        else:
            varianza += (z.iloc[0]-totales.mean())**2
    gl = len(g[['estrato','upm']].drop_duplicates())-g.estrato.nunique()
    margen = t.ppf(0.975,gl)*np.sqrt(varianza)
    return dict(ic_inf=100*max(0,tasa-margen),ic_sup=100*min(1,tasa+margen))

def resumen(g):
    f = g.dropna(subset=['informal'])
    i = g.dropna(subset=['Ingreso_Laboral'])
    j = g.dropna(subset=['jornada_extensa'])
    return dict(n=len(g),n_clasificados=len(f), n_ingreso=len(i),n_upm=g.upm.nunique(),
        informal_pct=100*media(f.informal,f.factor),formal_pct=100*media(f.Empleo_Formal,f.factor),
        ingreso_mediano=mediana(i.Ingreso_Laboral,i.factor), ingreso_medio=media(i.Ingreso_Laboral,i.factor),
        mas_48h_pct=100*media(j.jornada_extensa,j.factor))

def agrupar(h,out):
    v=h[h.pobre].copy()
    X=v[DIM].to_numpy()
    # Silueta no ponderada en una muestra fija de hogares. Ajuste con expansión.
    rng=np.random.default_rng(SEMILLA)
    muestra=rng.choice(len(v),min(2500,len(v)),replace=False)
    modelos={}
    for k in range(2,min(7,len(np.unique(X,axis=0)))):
        km=KMeans(n_clusters=k,n_init=20,random_state=SEMILLA).fit(X,sample_weight=v.factor)
        modelos[k]=(silhouette_score(X[muestra],km.labels_[muestra]),km)
    k=max(modelos,key=lambda n:modelos[n][0])
    km=modelos[k][1]
    v['grupo']=km.labels_+1
    # Conservar las combinaciones de carencias para proyectarlas sin exportar microdatos.
    patrones = (v.groupby(['grupo', *DIM], as_index=False)
                  .agg(hogares=('factor', 'size'), peso=('factor', 'sum')))
    patrones.to_csv(out/'patrones_kmeans.csv', index=False)
    filas=[]
    for grupo,g in v.groupby('grupo'):
        filas.append(dict(grupo=int(grupo),n=len(g),porcentaje=100*g.factor.sum()/v.factor.sum(),
                          ipm_medio=media(g.ipm,g.factor),**{c:100*media(g[c],g.factor) for c in DIM}))
    pd.DataFrame(filas).to_csv(out/'perfiles.csv',index=False)
    return {str(n):round(float(m[0]),4) for n,m in modelos.items()},int(k)

def modelos(a,out):
    a=a.dropna(subset=['informal','ipm','aestudio']).reset_index(drop=True)
    num=['ipm','s01a_03','aestudio']
    cat=['s01a_02','area','region']
    X=a[num+cat].copy()
    cols=ColumnTransformer([('num','passthrough',num),('categoria',OneHotEncoder(
        handle_unknown='ignore',sparse_output=False),cat)])
    y=a.informal.astype(int).to_numpy()
    grupos=a.upm.to_numpy()
    pesos=a.factor.to_numpy()
    ingreso=a.Ingreso_Laboral.to_numpy()
    externas=StratifiedGroupKFold(5,shuffle=True,random_state=SEMILLA)
    train,test=next(externas.split(X,y,grupos))
    cv=StratifiedGroupKFold(4,shuffle=True,random_state=SEMILLA+1)
    particiones=[(train[t],train[v]) for t,v in cv.split(X.iloc[train],y[train],grupos[train])]
    tareas={'informalidad':{'Bosque':RandomForestClassifier(n_estimators=100,min_samples_leaf=12,n_jobs=2,random_state=SEMILLA),
        'Boosting':HistGradientBoostingClassifier(max_iter=100,random_state=SEMILLA,early_stopping=False)},
        'ingreso':{'Bosque':RandomForestRegressor(n_estimators=100,min_samples_leaf=12,n_jobs=2,random_state=SEMILLA),
        'Boosting':HistGradientBoostingRegressor(max_iter=100,random_state=SEMILLA,early_stopping=False)}}
    filas=[]; importancia=[]; matrices={}
    for tarea,opciones in tareas.items():
        clas=tarea=='informalidad'
        yy=y if clas else ingreso
        for nombre,estimador in opciones.items():
            modelo=make_pipeline(cols,estimador)
            valores=[]
            for t,v in particiones+[(train,test)]:
                t=t[np.isfinite(yy[t])];v=v[np.isfinite(yy[v])]
                assert not set(grupos[t]) & set(grupos[v])
                kw={modelo.steps[-1][0]+'__sample_weight':pesos[t]/pesos[t].mean()}
                modelo.fit(X.iloc[t],yy[t],**kw)
                pr=modelo.predict_proba(X.iloc[v])[:,1] if clas else modelo.predict(X.iloc[v])
                valor=roc_auc_score(yy[v],pr,sample_weight=pesos[v]) if clas else mean_absolute_error(yy[v],pr,sample_weight=pesos[v])
                valores.append(valor)
            fila=dict(tarea=tarea,modelo=nombre,n_entrenamiento=len(t),n_prueba=len(v),
                      metrica_cv='AUC' if clas else 'MAE',cv_media=np.mean(valores[:-1]))
            if clas:
                pred=pr>=0.5
                fila.update(auc=valor,ap=average_precision_score(yy[v],pr,sample_weight=pesos[v]),
                    referencia_ap=media(yy[v],pesos[v]),f1=f1_score(yy[v],pred,sample_weight=pesos[v]),
                    exactitud=accuracy_score(yy[v],pred,sample_weight=pesos[v]))
                matrices[nombre]=confusion_matrix(yy[v],pred,labels=[0,1]).tolist()
                imp=permutation_importance(modelo,X.iloc[v],yy[v],scoring='roc_auc',n_repeats=5,
                     random_state=SEMILLA,sample_weight=pesos[v])
                importancia.extend(dict(modelo=nombre,variable=c,caida_auc=m) for c,m in zip(X,imp.importances_mean))
            else:
                referencia=np.full(len(v),mediana(yy[t],pesos[t]))
                fila.update(mae=valor,rmse=mean_squared_error(yy[v],pr,sample_weight=pesos[v])**0.5,
                    r2=r2_score(yy[v],pr,sample_weight=pesos[v]),
                    mae_referencia=mean_absolute_error(yy[v],referencia,sample_weight=pesos[v]))
            filas.append(fila)
    pd.DataFrame(filas).to_csv(out/'modelos.csv',index=False)
    pd.DataFrame(importancia).to_csv(out/'importancia.csv',index=False)
    return matrices

def ejecutar(carpeta,out):
    out.mkdir(exist_ok=True,parents=True)
    h,a,total=preparar(carpeta)
    H=media(h.pobre,h.factor); A=media(h.loc[h.pobre,'ipm'],h.loc[h.pobre,'factor'])
    ambitos={'Nacional':a, 'Bajo el umbral':a[a.pobre.eq(False)],'Desde el umbral':a[a.pobre.eq(True)]}
    for nom,g in a.groupby('region'): ambitos[nom]=g
    unidades=pd.MultiIndex.from_frame(leer(carpeta,"Persona",["estrato","upm"]).drop_duplicates())
    tasas=pd.DataFrame([dict(ambito=n,**resumen(g),**intervalo(g,unidades)) for n,g in ambitos.items()])
    tasas.to_csv(out/'tasas.csv',index=False)
    sil,k=agrupar(h,out)
    matrices=modelos(a,out)
    resultados=dict(hogares=total,hogares_ipm=len(h),incidencia=H,intensidad=A,M0=H*A,
        sensibilidad_umbral={str(k):media(h.ipm.ge(k),h.factor) for k in [1/6,1/3,0.5]},
        ocupados=len(a),ocupados_sin_ipm=int(a.ipm.isna().sum()),sin_ingreso=int(a.Ingreso_Laboral.isna().sum()),
        k=k,siluetas=sil,matrices_observadas=matrices,sklearn=sklearn_version)
    (out/'resumen.json').write_text(json.dumps(resultados,indent=2,ensure_ascii=False))
    return resultados
