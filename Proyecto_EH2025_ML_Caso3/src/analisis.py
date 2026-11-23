"""EH2025. Resultados agregados de la persona seleccionada por Kish."""
from pathlib import Path
import json, string
import duckdb, numpy as np, pandas as pd, pyreadstat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.inspection import permutation_importance
from scipy.stats import t as student

DEP=['Chuquisaca','La Paz','Cochabamba','Oruro','Potosí','Tarija','Santa Cruz','Beni','Pando']
DISC='Percepcion_Discriminacion'
VIC='Victimizacion'
INS='Inseguridad'
MOT=['s09a_01'+c for c in string.ascii_lowercase[:12]]
OBJ=[DISC,VIC,INS]
def construir(carpeta):
    claves=['folio','nro']
    q,meta=pyreadstat.read_sav(str(carpeta/'EH2025_Discriminacion.sav'),usecols=claves+MOT+['s09b_02a','s09b_02b','s09b_01','s09b_03','s09a_02','ponderador','upm','estrato','depto','area'])
    p,_=pyreadstat.read_sav(str(carpeta/'EH2025_Persona.sav'),usecols=claves+['s01a_03','s01a_02','p0','niv_ed_g','s01a_09'])
    assert q.folio.is_unique and not p.duplicated(claves).any() and q.ponderador.gt(0).all()
    q['_orden_modulo'] = np.arange(len(q))
    with duckdb.connect() as con:
        con.register('q',q);con.register('p',p)
        a=con.sql('SELECT q.* EXCLUDE (_orden_modulo), p.* EXCLUDE(folio,nro) FROM q LEFT JOIN p USING (folio,nro) ORDER BY q._orden_modulo').df()
    assert len(a)==len(q) and a.s01a_03.ge(15).all()
    for c in ['s09b_02a','s09b_02b']:
        assert 'ninguno' in meta.variable_value_labels[c][8].lower()
        assert a[c].dropna().isin(range(1,9)).all()
    m=a[MOT]
    assert m.stack().isin([1,2,3,4]).all()
    a[DISC]=np.select([m.eq(1).any(axis=1),m.eq(2).all(axis=1)],[1.,0.],default=np.nan)
    v=a[['s09b_02a','s09b_02b']]
    a[VIC]=np.select([v.isin(range(1,8)).any(axis=1),v.eq(8).any(axis=1)],[1.,0.],default=np.nan)
    a[INS]=a.s09b_01.map({1:1,2:1,3:0,4:0})
    a['Departamento']=a.depto.map(dict(enumerate(DEP,1)))
    a['Sexo']=a.s01a_02.map({1:'Hombre',2:'Mujer'})
    a['Área']=a.area.map({1:'Urbana',2:'Rural'})
    a['Edad']=pd.cut(a.s01a_03,[14,28,39,49,59,130],labels=['15–28','29–39','40–49','50–59','60 o más'])
    a['Pobreza monetaria']=a.p0.map({1:'Con pobreza',0:'Sin pobreza'})
    a['Autoidentificación NPIOC o afroboliviana']=a.s01a_09.map({1:'Sí',2:'No'})
    a['Nivel educativo']=a.niv_ed_g.map({0:'Ninguno',1:'Primaria',2:'Secundaria',3:'Superior',4:'Otro'})
    return a

def estimar(a,grupo,c,diseno):
    g=grupo.dropna(subset=[c])
    w=g.ponderador; propor=np.average(g[c],weights=w)
    lineal=w*(g[c]-propor)/w.sum()
    z=lineal.groupby([g.estrato,g.upm]).sum().reindex(diseno,fill_value=0)
    var=sum(len(h)*h.var(ddof=1) if len(h)>1 else (h.iloc[0]-z.mean())**2 for _,h in z.groupby(level=0))
    gl=len(g[['upm','estrato']].drop_duplicates())-g.estrato.nunique()
    err=student.ppf(.975,gl)*np.sqrt(var) if gl>0 else np.nan
    return dict(n=len(g),positivos=int(g[c].sum()),excluidos=len(grupo)-len(g),tasa=100*propor,
                inferior=100*max(0,propor-err),superior=100*min(1,propor+err))

def modelos(a, salida=None):
    """Compara los tres candidatos en las mismas UPM por desenlace."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    categoricas = ['Sexo', 'Área', 'Departamento', 'Nivel educativo',
                   'Pobreza monetaria', 'Autoidentificación NPIOC o afroboliviana']
    resultados, importancias, validacion, puntos, cortes = [], [], [], [], []
    for objetivo in [DISC, VIC]:
        datos = a.dropna(subset=[objetivo])
        X = datos[['s01a_03'] + categoricas].rename(columns={'s01a_03': 'Edad en años'}).copy()
        X[categoricas] = X[categoricas].fillna('Sin dato')
        y = datos[objetivo].astype(int).to_numpy()
        w = datos.ponderador.to_numpy()
        upm = datos.upm.to_numpy()
        externa = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
        train, test = next(externa.split(X, y, groups=upm))
        assert set(upm[train]).isdisjoint(upm[test]) and np.unique(y[test]).size == 2
        interna = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2027)
        pliegues = list(interna.split(X.iloc[train], y[train], groups=upm[train]))
        candidatos = {
            'Regresión logística': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=12, n_jobs=2, random_state=2026),
            'XGBoost': XGBClassifier(n_estimators=160, learning_rate=0.05, max_depth=2,
                                     min_child_weight=8, reg_lambda=3, subsample=0.9,
                                     objective='binary:logistic', eval_metric='logloss',
                                     tree_method='hist', n_jobs=2, random_state=2026)
        }
        for nombre, estimador in candidatos.items():
            identificacion = dict(objetivo=objetivo, modelo=nombre)
            # Se conserva la misma representación numérica entre candidatos.
            pre = ColumnTransformer([
                ('edad', StandardScaler(), ['Edad en años']),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricas)
            ])
            modelo = Pipeline([('preparacion', pre), ('clasificador', estimador)])
            def ajustar(indices):
                pesos = w[indices]
                modelo.fit(X.iloc[indices], y[indices], clasificador__sample_weight=pesos / pesos.mean())

            prob_cv = np.full(len(train), np.nan)
            aps, aucs = [], []
            for pliegue, (t, v) in enumerate(pliegues, 1):
                tr, ev = train[t], train[v]
                assert set(upm[tr]).isdisjoint(upm[ev]) and np.unique(y[ev]).size == 2
                ajustar(tr)
                p = modelo.predict_proba(X.iloc[ev])[:, 1]
                assert np.isnan(prob_cv[v]).all()
                prob_cv[v] = p
                ap = average_precision_score(y[ev], p, sample_weight=w[ev])
                auc = roc_auc_score(y[ev], p, sample_weight=w[ev])
                aps.append(ap)
                aucs.append(auc)
                validacion.append(dict(**identificacion, pliegue=pliegue, n=len(ev),
                                       positivos=int(y[ev].sum()), upm=len(np.unique(upm[ev])),
                                       prevalencia=np.average(y[ev], weights=w[ev]), AP=ap, AUC=auc))
            assert np.isfinite(prob_cv).all()
            prec, rec, umbrales = precision_recall_curve(y[train], prob_cv, sample_weight=w[train])
            denominador = prec[:-1] + rec[:-1]
            f1_cv = np.divide(2 * prec[:-1] * rec[:-1], denominador, out=np.zeros_like(denominador),
                              where=denominador > 0)
            umbral = float(umbrales[np.argmax(f1_cv)])
            ajustar(train)
            prob = modelo.predict_proba(X.iloc[test])[:, 1]
            obs, pesos = y[test], w[test]
            r = dict(**identificacion, AP_cv=np.mean(aps), DE_AP_cv=np.std(aps, ddof=1),
                     AUC_cv=np.mean(aucs), DE_AUC_cv=np.std(aucs, ddof=1),
                     AUC=roc_auc_score(obs, prob, sample_weight=pesos),
                     AP=average_precision_score(obs, prob, sample_weight=pesos),
                     AP_referencia=np.average(obs, weights=pesos), n_entrenamiento=len(train),
                     n_prueba=len(test), upm_entrenamiento=len(np.unique(upm[train])),
                     upm_prueba=len(np.unique(upm[test])))
            for regla, u in [('F1 en validación interna', umbral), ('Referencia 0,5', 0.5)]:
                pred = prob >= u
                tn, fp, fn, tp = confusion_matrix(obs, pred, labels=[0, 1]).ravel()
                detalle = dict(umbral=u, precision=precision_score(obs, pred, sample_weight=pesos, zero_division=0),
                               sensibilidad=recall_score(obs, pred, sample_weight=pesos, zero_division=0),
                               F1=f1_score(obs, pred, sample_weight=pesos, zero_division=0),
                               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
                cortes.append(dict(**identificacion, regla=regla, **detalle))
                if regla == 'F1 en validación interna':
                    r.update(detalle)
            resultados.append(r)
            fpr, tpr, _ = roc_curve(obs, prob, sample_weight=pesos)
            pre_test, rec_test, _ = precision_recall_curve(obs, prob, sample_weight=pesos)
            for curva, xs, ys in [('ROC', fpr, tpr), ('PR', rec_test, pre_test)]:
                # Se publican 51 coordenadas de visualización por curva.
                # Las métricas conservan el cálculo exacto con todas las observaciones.
                eje = np.linspace(0, 1, 51)
                if curva == 'PR':
                    xs, ys = xs[::-1], ys[::-1]
                valores = np.interp(eje, xs, ys)
                puntos.extend(dict(**identificacion, curva=curva, x=round(x, 2), y=round(z, 4))
                              for x, z in zip(eje, valores))
            imp = permutation_importance(modelo, X.iloc[test], obs, scoring='average_precision',
                                        sample_weight=pesos, n_repeats=10, random_state=2026)
            for k, media, sd in zip(X.columns, imp.importances_mean, imp.importances_std):
                importancias.append(dict(**identificacion, variable=k, caida_AP=media, DE=sd))
    metricas = pd.DataFrame(resultados)
    metricas['seleccionado_cv'] = metricas.AP_cv == metricas.groupby('objetivo').AP_cv.transform('max')
    if salida is not None:
        pd.DataFrame(validacion).to_csv(salida/'validacion_pliegues.csv', index=False)
        pd.DataFrame(puntos).to_csv(salida/'curvas_modelos.csv', index=False)
        pd.DataFrame(cortes).to_csv(salida/'umbrales_matrices.csv', index=False)
    return metricas, pd.DataFrame(importancias)

def ejecutar(carpeta,salida):
    salida.mkdir(parents=True,exist_ok=True)
    a=construir(carpeta); diseno=pd.MultiIndex.from_frame(a[['estrato','upm']].drop_duplicates())
    filas=[]
    for dim in ['Nacional','Departamento','Área','Sexo','Edad','Nivel educativo','Pobreza monetaria','Autoidentificación NPIOC o afroboliviana']:
        grupos=[('Bolivia',a)] if dim=='Nacional' else a.groupby(dim,observed=True)
        for nombre,g in grupos:
            for c in OBJ:filas.append(dict(dimension=dim,grupo=nombre,indicador=c,**estimar(a,g,c,diseno)))
    tasas=pd.DataFrame(filas);tasas.to_csv(salida/'tasas.csv',index=False)
    r=tasas.query("dimension=='Departamento'").pivot(index='grupo',columns='indicador',values='tasa').sort_index()
    # Seguridad territorial: se agrupa exclusivamente por victimización e inseguridad.
    X=StandardScaler().fit_transform(r[[VIC,INS]])
    candidatos={}
    for k in [2,3,4]:
        labels=KMeans(k,n_init=50,random_state=2026).fit_predict(X)
        candidatos[k]=(silhouette_score(X,labels),labels,int(np.bincount(labels).min()))
    k=max([k for k in candidatos if candidatos[k][2]>=2],key=lambda k:candidatos[k][0])
    r['perfil_raw']=candidatos[k][1]
    orden=r.groupby('perfil_raw')[INS].mean().sort_values().index
    r['perfil']=r.pop('perfil_raw').map(dict(zip(orden,range(1,k+1))))
    r.to_csv(salida/'regiones.csv')
    pd.DataFrame([dict(k=j,silueta=x[0],minimo=x[2],seleccionado=j==k) for j,x in candidatos.items()]).to_csv(salida/'seleccion_k.csv',index=False)
    met,imp=modelos(a,salida);met.to_csv(salida/'modelos.csv',index=False);imp.to_csv(salida/'importancia.csv',index=False)
    motivos=[]
    for col,n in zip(MOT,['Color de piel','Pertenencia a pueblo','Procedencia','Orientación o identidad','Edad','Sexo','Idioma','Vestimenta','Discapacidad','Religión','Condición económica o social','Otro motivo']):
        valido=a[col].isin([1,2]); motivos.append(dict(motivo=n,n=int(valido.sum()),porcentaje=100*np.average(a.loc[valido,col].eq(1),weights=a.loc[valido,'ponderador'])))
    pd.DataFrame(motivos).to_csv(salida/'motivos.csv',index=False)
    print(tasas[tasas.dimension.isin(['Nacional','Área','Sexo'])].to_string(index=False))
    print(r.to_string()); print(met.round(3).to_string(index=False))
