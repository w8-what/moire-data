
import os, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# (1) Viewing data structure and format
for fn in ['Rxx_matrix_E-87mV_nm.csv','Rxx_matrix_E-96mV_nm.csv','Rxx_matrix_E-99mV_nm.csv','Rxx_matrix_E-103mV_nm.csv']:
    p='source_data/'+fn
    print('\n',fn, os.path.getsize(p))
    print(open(p).readline()[:500])
    df=pd.read_csv(p, header=None)
    print(df.shape)
    print(df.iloc[:5,:5])


# (2) not sure, some data analysis
for E in [87,96,99,103]:
    df=pd.read_csv(f'source_data/Rxx_matrix_E-{E}mV_nm.csv')
    T=df.iloc[:,0].astype(float).to_numpy(); nu=np.array([float(c) for c in df.columns[1:]])
    R=df.iloc[:,1:].astype(float).to_numpy()
    print('E',E,'T',T.min(),T.max(),len(T),'nu',nu.min(),nu.max(),len(nu),'R min max',np.nanmin(R),np.nanmax(R))
    # low-res threshold pockets
    sc=[]
    for j,v in enumerate(nu):
        m=(R[:,j]<=20)&(T<=0.8)
        if np.sum(m)>=2 or (np.any(m) and T[m].max()>=0.12): sc.append((v,T[m].max(),np.sum(m)))
    if sc:
        print('SC ranges', (min(v for v,_,__ in sc), max(v for v,_,__ in sc)), 'n',len(sc),'max Tc',max(t for _,t,__ in sc))
        # cluster ranges
        clusters=[]; curr=[]; prev=None
        for x in sc:
            if prev is None or x[0]-prev<=0.003: curr.append(x)
            else: clusters.append(curr); curr=[x]
            prev=x[0]
        if curr: clusters.append(curr)
        for c in clusters: print('  ',c[0][0],c[-1][0],'max',max(t for _,t,_ in c), 'n',len(c))
    # central peak at v=1? values
    for v0 in [0.88,0.94,1.0,1.06,1.13]:
        j=np.argmin(np.abs(nu-v0)); print(' v',nu[j], 'Rlow',R[0,j], 'Rhi',R[-1,j])

# (3) Making heatplots of all of the data

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
fig, axs = plt.subplots(1,4, figsize=(16,4))
for ax,E in zip(axs,[87,96,99,103]):
    df=pd.read_csv(f'source_data/Rxx_matrix_E-{E}mV_nm.csv')
    T=df.iloc[:,0].astype(float).to_numpy(); nu=np.array([float(c) for c in df.columns[1:]])
    R=df.iloc[:,1:].astype(float).to_numpy()
    mesh=ax.pcolormesh(nu,T,np.log10(np.clip(R,1,1e6)),shading='auto',cmap='RdBu_r',vmin=0,vmax=5)
    ax.set_title(f'E=-{E}')
    ax.set_xlabel('nu')
    ax.set_ylim(0,4)
axs[0].set_ylabel('T')
fig.colorbar(mesh, ax=axs.ravel().tolist())
fig.savefig('output/heatmaps_all.png')


# (4) Making linecuts of all datasets

fig, axs=plt.subplots(2,2,figsize=(10,8),dpi=160,sharex=True)
vs=[0.84,0.88,0.92,0.94,0.96,0.98,1.0,1.02,1.04,1.06,1.1,1.13]
for ax,E in zip(axs.ravel(),[87,96,99,103]):
    df=pd.read_csv(f'source_data/Rxx_matrix_E-{E}mV_nm.csv')
    T=df.iloc[:,0].astype(float).to_numpy(); nu=np.array([float(c) for c in df.columns[1:]]); R=df.iloc[:,1:].astype(float).to_numpy()
    for v0 in vs:
        j=np.argmin(abs(nu-v0)); y=np.clip(R[:,j],1,np.inf)
        ax.plot(T,y,label=f'{nu[j]:.3f}',lw=1)
    ax.set_yscale('log')
    ax.set_title(f'E=-{E}')
    ax.set_ylim(1,3e5)
    ax.grid(True,alpha=.2)
axs[0,0].legend(ncol=3,fontsize=6)
for ax in axs[-1,:]: ax.set_xlabel('T')
for ax in axs[:,0]: ax.set_ylabel('R')
fig.tight_layout()
fig.savefig('output/linecuts_all_log.png')


# (5) T_N extractiona

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

for E in [87,96,99,103]:
 df=pd.read_csv(f'source_data/Rxx_matrix_E-{E}mV_nm.csv')
 T=df.iloc[:,0].astype(float).to_numpy(); nu=np.array([float(c) for c in df.columns[1:]]); R=df.iloc[:,1:].astype(float).to_numpy()
 logR=np.log10(np.clip(R,1,1e6)); logRs=gaussian_filter(logR,sigma=(1.15,1.0))
 print('\nE',E)
 for v0 in [0.96,0.98,0.99,1.0,1.01,1.02,1.03]: # Why is the range hard coded
  j=np.argmin(abs(nu-v0)); y=logRs[:,j]
  idxs, props=find_peaks(-y,prominence=0.006,distance=3)
  cands=[(float(T[i]),float(props['prominences'][k])) for k,i in enumerate(idxs) if 0.5<=T[i]<=3.8]
  m=(T>=2.5)&(T<=3.8)
  tn=float(T[m][np.argmin(y[m])])
  ratio_low_high=10**(y[0]-y[-1])
  print(v0,'nu',nu[j],'logR0,4',y[0],y[-1],'ratio',ratio_low_high,'cands',cands[:4],'hi_min',tn)


# (6) T_coh extraction  

from sklearn.metrics import r2_score

def tcoh_auto(E):
 df=pd.read_csv(f'source_data/Rxx_matrix_E-{E}mV_nm.csv')
 T=df.iloc[:,0].astype(float).to_numpy(); nu=np.array([float(c) for c in df.columns[1:]]); R=df.iloc[:,1:].astype(float).to_numpy()
 Rs=gaussian_filter(np.clip(R,1,None), sigma=(1.0,1.0))
 outs=[]
 for j,v in enumerate(nu):
  y=Rs[:,j]
  # skip SC deep if low t <=20 in range 0.05-0.5? maybe still can fit above sc transition but not for Tcoh
  # candidates tmax from 0.8 to 4
  best=0
  for tmax in T[(T>=0.8)&(T<=4.0)]:
   m=(T>=0.4)&(T<=tmax)
   if m.sum()<5: continue
   x=T[m]**2; Y=y[m]
   A=np.vstack([np.ones_like(x),x]).T
   coef, *_=np.linalg.lstsq(A,Y,rcond=None)
   pred=A@coef
   # normalized max residual vs change over range and absolute R. choose MAPE? 
   rel=np.max(np.abs(Y-pred)/np.maximum(np.abs(Y),10))
   # require positive curvature and decent increasing trend
   if coef[1]>0 and rel<0.12:
    best=float(tmax)
  if best>0: outs.append((v,best))
 return outs
for E in [87,96,99,103]:
 outs=tcoh_auto(E)
 # summarize above 0.5
 print('\nE',E,'n',len(outs))
 print([(round(v,3),t) for v,t in outs[:10]])
 print('left',[(round(v,3),t) for v,t in outs if v<0.9][-10:])
 print('right',[(round(v,3),t) for v,t in outs if v>1.05][:20], '... last', [(round(v,3),t) for v,t in outs if v>1.05][-5:])


# (7) T' extraction

def tprime_auto(E, region=(0.86,0.98), fit_min=2.2, tol=0.1):
 df=pd.read_csv(f'source_data/Rxx_matrix_E-{E}mV_nm.csv')
 T=df.iloc[:,0].astype(float).to_numpy(); nu=np.array([float(c) for c in df.columns[1:]]); R=df.iloc[:,1:].astype(float).to_numpy()
 Rs=gaussian_filter(np.clip(R,1,None), sigma=(1.0,1.0))
 outs=[]
 for v0 in np.arange(region[0], region[1]+1e-6, 0.008):
  j=np.argmin(abs(nu-v0)); v=nu[j]; y=Rs[:,j]
  # Skip central insulating columns: low/high ratio >1.3
  if np.mean(y[T<=0.2]) > 1.3*y[T>=3.8].mean(): continue
  mfit=(T>=fit_min)&(T<=4.0)
  if mfit.sum()<5: continue
  X=np.vstack([np.ones(mfit.sum()),T[mfit]]).T
  coef,*_=np.linalg.lstsq(X,y[mfit],rcond=None)
  if coef[1]<=0: continue
  pred=coef[0]+coef[1]*T
  # deviations for t from 0.2 to 4; use absolute ratio normalized by local R span? 
  rel=np.abs(y-pred)/np.maximum(np.abs(y),20)
  # find lowest T such that for all higher T deviations under tol (plus a small tolerance)
  cand=None
  for t in T[(T>=0.2)&(T<=3.5)]:
   mh=(T>=t)&(T<=4.0)
   if np.percentile(rel[mh],90) <= tol:
    cand=float(t); break
  if cand is not None:
    outs.append((float(v),cand, float(np.percentile(rel[T>=cand],90))))
 return outs
for E in [87,96,99,103]:
 outs=tprime_auto(E)
 print('\nE',E,[(round(v,3),t) for v,t,_ in outs])



