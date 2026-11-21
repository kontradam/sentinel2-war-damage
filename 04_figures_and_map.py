# -*- coding: utf-8 -*-
"""Rysunki do Rozdziału IV + mapa predykcji (model L2A)."""
import os, json, warnings
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"]="EMPTY_DIR"; os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]=".tif"

ROOT=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(ROOT,"wyniki"); os.makedirs(OUT,exist_ok=True)
AOI=(383440,5210220,401140,5231700)
sc=json.load(open(os.path.join(ROOT,"stac_scenes.json")))
BANDS=["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22"]

df=pd.read_csv(os.path.join(ROOT,"dataset_mariupol_L2A.csv"))
FEAT=[c for c in df.columns if c.endswith(("_2020","_2025","_diff"))]
X=df[FEAT].values; y=df["severe"].astype(int).values
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,n_jobs=-1,class_weight="balanced",random_state=42).fit(Xtr,ytr)

# 1) Macierz błędów L2A
cm=confusion_matrix(yte,rf.predict(Xte)); labs=["lekkie/umiark.","poważne"]
fig,ax=plt.subplots(figsize=(5,4.2)); im=ax.imshow(cm,cmap="Blues")
ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xticklabels(labs);ax.set_yticklabels(labs)
ax.set_xlabel("Predykcja");ax.set_ylabel("Rzeczywista");ax.set_title("Macierz błędów — model L2A (poprawiony)")
for i in range(2):
    for j in range(2): ax.text(j,i,int(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
fig.colorbar(im,fraction=0.046,pad=0.04);fig.tight_layout();fig.savefig(os.path.join(OUT,"cm_L2A.png"),dpi=160);plt.close(fig)

# 2) Ważność cech L2A
imp=pd.Series(rf.feature_importances_,index=FEAT).sort_values().tail(15)
fig,ax=plt.subplots(figsize=(7,6)); imp.plot.barh(ax=ax,color="#2b6cb0")
ax.set_title("Ważność cech — model L2A (top 15)");ax.set_xlabel("Istotność (Gini)")
fig.tight_layout();fig.savefig(os.path.join(OUT,"feature_importance_L2A.png"),dpi=160);plt.close(fig)

# 3) Porównanie L1C vs L2A
fig,ax=plt.subplots(figsize=(5.5,4))
m=["F1","ROC-AUC"]; l1c=[0.557,0.678]; l2a=[0.589,0.705]
xp=np.arange(2);w=0.35
ax.bar(xp-w/2,l1c,w,label="Bazowy (L1C, 3 pasma)",color="#a0aec0")
ax.bar(xp+w/2,l2a,w,label="Poprawiony (L2A, pełne pasma)",color="#2b6cb0")
for i,(a,b) in enumerate(zip(l1c,l2a)):
    ax.text(i-w/2,a+.01,f"{a:.3f}",ha="center",fontsize=9);ax.text(i+w/2,b+.01,f"{b:.3f}",ha="center",fontsize=9)
ax.set_xticks(xp);ax.set_xticklabels(m);ax.set_ylim(0,0.8);ax.set_ylabel("Wartość metryki")
ax.set_title("Wpływ jakości danych na skuteczność modelu");ax.legend()
fig.tight_layout();fig.savefig(os.path.join(OUT,"porownanie_L1C_L2A.png"),dpi=160);plt.close(fig)
print("Rysunki zapisane.")

# 4) Mapa predykcji nad AOI (model L2A)
print("Buduję mapę predykcji (odczyt pasm L2A nad AOI)...")
def read_grid(href,ref_shape,ref_tr):
    with rasterio.open(href) as r:
        win=from_bounds(*AOI,transform=r.transform)
        return r.read(1,window=win,out_shape=ref_shape,resampling=Resampling.bilinear).astype("float32")/10000.0
# siatka 10 m z pasma red 2020
with rasterio.open(sc["assets2020"]["red"]) as r:
    win=from_bounds(*AOI,transform=r.transform); ref_shape=(int(win.height),int(win.width)); ref_tr=r.window_transform(win)
def stack(assets):
    return {b:read_grid(assets[b],ref_shape,ref_tr) for b in BANDS}
def idx(B):
    def nd(a,b): s=a+b; return np.where(s==0,np.nan,(a-b)/s)
    B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12=[B[k] for k in BANDS]
    return {"NDVI":nd(B8,B4),"NDBI":nd(B11,B8),"NBU":nd(B11,B4),"NDMI":nd(B8,B11),
            "NDRE":nd(B8,B5),"NBU_rubble":nd(B11,(B8+B4)),"BU":nd(B11,B8)-nd(B8,B4),
            "BSI":nd((B11+B4),(B8+B2)),"SWIR2_B12":B12,"REDEDGE2_B6":B6}
i20=idx(stack(sc["assets2020"])); i25=idx(stack(sc["assets2025"]))
H,W=ref_shape; cols=[]; mat=[]
for n in i20:
    cols+= [f"{n}_2020",f"{n}_2025",f"{n}_diff"]
    mat+= [i20[n].ravel(), i25[n].ravel(), (i25[n]-i20[n]).ravel()]
Xg=np.array(mat).T
order=[cols.index(c) for c in FEAT]; Xg=Xg[:,order]
valid=np.isfinite(Xg).all(axis=1)
proba=np.full(Xg.shape[0],np.nan,dtype="float32")
proba[valid]=rf.predict_proba(Xg[valid])[:,1]
pmap=proba.reshape(H,W)
# zapis GeoTIFF
prof={"driver":"GTiff","height":H,"width":W,"count":1,"dtype":"float32","crs":"EPSG:32637","transform":ref_tr,"nodata":np.nan}
with rasterio.open(os.path.join(OUT,"mapa_predykcji_L2A.tif"),"w",**prof) as dst: dst.write(pmap,1)
# PNG z nakładką punktów UNOSAT
fig,ax=plt.subplots(figsize=(7,7))
im=ax.imshow(pmap,cmap="RdYlGn_r",vmin=0,vmax=1,extent=[AOI[0],AOI[2],AOI[1],AOI[3]])
g=gpd.read_file(os.path.join(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "unosat", "mariupol_12May2022", "UNOSAT_Mariupol_Livoberezhyny_DA_12May2022.shp")).to_crs("EPSG:32637")
sev=g[g["d_Main_D_1"].isin(["Severe Damage","Destroyed"])]
ax.scatter(sev.geometry.x,sev.geometry.y,s=1,c="black",alpha=0.25,label="UNOSAT: poważne/zniszczone")
ax.set_title("Prawdopodobieństwo poważnych zniszczeń — model L2A\n(Mariupol, dzielnice Livoberezhnyi/Zhovtnevyi)")
ax.set_xlabel("Easting [m]");ax.set_ylabel("Northing [m]")
fig.colorbar(im,fraction=0.046,pad=0.04,label="P(poważne zniszczenia)")
ax.legend(loc="upper right",fontsize=8,markerscale=6)
fig.tight_layout();fig.savefig(os.path.join(OUT,"mapa_predykcji_L2A.png"),dpi=160);plt.close(fig)
print("Mapa predykcji zapisana:",os.path.join(OUT,"mapa_predykcji_L2A.png"))
