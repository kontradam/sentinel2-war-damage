# -*- coding: utf-8 -*-
"""PoC (klamra Rozdz. IV): to samo miasto liczone RECZNIE na zajeciach (Kreminna)
przetworzone AUTOMATYCZNIE modelem. Porownanie: reczna analiza wskaznikowa (NBU_rubble)
vs mapa prawdopodobienstwa z modelu (pelne pasma L2A, 30 cech, pobrane skryptem ze STAC).
Bez etykiet dla Kreminnej -> porownanie jakosciowe (przestrzenne)."""
import os, json, urllib.request, warnings
import numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"]="EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]=".tif"

ROOT=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(ROOT,"wyniki_poc"); os.makedirs(OUT,exist_ok=True)
MANUAL=os.path.join(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "rasters", "kreminna")
AOI=(439014,5430360,446658,5436311)  # Kreminna, EPSG:32637
BANDS=["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22"]
STAC="https://earth-search.aws.element84.com/v1/search"

# --- model: przetrenuj z CSV (gwarancja kolejnosci 30 cech) ---
df=pd.read_csv(os.path.join(ROOT,"dataset_mariupol_L2A.csv"))
FEAT=[c for c in df.columns if c.endswith(("_2020","_2025","_diff"))]
rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,n_jobs=-1,
                          class_weight="balanced",random_state=42).fit(df[FEAT].values,df["severe"].astype(int).values)
print("Model gotowy, cech:",len(FEAT))

# --- STAC search dla AOI (2020 i 2025, lato, male zachmurzenie) ---
def bbox_lonlat():
    return transform_bounds("EPSG:32637","EPSG:4326",*AOI)
def search(year):
    body={"collections":["sentinel-2-l2a"],"bbox":list(bbox_lonlat()),
          "datetime":f"{year}-06-01T00:00:00Z/{year}-09-15T23:59:59Z",
          "query":{"eo:cloud_cover":{"lt":10}},"limit":50}
    req=urllib.request.Request(STAC,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","User-Agent":"poc"})
    r=json.load(urllib.request.urlopen(req,timeout=60))
    feats=sorted(r["features"],key=lambda f:f["properties"].get("eo:cloud_cover",100))
    if not feats: raise RuntimeError(f"brak scen {year}")
    best=feats[0]
    print(f"  {year}: {best['id']} chmury={best['properties'].get('eo:cloud_cover'):.1f}%")
    return {b:best["assets"][b]["href"] for b in BANDS}

print("STAC 2020..."); a20=search(2020)
print("STAC 2025...");
try: a25=search(2025)
except Exception: print("  2025 brak -> probuje 2024"); a25=search(2024)

# --- odczyt okienny pasm nad AOI (siatka 10 m z red 2020) ---
def read_grid(href,shape):
    with rasterio.open(href) as r:
        win=from_bounds(*AOI,transform=r.transform)
        return r.read(1,window=win,out_shape=shape,resampling=Resampling.bilinear).astype("float32")/10000.0
with rasterio.open(a20["red"]) as r:
    win=from_bounds(*AOI,transform=r.transform); H,W=int(win.height),int(win.width); ref_tr=r.window_transform(win)
print("Siatka:",H,"x",W)
def stack(a): return {b:read_grid(a[b],(H,W)) for b in BANDS}
def nd(a,b): s=a+b; return np.where(s==0,np.nan,(a-b)/s)
def idx(B):
    B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12=[B[k] for k in BANDS]
    return {"NDVI":nd(B8,B4),"NDBI":nd(B11,B8),"NBU":nd(B11,B4),"NDMI":nd(B8,B11),
            "NDRE":nd(B8,B5),"NBU_rubble":nd(B11,(B8+B4)),"BU":nd(B11,B8)-nd(B8,B4),
            "BSI":nd((B11+B4),(B8+B2)),"SWIR2_B12":B12,"REDEDGE2_B6":B6}
print("Pobieram pasma 2020..."); i20=idx(stack(a20))
print("Pobieram pasma 2025..."); i25=idx(stack(a25))
cols=[]; mat=[]
for n in i20:
    cols+=[f"{n}_2020",f"{n}_2025",f"{n}_diff"]; mat+=[i20[n].ravel(),i25[n].ravel(),(i25[n]-i20[n]).ravel()]
Xg=np.array(mat).T; Xg=Xg[:,[cols.index(c) for c in FEAT]]
valid=np.isfinite(Xg).all(axis=1)
proba=np.full(Xg.shape[0],np.nan,dtype="float32"); proba[valid]=rf.predict_proba(Xg[valid])[:,1]
pmap=proba.reshape(H,W)
# zapis GeoTIFF
prof={"driver":"GTiff","height":H,"width":W,"count":1,"dtype":"float32","crs":"EPSG:32637","transform":ref_tr,"nodata":np.nan}
with rasterio.open(os.path.join(OUT,"kreminna_pred.tif"),"w",**prof) as dst: dst.write(pmap,1)
print("P(poważne) statystyki: min=%.2f mediana=%.2f max=%.2f udział>0.5=%.1f%%" % (
    np.nanmin(pmap),np.nanmedian(pmap),np.nanmax(pmap),100*np.nanmean(pmap>0.5)))

# --- reczna mapa NBU_rubble difference (Kreminna, z zajec) ---
def load_manual(name):
    for fn in [name+".tif", name.replace("rubble","Rubble")+".tif"]:
        p=os.path.join(MANUAL,fn)
        if os.path.exists(p):
            with rasterio.open(p) as r: return r.read(1).astype("float32"), r.bounds
    return None,None
man,manb=load_manual("NBU_rubble_difference")

# --- figura porownawcza ---
fig,axs=plt.subplots(1,2,figsize=(12,5.5))
im0=axs[0].imshow(man,cmap="RdYlGn_r",extent=[manb.left,manb.right,manb.bottom,manb.top]) if man is not None else None
axs[0].set_title("Analiza ręczna (zajęcia)\nΔNBU_rubble 2025−2020")
if im0: fig.colorbar(im0,ax=axs[0],fraction=0.046,pad=0.04,label="ΔNBU_rubble")
im1=axs[1].imshow(pmap,cmap="RdYlGn_r",vmin=0,vmax=1,extent=[AOI[0],AOI[2],AOI[1],AOI[3]])
axs[1].set_title("Model uczenia maszynowego (skrypt)\nP(poważne zniszczenia)")
fig.colorbar(im1,ax=axs[1],fraction=0.046,pad=0.04,label="P(poważne)")
for ax in axs: ax.set_xlabel("Easting [m]"); ax.set_ylabel("Northing [m]")
fig.suptitle("Kreminna — porównanie: analiza ręczna vs model (proof of concept)")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"kreminna_porownanie.png"),dpi=160); plt.close(fig)
print("Zapisano:",os.path.join(OUT,"kreminna_porownanie.png"))
