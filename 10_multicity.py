# -*- coding: utf-8 -*-
"""Multi-city: (A) czy istnieje zwarty przenaszalny wzor (analiza cech, offline),
(B) test transferu na Sumach - miasto, ktorego model nigdy nie widzial.
Sumy: produkt UNOSAT to komorki siatki 500 m (Shape_Area=250000 m2) z flaga
zniszczen - inny typ danych niz punktowy Mariupol. Sprawdzamy, czy predykcje
modelu sa PODWYZSZONE w komorkach oznaczonych przez UNOSAT."""
import os, json, urllib.request, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"]="EMPTY_DIR"; os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]=".tif"
ROOT=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(ROOT,"wyniki_poc"); os.makedirs(OUT,exist_ok=True)
SUMY=os.path.join(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "unosat", "full_damage", "Sumy_22March2022_RDA.shp")
BANDS=["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22"]
STAC="https://earth-search.aws.element84.com/v1/search"

df=pd.read_csv(os.path.join(ROOT,"dataset_mariupol_L2A.csv"))
FEAT=[c for c in df.columns if c.endswith(("_2020","_2025","_diff"))]
Xtr=df[FEAT].fillna(df[FEAT].median()).values; ytr=df["severe"].astype(int).values
rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,n_jobs=-1,class_weight="balanced",
                          random_state=42).fit(Xtr,ytr)
imp=pd.Series(rf.feature_importances_,index=FEAT).sort_values(ascending=False)

# ---------- (A) FIGURA: zwarty wzor ----------
import re
grp={}
for k,v in imp.items():
    b=re.sub(r'_(2020|2025|diff)$','',k); grp[b]=grp.get(b,0)+v
gs=pd.Series(grp).sort_values()
fig,axs=plt.subplots(1,2,figsize=(12,4.6))
gs.plot.barh(ax=axs[0],color="#2b6cb0")
axs[0].set_xlabel("Sumaryczna istotność (Gini)"); axs[0].set_title("(a) Waga wskaźnika w modelu")
# krzywa retencji AUC — wartosci z eksperymentu 5-fold CV
K=[3,5,8,12,18,30]; A=[0.648,0.666,0.689,0.703,0.708,0.705]
axs[1].plot(K,A,"o-",color="#2b6cb0"); axs[1].axhline(0.705,ls="--",c="gray",lw=1)
for k,a in zip(K,A): axs[1].annotate(f"{a:.3f}",(k,a),textcoords="offset points",xytext=(0,7),ha="center",fontsize=9)
axs[1].set_xlabel("Liczba cech w modelu zredukowanym"); axs[1].set_ylabel("ROC-AUC (walidacja krzyżowa)")
axs[1].set_title("(b) Trafność vs liczba cech"); axs[1].set_ylim(0.6,0.74)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"wzor_przenaszalny.png"),dpi=160); plt.close(fig)
print("Figura (a)/(b) zapisana.")

# ---------- (B) TRANSFER: Sumy ----------
def search(bbox_ll,y):
    body={"collections":["sentinel-2-l2a"],"bbox":list(bbox_ll),
          "datetime":f"{y}-06-01T00:00:00Z/{y}-09-15T23:59:59Z",
          "query":{"eo:cloud_cover":{"lt":12}},"limit":50}
    req=urllib.request.Request(STAC,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","User-Agent":"poc"})
    r=json.load(urllib.request.urlopen(req,timeout=60))
    f=sorted(r["features"],key=lambda x:x["properties"].get("eo:cloud_cover",100))[0]
    print(f"  {y}: {f['id']} chmury={f['properties'].get('eo:cloud_cover'):.1f}%")
    return {b:f["assets"][b]["href"] for b in BANDS}

g=gpd.read_file(SUMY)
flagged=g["Main_Damag"].isin([6,14]).values  # komorki oznaczone przez UNOSAT
print(f"\nSumy: {len(g)} komorek 500 m | oznaczonych: {flagged.sum()} | nieoznaczonych: {(~flagged).sum()}")
bb_ll=transform_bounds(g.crs,"EPSG:4326",*g.total_bounds)
a20=search(bb_ll,2020); a25=search(bb_ll,2025)
with rasterio.open(a20["red"]) as r: SCRS=r.crs
gp=g.to_crs(SCRS); AOI=tuple(gp.total_bounds+np.array([-200,-200,200,200]))
with rasterio.open(a20["red"]) as r:
    win=from_bounds(*AOI,transform=r.transform); H,W=int(win.height),int(win.width); TR=r.window_transform(win)
print("  siatka:",H,"x",W)
def rd(href):
    with rasterio.open(href) as r:
        from rasterio.enums import Resampling
        return r.read(1,window=from_bounds(*AOI,transform=r.transform),out_shape=(H,W),
                      resampling=Resampling.bilinear).astype("float32")/10000.0
def nd(a,b): s=a+b; return np.where(s==0,np.nan,(a-b)/s)
def idx(A):
    B={b:rd(A[b]) for b in BANDS}
    B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12=[B[k] for k in BANDS]
    return {"NDVI":nd(B8,B4),"NDBI":nd(B11,B8),"NBU":nd(B11,B4),"NDMI":nd(B8,B11),"NDRE":nd(B8,B5),
            "NBU_rubble":nd(B11,(B8+B4)),"BU":nd(B11,B8)-nd(B8,B4),"BSI":nd((B11+B4),(B8+B2)),
            "SWIR2_B12":B12,"REDEDGE2_B6":B6}
print("  pobieram pasma 2020..."); i20=idx(a20)
print("  pobieram pasma 2025..."); i25=idx(a25)
cols=[]; mat=[]
for n in i20: cols+=[f"{n}_2020",f"{n}_2025",f"{n}_diff"]; mat+=[i20[n].ravel(),i25[n].ravel(),(i25[n]-i20[n]).ravel()]
Xg=np.array(mat).T; Xg=Xg[:,[cols.index(c) for c in FEAT]]
med=np.nanmedian(Xg,axis=0); Xg=np.where(np.isfinite(Xg),Xg,med)
P=rf.predict_proba(Xg)[:,1].reshape(H,W)

# agregacja P do komorek 500 m
from rasterio.features import rasterize
vals=[]
for geom,fl in zip(gp.geometry,flagged):
    m=geometry_mask([geom],(H,W),TR,invert=True)
    if m.sum()>0: vals.append((fl,float(np.nanmean(P[m]))))
va=np.array([v for _,v in vals]); fa=np.array([f for f,_ in vals])
print("\n=== TRANSFER NA SUMY (model z Mariupola, miasto niewidziane) ===")
print(f"  Srednie P w komorkach OZNACZONYCH przez UNOSAT : {va[fa].mean():.3f} (n={fa.sum()})")
print(f"  Srednie P w komorkach NIEOZNACZONYCH           : {va[~fa].mean():.3f} (n={(~fa).sum()})")
from scipy import stats
if fa.sum()>5 and (~fa).sum()>5:
    u,p=stats.mannwhitneyu(va[fa],va[~fa],alternative="greater")
    print(f"  Test Manna-Whitneya (H1: oznaczone > nieoznaczone): p={p:.4f}")
    auc=(stats.rankdata(np.r_[va[fa],va[~fa]])[:fa.sum()].sum()-fa.sum()*(fa.sum()+1)/2)/(fa.sum()*(~fa).sum())
    print(f"  AUC (P jako detektor komorki oznaczonej)         : {auc:.3f}")
json.dump({"sumy_flagged_meanP":float(va[fa].mean()),"sumy_unflagged_meanP":float(va[~fa].mean()),
           "n_flagged":int(fa.sum()),"n_unflagged":int((~fa).sum())},
          open(os.path.join(OUT,"transfer_sumy.json"),"w"),indent=2)
print("Zapisano transfer_sumy.json")
