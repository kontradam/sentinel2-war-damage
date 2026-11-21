# -*- coding: utf-8 -*-
"""Test przenaszalnosci (multi-city): model wytrenowany na Mariupolu stosowany do
budynkow 'No Visible Damage' (kod 6) w Charkowie i Sumach. Mierzy, czy model
poprawnie NIE oznacza nienaruszonych budynkow w innych miastach (swoistosc/transfer).
UWAGA: etykiety UNOSAT z III.2022, obrazy 2020/2025 -> mozliwa niespojnosc czasowa (opis w pracy)."""
import os, json, urllib.request, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"]="EMPTY_DIR"; os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]=".tif"
ROOT=os.path.dirname(os.path.abspath(__file__)); LB=os.path.join(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "unosat", "full_damage")
BANDS=["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22"]
STAC="https://earth-search.aws.element84.com/v1/search"

# model (30 cech) z CSV
df=pd.read_csv(os.path.join(ROOT,"dataset_mariupol_L2A.csv"))
FEAT=[c for c in df.columns if c.endswith(("_2020","_2025","_diff"))]
rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,n_jobs=-1,class_weight="balanced",
                          random_state=42).fit(df[FEAT].values,df["severe"].astype(int).values)

def search(bbox_ll, year):
    body={"collections":["sentinel-2-l2a"],"bbox":list(bbox_ll),
          "datetime":f"{year}-06-01T00:00:00Z/{year}-09-15T23:59:59Z",
          "query":{"eo:cloud_cover":{"lt":15}},"limit":50}
    req=urllib.request.Request(STAC,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","User-Agent":"poc"})
    r=json.load(urllib.request.urlopen(req,timeout=60))
    f=sorted(r["features"],key=lambda x:x["properties"].get("eo:cloud_cover",100))[0]
    print("  %d: %s chmury=%.1f%%"%(year,f["id"],f["properties"].get("eo:cloud_cover")))
    return {b:f["assets"][b]["href"] for b in BANDS}

def sample_city(name, shp, dmg_col):
    print("\n===",name,"===")
    g=gpd.read_file(shp)
    g=g[g[dmg_col]==6]  # tylko 'No Visible Damage'
    print("  punktow 'No Visible Damage':",len(g))
    bb=transform_bounds(g.crs,"EPSG:4326",*g.total_bounds)
    a20=search(bb,2020); a25=search(bb,2025)
    # CRS sceny
    with rasterio.open(a20["red"]) as r: scene_crs=r.crs
    gp=g.to_crs(scene_crs); xs=gp.geometry.x.values; ys=gp.geometry.y.values
    minx,miny,maxx,maxy=xs.min()-100,ys.min()-100,xs.max()+100,ys.max()+100
    def read(href):
        with rasterio.open(href) as r:
            win=from_bounds(minx,miny,maxx,maxy,transform=r.transform)
            return r.read(1,window=win).astype("float32")/10000.0, r.window_transform(win)
    def sample(arr,tr):
        H,W=arr.shape; inv=~tr; out=np.full(len(xs),np.nan,"float32")
        for i,(x,y) in enumerate(zip(xs,ys)):
            c,rr=inv*(x,y); col,row=int(c),int(rr)
            if 0<=row<H and 0<=col<W:
                w=arr[max(0,row-1):row+2,max(0,col-1):col+2]; m=w[np.isfinite(w)]
                if m.size: out[i]=m.mean()
        return out
    def bandvals(a): return {b:sample(*read(a[b])) for b in BANDS}
    print("  pobieram pasma 2020/2025...")
    b20=bandvals(a20); b25=bandvals(a25)
    def nd(a,b): s=a+b; return np.where(s==0,np.nan,(a-b)/s)
    def idx(B):
        B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12=[B[k] for k in BANDS]
        return {"NDVI":nd(B8,B4),"NDBI":nd(B11,B8),"NBU":nd(B11,B4),"NDMI":nd(B8,B11),"NDRE":nd(B8,B5),
                "NBU_rubble":nd(B11,(B8+B4)),"BU":nd(B11,B8)-nd(B8,B4),"BSI":nd((B11+B4),(B8+B2)),
                "SWIR2_B12":B12,"REDEDGE2_B6":B6}
    i20=idx(b20); i25=idx(b25)
    cols=[]; mat=[]
    for n in i20: cols+=[f"{n}_2020",f"{n}_2025",f"{n}_diff"]; mat+=[i20[n],i25[n],i25[n]-i20[n]]
    X=np.array(mat).T; X=X[:,[cols.index(c) for c in FEAT]]
    ok=np.isfinite(X).all(axis=1)
    P=rf.predict_proba(X[ok])[:,1]
    print("  ocenionych punktow:",ok.sum())
    print("  Srednie P(powazne)=%.3f | mediana=%.3f | odsetek P>0.5 (falszywe alarmy)=%.1f%%"%(
        P.mean(),np.median(P),100*np.mean(P>0.5)))
    return P

# Charkow: klasa w Main_Dam_4; Sumy: w Main_Damag
res={}
res["Charków"]=sample_city("Charków", os.path.join(LB,"Kharkiv_23March2022_RDA.shp"), "Main_Dam_4")
res["Sumy"]=sample_city("Sumy", os.path.join(LB,"Sumy_22March2022_RDA.shp"), "Main_Damag")
print("\n=== PODSUMOWANIE (swoistosc na nienaruszonych budynkach innych miast) ===")
for k,P in res.items():
    print("  %s: odsetek falszywych alarmow (P>0.5) = %.1f%%  (n=%d)"%(k,100*np.mean(P>0.5),len(P)))
