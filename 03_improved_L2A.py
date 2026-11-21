# -*- coding: utf-8 -*-
"""
Eksperyment poprawy modelu — dane LEPSZE (wciąż darmowe):
  - poziom L2A (reflektancja powierzchniowa, korekcja atmosferyczna Sen2Cor) zamiast L1C,
  - PEŁNY zestaw pasm Sentinel-2 (w tym Red Edge B5-B7, B8A, B12),
  - maskowanie chmur/cieni warstwą SCL,
  - dodatkowe wskaźniki (NDRE, BSI).
Sceny bezchmurne (0%) pobierane jako publiczne COG-i z AWS (bez logowania):
  2020-08-31 i 2025-08-02, kafel T37TCN.
Porównanie z wersją bazową (L1C, wskaźniki z B4/B8/B11): AUC 0.678 / F1 0.557.
"""
import os, json, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"]="EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]=".tif"

ROOT=os.path.dirname(os.path.abspath(__file__))
SHP=os.path.join(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "unosat", "mariupol_12May2022", "UNOSAT_Mariupol_Livoberezhyny_DA_12May2022.shp")
AOI=(383440,5210220,401140,5231700)  # EPSG:32637
sc=json.load(open(os.path.join(ROOT,"stac_scenes.json")))
BANDS=["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22"]

def read_win(href):
    with rasterio.open(href) as r:
        win=from_bounds(*AOI,transform=r.transform)
        a=r.read(1,window=win).astype("float32")
        return a, r.window_transform(win)

def load_year(assets):
    data={}
    for b in BANDS+["scl"]:
        print("  pobieram",b,"...",flush=True)
        data[b]=read_win(assets[b])
    return data

def sample3(arr, tr, xs, ys):
    H,W=arr.shape; inv=~tr; out=np.full(len(xs),np.nan,dtype="float32")
    for i,(x,y) in enumerate(zip(xs,ys)):
        c,rr=inv*(x,y); col,row=int(c),int(rr)
        if 0<=row<H and 0<=col<W:
            r0,r1=max(0,row-1),min(H,row+2); c0,c1=max(0,col-1),min(W,col+2)
            w=arr[r0:r1,c0:c1]; m=w[np.isfinite(w)]
            if m.size: out[i]=m.mean()
    return out

def indices(band_vals):
    B=band_vals
    B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12=[B[k] for k in ["blue","green","red","rededge1","rededge2","rededge3","nir","nir08","swir16","swir22"]]
    def nd(a,b):
        s=a+b; return np.where(s==0,np.nan,(a-b)/s)
    return {
        "NDVI":nd(B8,B4), "NDBI":nd(B11,B8), "NBU":nd(B11,B4),
        "NDMI":nd(B8,B11), "NDRE":nd(B8,B5),
        "NBU_rubble":nd(B11,(B8+B4)),
        "BU":nd(B11,B8)-nd(B8,B4),
        "BSI":nd((B11+B4),(B8+B2)),
        # surowe pasma istotne dla gruzu/betonu
        "SWIR2_B12":B12, "REDEDGE2_B6":B6,
    }

def main():
    g=gpd.read_file(SHP).to_crs("EPSG:32637")
    xs,ys=g.geometry.x.values,g.geometry.y.values
    print("Ładowanie sceny 2020 (L2A)..."); d20=load_year(sc["assets2020"])
    print("Ładowanie sceny 2025 (L2A)..."); d25=load_year(sc["assets2025"])

    # maska ważności z SCL: usuń chmury(3,8,9,10), cień(2?),wodę(6) opcjonalnie; zostaw wegetację/zabudowę/glebę
    def valid_mask(scl_arr_tr):
        a,tr=scl_arr_tr
        bad=np.isin(a,[0,1,3,8,9,10,11])  # nodata,defekt,chmura,cień chmury,cirrus,śnieg
        return (~bad).astype("float32"), tr

    # próbkowanie pasm w punktach
    def bandvals(d):
        return {b: sample3(d[b][0]*(1/10000.0), d[b][1], xs, ys) for b in BANDS}
    bv20=bandvals(d20); bv25=bandvals(d25)
    ix20=indices(bv20); ix25=indices(bv25)

    df=pd.DataFrame()
    for name in ix20:
        df[f"{name}_2020"]=ix20[name]; df[f"{name}_2025"]=ix25[name]
        df[f"{name}_diff"]=ix25[name]-ix20[name]
    # etykieta
    grade={"Possible Damage":1,"Moderate Damage":2,"Severe Damage":3,"Destroyed":4}
    df["grade"]=g["d_Main_D_1"].map(grade).values
    df["severe"]=(df["grade"]>=3).astype("Int64")
    # ważność SCL (piksel czysty w obu latach)
    vm20,_=valid_mask(d20["scl"]); vm25,_=valid_mask(d25["scl"])
    df["valid20"]=sample3(vm20,d20["scl"][1],xs,ys)
    df["valid25"]=sample3(vm25,d25["scl"][1],xs,ys)

    FEAT=[c for c in df.columns if c.endswith(("_2020","_2025","_diff"))]
    n0=len(df)
    df=df[(df["valid20"]>0.5)&(df["valid25"]>0.5)]
    df=df.dropna(subset=FEAT+["grade"])
    print(f"\nRekordów po maskowaniu/czyszczeniu: {len(df)}/{n0}  | cech: {len(FEAT)}")
    df.to_csv(os.path.join(ROOT,"dataset_mariupol_L2A.csv"),index=False,encoding="utf-8")

    X=df[FEAT].values; y=df["severe"].astype(int).values
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,n_jobs=-1,class_weight="balanced",random_state=42)
    pp=cross_val_predict(rf,X,y,cv=cv,method="predict_proba")[:,1]; p=(pp>=0.5).astype(int)
    f1=f1_score(y,p); auc=roc_auc_score(y,pp); acc=accuracy_score(y,p)
    print("\n=== WYNIK L2A + pełne pasma + Red Edge (CV 5-fold) ===")
    print(f"F1={f1:.3f}  ROC-AUC={auc:.3f}  Accuracy={acc:.3f}")
    print(f"(baseline L1C: F1=0.557  AUC=0.678)  ->  zmiana AUC {auc-0.678:+.3f}")
    # ważność cech
    rf.fit(X,y)
    imp=pd.Series(rf.feature_importances_,index=FEAT).sort_values(ascending=False)
    print("\nTop 8 cech:"); print(imp.head(8).round(4).to_string())
    imp.to_csv(os.path.join(ROOT,"wyniki","feature_importance_L2A.csv"),encoding="utf-8")
    json.dump({"F1":f1,"AUC":auc,"ACC":acc,"n":len(df),"features":FEAT,
               "top_features":imp.head(10).to_dict()},
              open(os.path.join(ROOT,"wyniki","metryki_L2A.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

if __name__=="__main__":
    main()
