# -*- coding: utf-8 -*-
"""
Rozdział IV — trening i ewaluacja modelu Random Forest.
Zadanie A: klasyfikacja binarna (poważne uszkodzenia vs lekkie/umiarkowane)
Zadanie B: klasyfikacja 4-klasowa (stopień zniszczeń)
Baseline: progowanie różnicy NBU_rubble (metoda bez ML).
Zapisuje metryki, macierze błędów, ważność cech i model.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, roc_auc_score)
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "wyniki"); os.makedirs(OUT, exist_ok=True)
RS = 42

df = pd.read_csv(os.path.join(ROOT, "dataset_mariupol.csv"))
FEATURES = [c for c in df.columns if c.endswith(("_2020","_2025","_diff"))]
print(f"Rekordów: {len(df)} | cech: {len(FEATURES)}")

X = df[FEATURES].values
results = {}

def plot_cm(cm, labels, title, fname):
    fig, ax = plt.subplots(figsize=(5.2,4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predykcja"); ax.set_ylabel("Rzeczywista klasa"); ax.set_title(title)
    thr = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,int(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>thr else "black", fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,fname), dpi=160); plt.close(fig)

# ---------------- Zadanie A: binarne ----------------
print("\n=== ZADANIE A: klasyfikacja binarna (poważne vs lekkie) ===")
yA = df["severe"].astype(int).values
XtrA,XteA,ytrA,yteA = train_test_split(X,yA,test_size=0.3,stratify=yA,random_state=RS)
rfA = RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=2,
                             n_jobs=-1,class_weight="balanced",random_state=RS)
rfA.fit(XtrA,ytrA)
pA = rfA.predict(XteA); ppA = rfA.predict_proba(XteA)[:,1]
accA=accuracy_score(yteA,pA); f1A=f1_score(yteA,pA); aucA=roc_auc_score(yteA,ppA)
cvA=cross_val_score(rfA,X,yA,cv=StratifiedKFold(5,shuffle=True,random_state=RS),scoring="f1")
print(classification_report(yteA,pA,target_names=["lekkie/umiark.","poważne"],digits=3))
print(f"Accuracy={accA:.3f}  F1={f1A:.3f}  ROC-AUC={aucA:.3f}  CV-F1={cvA.mean():.3f}±{cvA.std():.3f}")
plot_cm(confusion_matrix(yteA,pA),["lekkie/umiark.","poważne"],
        "Macierz błędów — model binarny (RF)","cm_binarny.png")
results["binarny"]={"accuracy":accA,"f1":f1A,"roc_auc":aucA,
                    "cv_f1_mean":cvA.mean(),"cv_f1_std":cvA.std(),
                    "report":classification_report(yteA,pA,target_names=["lekkie/umiark.","poważne"],output_dict=True)}

# ---------------- Zadanie B: 4-klasowe ----------------
print("\n=== ZADANIE B: klasyfikacja 4-klasowa (stopień zniszczeń) ===")
yB = df["grade"].astype(int).values
names4=["Możliwe","Umiarkowane","Poważne","Zniszczony"]
XtrB,XteB,ytrB,yteB = train_test_split(X,yB,test_size=0.3,stratify=yB,random_state=RS)
rfB = RandomForestClassifier(n_estimators=400,min_samples_leaf=2,n_jobs=-1,
                             class_weight="balanced",random_state=RS)
rfB.fit(XtrB,ytrB)
pB=rfB.predict(XteB)
accB=accuracy_score(yteB,pB); f1mB=f1_score(yteB,pB,average="macro")
print(classification_report(yteB,pB,target_names=names4,digits=3))
print(f"Accuracy={accB:.3f}  F1-macro={f1mB:.3f}")
plot_cm(confusion_matrix(yteB,pB),names4,"Macierz błędów — model 4-klasowy (RF)","cm_4klasy.png")
results["czteroklasowy"]={"accuracy":accB,"f1_macro":f1mB,
                          "report":classification_report(yteB,pB,target_names=names4,output_dict=True)}

# ---------------- Ważność cech (z modelu binarnego) ----------------
imp=pd.Series(rfA.feature_importances_,index=FEATURES).sort_values(ascending=False)
imp.to_csv(os.path.join(OUT,"feature_importance.csv"),encoding="utf-8")
fig,ax=plt.subplots(figsize=(7,6))
imp.iloc[::-1].plot.barh(ax=ax,color="#2b6cb0")
ax.set_title("Ważność cech (Random Forest, model binarny)")
ax.set_xlabel("Istotność (Gini importance)")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"feature_importance.png"),dpi=160); plt.close(fig)
print("\nTop 6 cech:"); print(imp.head(6).round(4).to_string())

# ---------------- Baseline: progowanie NBU_rubble_diff ----------------
print("\n=== BASELINE: progowanie różnicy NBU_rubble (bez ML) ===")
rub=df["NBU_rubble_diff"].values
best=None
for thr in np.quantile(rub,np.linspace(0.3,0.9,25)):
    pred=(rub>=thr).astype(int)
    f=f1_score(yA,pred)
    if best is None or f>best[1]: best=(thr,f)
thr,f1base=best
accbase=accuracy_score(yA,(rub>=thr).astype(int))
print(f"Najlepszy próg NBU_rubble_diff={thr:.4f}  ->  F1={f1base:.3f}  Acc={accbase:.3f}")
results["baseline_progowy"]={"prog":float(thr),"f1":float(f1base),"accuracy":float(accbase)}

# ---------------- Zapis ----------------
joblib.dump({"model_bin":rfA,"model_4kl":rfB,"features":FEATURES}, os.path.join(OUT,"rf_model.joblib"))
with open(os.path.join(OUT,"metryki.json"),"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)
print("\nZapisano metryki, wykresy i model do:", OUT)
print(f"\nPODSUMOWANIE: RF binarny F1={f1A:.3f} vs baseline progowy F1={f1base:.3f}  (zysk {f1A-f1base:+.3f})")
