# -*- coding: utf-8 -*-
"""
Rozdział III/IV — budowa zbioru uczącego.
Próbkowanie wskaźników spektralnych Sentinel-2 (2020, 2025) w lokalizacjach
budynków ocenionych przez UNOSAT (Mariupol, ocena z 12 maja 2022).

Wynik: analiza_ML/dataset_mariupol.csv  (cechy + etykieta zniszczeń)
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

warnings.filterwarnings("ignore")

DATA = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))   # rastry wskaźników i etykiety UNOSAT (patrz README)
MAR  = os.path.join(DATA, "rasters", "mariupol")
MARB = os.path.join(DATA, "rasters", "mariupol_extra")
SHP  = os.path.join(DATA, "unosat", "mariupol_12May2022", "UNOSAT_Mariupol_Livoberezhyny_DA_12May2022.shp")
OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# Warstwy wskaźnikowe: nazwa -> (plik 2020, plik 2025)
LAYERS = {
    "NBU":        (os.path.join(MAR,  "NBU_Mariupol_2020.tif"),  os.path.join(MAR,  "NBU_Mariupol_2025.tif")),
    "NDBI":       (os.path.join(MAR,  "NDBI_Mariupol_2020.tif"), os.path.join(MAR,  "NDBI_Mariupol_2025.tif")),
    "NDMI":       (os.path.join(MAR,  "NDMI_Mariupol_2020.tif"), os.path.join(MAR,  "NDMI_Mariupol_2025.tif")),
    "NDVI":       (os.path.join(MAR,  "NDVI_Mariupol_2020.tif"), os.path.join(MAR,  "NDVI_Mariupol_2025.tif")),
    "BU":         (os.path.join(MARB, "BU_2020(P).tif"),         os.path.join(MARB, "BU_2025(P).tif")),
    "NBU_rubble": (os.path.join(MARB, "NBU_rubble_2020(P).tif"), os.path.join(MARB, "NBU_rubble_2025(P).tif")),
}

NODATA_THR = -1e30  # wartości poniżej = nodata (float32 -3.4e38)

def sample(path, xy):
    with rasterio.open(path) as r:
        vals = np.array([v[0] for v in r.sample(xy)], dtype="float64")
    vals[vals < NODATA_THR] = np.nan
    return vals

def main():
    g = gpd.read_file(SHP).to_crs("EPSG:32637")
    xy = list(zip(g.geometry.x.values, g.geometry.y.values))
    print(f"Wczytano {len(g)} punktów UNOSAT (CRS -> EPSG:32637)")

    df = pd.DataFrame(index=range(len(g)))
    for name, (f20, f25) in LAYERS.items():
        v20 = sample(f20, xy); v25 = sample(f25, xy)
        df[f"{name}_2020"] = v20
        df[f"{name}_2025"] = v25
        df[f"{name}_diff"] = v25 - v20
        print(f"  próbkowano {name}: valid2020={np.isfinite(v20).sum()} valid2025={np.isfinite(v25).sum()}")

    # Etykieta: kolumna d_Main_D_1 = stan zniszczeń wg oceny z 12 maja 2022
    df["damage_raw"] = g["d_Main_D_1"].values
    df["neighborhood"] = g["Neighborho"].values
    df["confidence"] = g["d_Confid_1"].values
    df["x"] = g.geometry.x.values
    df["y"] = g.geometry.y.values

    # Mapowanie na klasy porządkowe (1..4); kratery pomijamy
    grade = {
        "Possible Damage": 1, "Moderate Damage": 2,
        "Severe Damage": 3, "Destroyed": 4,
    }
    df["grade"] = df["damage_raw"].map(grade)
    # Cel binarny: poważne (Severe+Destroyed)=1 vs lekkie (Possible+Moderate)=0
    df["severe"] = (df["grade"] >= 3).astype("Int64")

    feat_cols = [c for c in df.columns if c.endswith(("_2020","_2025","_diff"))]
    before = len(df)
    df = df.dropna(subset=feat_cols + ["grade"])
    print(f"Po usunięciu nodata/kraterów: {len(df)} / {before} rekordów")
    print("Rozkład klas (grade):", dict(df['grade'].value_counts().sort_index()))
    print("Rozkład binarny (severe):", dict(df['severe'].value_counts()))

    path = os.path.join(OUT, "dataset_mariupol.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print("Zapisano:", path)

if __name__ == "__main__":
    main()
