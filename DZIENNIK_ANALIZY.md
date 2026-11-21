# Dziennik analizy — część praktyczna pracy magisterskiej
**Automatyczna detekcja zniszczeń miejskich z wykorzystaniem uczenia maszynowego i danych satelitarnych**
Autor: Adam Juśko · Data wykonania analizy: 2026-09-10

> Ten dokument to pełny, chronologiczny zapis toku badań własnych: źródeł danych,
> narzędzi, decyzji metodycznych, przeprowadzonych eksperymentów, uzyskanych liczb
> oraz ograniczeń. Stanowi bazę tekstową dla Rozdziału III (metodologia) i IV (wyniki).
> Mapowanie na rozdziały pracy znajduje się na końcu.

---

## 1. Cel części praktycznej
Zbudowanie i ocena nadzorowanego modelu uczenia maszynowego, który na podstawie
wielospektralnych wskaźników satelitarnych Sentinel-2 (stan przed- i powojenny)
klasyfikuje stopień zniszczenia zabudowy miejskiej, przy użyciu niezależnych
etykiet referencyjnych (ground truth) z oceny UNOSAT.

Pytanie badawcze: **czy darmowe dane optyczne Sentinel-2 (10 m) pozwalają na
automatyczną, wiarygodną detekcję zniszczeń budynków przy użyciu ML?**

---

## 2. Dane źródłowe

### 2.1. Cechy (features) — wskaźniki spektralne Sentinel-2
- Źródło: sceny Sentinel-2 MSI, poziom L1C, misja Copernicus/ESA (dane bezpłatne).
- Miasto treningowe: **Mariupol** (dzielnice Livoberezhnyi i Zhovtnevyi).
- Daty: stan przedwojenny **2020** vs powojenny **2025**.
- Wskaźniki (rastry 10 m, EPSG:32637 / UTM 37N), dla 2020 i 2025 oraz różnica 2025−2020:
  - **NBU** — (B11−B4)/(B11+B4), zabudowa/beton
  - **NDBI** — (B11−B8)/(B11+B8), zabudowa
  - **NDMI** — wilgotność
  - **NDVI** — roślinność (B8, B4)
  - **BU** — NDBI−NDVI, kompozyt zabudowy
  - **NBU_rubble** — (B11−(B8+B4))/(B11+(B8+B4)), indeks gruzowiska (kluczowy dla zniszczeń)
- Łącznie **18 cech** (6 wskaźników × {2020, 2025, różnica}).
- Rastry pochodzą z Ćwiczenia 9 (zajęcia z GIS: dane własne i projekty studenckie; rastry nie są publikowane).

### 2.2. Etykiety (ground truth) — UNOSAT
- Źródło: **UNOSAT/UNITAR**, „Updated Building Damage Assessment — Livoberezhnyi
  and Zhovtnevyi Districts, Mariupol", kod zdarzenia **CE20220223UKR**,
  analiza obrazów bardzo wysokiej rozdzielczości (WorldView-2/3) z 12 maja 2022.
- Pobrano przez Humanitarian Data Exchange (HDX), format SHP:
  `UNOSAT_LivoberezhnyiDistrict_DamageAssessment_12May2022_shp.zip`
  (host: unosat-maps.web.cern.ch). Dane bezpłatne, publiczne.
- Zawartość: **5 660 punktów** (budynki i obiekty) z klasą zniszczeń.
- Kolumna `d_Main_D_1` (stan na 12.05.2022):
  - Moderate Damage: 3002
  - Severe Damage: 2132
  - Destroyed: 315
  - Possible Damage: 194
  - Impact Crater (road/field): 17 (pominięte)
- CRS oryginału: EPSG:3857 → przeliczono do EPSG:32637 (zgodność z rastrami).
- **Wszystkie 5 660 punktów mieści się w zasięgu rastrów Mariupola** (weryfikacja przestrzenna).

---

## 3. Środowisko i narzędzia
- Python 3.11; biblioteki: `rasterio` 1.4.4, `geopandas` 1.1.4, `scikit-learn`,
  `numpy`, `pandas`, `matplotlib`, `joblib`.
- Skrypty (folder `analiza_ML/`):
  - `01_build_dataset.py` — budowa zbioru uczącego (próbkowanie rastrów w punktach UNOSAT).
  - `02_train_eval.py` — trening Random Forest, ewaluacja, macierze błędów, ważność cech, baseline.
  - (planowane) `03_predict_map.py` — mapa predykcji nad całym AOI.

---

## 4. Metodyka krok po kroku
1. **Wczytanie etykiet** UNOSAT (5 660 pkt) i reprojekcja do EPSG:32637.
2. **Próbkowanie** wartości 18 wskaźników w lokalizacji każdego budynku
   (rasterio `sample`). Wersja finalna: **średnia z okna 3×3 piksele** (budynek
   zajmuje kilka pikseli — redukcja szumu pojedynczego piksela).
3. **Etykietowanie**:
   - Cel porządkowy 4-klasowy: Możliwe(1) < Umiarkowane(2) < Poważne(3) < Zniszczony(4).
   - Cel binarny: **poważne uszkodzenia** (Severe+Destroyed = 1) vs
     **lekkie/umiarkowane** (Possible+Moderate = 0).
4. **Czyszczenie**: usunięcie nodata i kraterów → **5 618–5 624** rekordów.
   Rozkład binarny: 2 439 poważne vs 3 179 lekkie (zbalansowany).
5. **Model**: Random Forest (400 drzew, `class_weight=balanced`), podział
   train/test 70/30 ze stratyfikacją + 5-krotna walidacja krzyżowa.
6. **Baseline bez ML**: progowanie różnicy NBU_rubble (dobór progu maksymalizujący F1).
7. **Ewaluacja**: accuracy, precision, recall, F1, ROC-AUC, macierz błędów,
   ważność cech (Gini importance).

---

## 5. Przeprowadzone eksperymenty i wyniki (dane bezpłatne)

### 5.1. Model binarny (poważne vs lekkie) — próbkowanie 1 piksel
| Metryka | Wartość |
|---|---|
| Accuracy | 0.620 |
| F1 (klasa „poważne") | 0.541 |
| ROC-AUC | 0.663 |
| CV-F1 (5-fold) | 0.537 ± 0.018 |

### 5.2. Model binarny — próbkowanie okno 3×3 (wersja finalna)
| Metryka | Wartość |
|---|---|
| F1 | **0.557** |
| ROC-AUC | **0.678** |

Okno 3×3 daje niewielką, ale spójną poprawę (redukcja szumu).

### 5.3. Model 4-klasowy (stopień zniszczeń)
| Metryka | Wartość |
|---|---|
| Accuracy | 0.571 |
| F1-macro | 0.343 |

Klasy skrajne (Możliwe, Zniszczony) są rzadkie i słabo rozróżnialne;
model myli sąsiednie stopnie (Umiarkowane↔Poważne).

### 5.4. Baseline progowy (NBU_rubble_diff, bez ML)
| Metryka | Wartość |
|---|---|
| Najlepszy próg | 0.0062 |
| F1 | 0.543 |
| Accuracy | 0.482 |

**RF minimalnie przewyższa progowanie** — ML wydobywa dodatkowy sygnał, ale przewaga jest niewielka.

### 5.5. Ważność cech (top 6, model binarny)
1. NBU_2025 (0.070)
2. NBU_diff (0.068)
3. NDVI_2025 (0.064)
4. NBU_2020 (0.062)
5. BU_2025 (0.062)
6. NBU_rubble_2025 (0.060)

Wniosek: dominują wskaźniki zabudowy (NBU/BU) w stanie powojennym oraz ich zmiana;
NBU_rubble istotny, ale nie dominujący.

### 5.6. Warianty testowane (transparentność metodyczna)
| Wariant etykiet | F1 | AUC |
|---|---|---|
| Zniszczony(4) vs reszta | 0.138 | 0.624 |
| Poważne+Zniszcz.(3,4) vs (1,2) — przyjęty | 0.541 | 0.663 |
| ≥Umiarkowane(2,3,4) vs Możliwe(1) | 0.982* | 0.648 |
| Zniszczony(4) vs lekkie(1,2) bez „Poważne" | 0.143 | 0.680 |

\* wynik pozorny — 99 % obiektów należy do klasy pozytywnej (brak wartości informacyjnej).

### 5.7. Agregacja do siatki (jednostka analizy = komórka, nie budynek)
| Siatka | Komórek | F1 | AUC |
|---|---|---|---|
| 50 m | 530 | 0.673 | 0.545 |
| 100 m | 709 | 0.630 | 0.653 |
| 200 m | 507 | 0.561 | 0.703 |

Agregacja przestrzenna nieznacznie poprawia AUC przy większych komórkach
(dopasowanie skali analizy do rozdzielczości danych), kosztem szczegółowości.

---

## 6. Wnioski i ograniczenia (na podstawie danych bezpłatnych)
1. **Model działa i jest zwalidowany** na niezależnym ground truth (UNOSAT), ale
   osiąga **umiarkowaną skuteczność** (AUC ≈ 0.66–0.68).
2. **Główne ograniczenie: rozdzielczość 10 m** — jeden piksel obejmuje kilka
   budynków/ulicę; detekcja na poziomie pojedynczego budynku jest na granicy możliwości.
3. **Dane optyczne L1C, bez korekcji atmosferycznej (nie L2A)** i **bez SAR** —
   brak informacji strukturalnej (koherencja Sentinel-1), która najsilniej różnicuje zniszczenia.
4. **Różnice sezonowe/atmosferyczne** 2020 vs 2025 wprowadzają szum.
5. ML daje niewielką przewagę nad prostym progowaniem — sygnał w danych optycznych
   jest ograniczony niezależnie od klasyfikatora.
6. **To wynik naukowo wartościowy**: ilościowo potwierdza granice darmowej
   teledetekcji optycznej i uzasadnia potrzebę danych płatnych/SAR (patrz Rozdz. dalsze badania).

---

## 7. Eksperyment poprawy modelu — dane lepszej jakości (WYKONANY)

Cel: **zmierzyć, o ile poprawia się model po zastąpieniu/uzupełnieniu danych bazowych
danymi lepszej jakości.** Wszystkie dane pozyskano jako publiczne COG-i z chmury AWS
(katalog STAC earth-search, bez logowania) — czyli wciąż darmowe, ale wyższej jakości
niż wskaźniki bazowe liczone tylko z 3 pasm L1C.

### 7.1. Wariant A — L2A + pełny zestaw pasm + Red Edge (WYKONANY, realny wynik)
- Poziom **L2A** (reflektancja powierzchniowa, korekcja atmosferyczna Sen2Cor) zamiast L1C.
- Pełny zestaw pasm: dodatkowo **B2 (blue), B5/B6/B7 (Red Edge), B8A, B12 (SWIR2)**.
- Maskowanie chmur/cieni warstwą **SCL**.
- Nowe wskaźniki: **NDRE** (Red Edge), **BSI** (Bare Soil Index), surowe B6, B12.
- Sceny bezchmurne (0 %): **2020-08-31** i **2025-08-02**, kafel T37TCN.
- 30 cech, 5 643 budynki. Skrypt: `03_improved_L2A.py`.

| Model | F1 | ROC-AUC | Accuracy |
|---|---|---|---|
| Bazowy (L1C, 3 pasma → 6 wskaźników, 18 cech) | 0.557 | 0.678 | — |
| **Poprawiony (L2A + pełne pasma, 30 cech)** | **0.589** | **0.705** | 0.650 |
| **Przyrost** | **+0.032** | **+0.027** | — |

**Najważniejsze cechy modelu poprawionego (Gini):** REDEDGE2_B6 (2020 i 2025),
SWIR2_B12 (2020, różnica), NDVI_2025, NBU_2025, BU_2025, BSI_diff.
→ **Wniosek: największy przyrost informacji dają właśnie pasma NIEobecne w wersji bazowej
(Red Edge B6 i SWIR2 B12).** To ilościowo dowodzi, że ograniczenie do 3 pasm (B4/B8/B11)
zubaża model, a lepsza jakość/kompletność danych realnie podnosi skuteczność.

### 7.2. Wariant B — Sentinel-1 SAR (PRÓBA + oszacowanie)
- Dane S1 GRD (VV/VH) zlokalizowano dla obu epok (2020-08 i 2025-08) w katalogach
  AWS (`sentinel-s1-l1c`, requester-pays) oraz Microsoft Planetary Computer
  (`sentinel-1-grd`, `sentinel-1-rtc`).
- **Ograniczenie techniczne:** publicznie dostępny GRD jest w geometrii radarowej
  (nie geokodowany) i wymaga kalibracji radiometrycznej + korekcji terenowej
  Range-Doppler (SNAP/pyroSAR) przed próbkowaniem w punktach; produkt geokodowany
  (RTC) jest za bramką subskrypcji. Pełny pipeline SAR wykracza poza ramy czasowe/narzędziowe
  tego etapu.
- **Oszacowanie z literatury:** fuzja Sentinel-1 (SAR) + Sentinel-2 (optyka) w detekcji
  zniszczeń zabudowy przewyższa pojedynczy sensor, szczególnie dla uszkodzeń umiarkowanych
  (Aimaiti i in., „War Related Building Damage Assessment in Kyiv, Ukraine, Using Sentinel-1
  Radar and Sentinel-2 Optical Images", 2022). Koherencja SAR jest bardzo czuła na utratę
  geometrii dachu przy zawaleniu. Oczekiwany przyrost dokładności z dodania SAR: rzędu
  kilku–kilkunastu punktów procentowych (zależnie od stopnia zniszczeń) — to najsilniejszy
  kolejny lewar, rekomendowany jako pierwszy krok dalszych badań.

### 7.3. Dane komercyjne (wysoka rozdzielczość) — poza zakresem, oszacowanie
- PlanetScope (3 m) / Maxar (0.3–0.5 m) — **dane płatne**, umożliwiają detekcję na poziomie
  pojedynczego budynku (której 10 m nie zapewnia). W literaturze służą głównie jako źródło
  etykiet referencyjnych (jak WorldView w UNOSAT). Realnie niedostępne w tym projekcie;
  opisane jako kierunek przy dostępie do środków/licencji.

**Podsumowanie eksperymentu:** przejście na lepsze (ale wciąż darmowe) dane L2A z pełnym
zestawem pasm dało **realny, zmierzony przyrost** skuteczności (AUC 0.678 → 0.705). Kolejne
lewary (SAR, dane komercyjne) opisano z oszacowaniem, oddzielając to, co zmierzono, od
tego, co zaprojektowano.

---

## 8. Mapowanie na strukturę pracy

### Rozdział III — Metodologia badań własnych
- **3.1 Obszar badań** → sekcja 2 (Mariupol, dzielnice, kontekst wojenny; miasta studenckie jako tło).
- **3.2 Pozyskiwanie i wstępne przetwarzanie danych** → sekcje 2.1, 2.2, 4 (Sentinel-2 L1C, UNOSAT, reprojekcja, próbkowanie 3×3).
- **3.3 Wskaźniki spektralne** → sekcja 2.1 (wzory NBU, NDBI, NDMI, NDVI, BU, NBU_rubble).
- **3.4 Przygotowanie danych treningowych** → sekcja 4 (etykiety UNOSAT, mapowanie klas, balans, podział).
- **3.5 Budowa modelu** → sekcja 4.5–4.6 (Random Forest, parametry, cechy).
- **3.6 Walidacja i metryki** → sekcja 4.7 (CV, metryki, baseline progowy).

### Rozdział IV — Wyniki i analiza
- **4.1 Analiza indeksowa (bez ML)** → statystyki różnicowe, histogramy (CSV z Ćw. 9), mapy różnicowe.
- **4.2 Wyniki modelu ML** → sekcje 5.1–5.5 (metryki, macierze błędów, ważność cech, model binarny i 4-klasowy).
- **4.3 Porównanie z baseline i dyskusja** → sekcje 5.4, 5.6, 5.7 (RF vs progowanie; warianty; agregacja siatkowa).
- **4.4 Dyskusja ograniczeń** → sekcja 6.
- **4.5 Kierunki dalszych badań** → sekcja 7 (dane płatne/SAR i szacowany przyrost skuteczności).

### Załączniki
- Kod źródłowy: `01_build_dataset.py`, `02_train_eval.py`, `03_predict_map.py`.
- Wykresy: `wyniki/cm_binarny.png`, `wyniki/cm_4klasy.png`, `wyniki/feature_importance.png`.
- Tabele metryk: `wyniki/metryki.json`, `wyniki/feature_importance.csv`.
