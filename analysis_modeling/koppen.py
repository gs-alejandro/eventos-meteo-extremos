import rasterio
from scipy.ndimage import distance_transform_edt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

KOPPEN_TIF = "../datasets/koppen/1991_2020/koppen_geiger_0p00833333.tif"
COL_LON, COL_LAT, COL_ID = "longitud", "latitud", "indicativo"

KOPPEN_LEGEND = {
    1:  'Af',   # Tropical, rainforest                  [0 0 255]
    2:  'Am',   # Tropical, monsoon                     [0 120 255]
    3:  'Aw',   # Tropical, savannah                    [70 170 250]
    4:  'BWh',  # Arid, desert, hot                     [255 0 0]
    5:  'BWk',  # Arid, desert, cold                    [255 150 150]
    6:  'BSh',  # Arid, steppe, hot                     [245 165 0]
    7:  'BSk',  # Arid, steppe, cold                    [255 220 100]
    8:  'Csa',  # Temperate, dry summer, hot summer     [255 255 0]
    9:  'Csb',  # Temperate, dry summer, warm summer    [200 200 0]
    10: 'Csc',  # Temperate, dry summer, cold summer    [150 150 0]
    11: 'Cwa',  # Temperate, dry winter, hot summer     [150 255 150]
    12: 'Cwb',  # Temperate, dry winter, warm summer    [100 200 100]
    13: 'Cwc',  # Temperate, dry winter, cold summer    [50 150 50]
    14: 'Cfa',  # Temperate, no dry season, hot summer  [200 255 80]
    15: 'Cfb',  # Temperate, no dry season, warm summer [100 255 80]
    16: 'Cfc',  # Temperate, no dry season, cold summer [50 200 0]
    17: 'Dsa',  # Cold, dry summer, hot summer          [255 0 255]
    18: 'Dsb',  # Cold, dry summer, warm summer         [200 0 200]
    19: 'Dsc',  # Cold, dry summer, cold summer         [150 50 150]
    20: 'Dsd',  # Cold, dry summer, very cold winter    [150 100 150]
    21: 'Dwa',  # Cold, dry winter, hot summer          [170 175 255]
    22: 'Dwb',  # Cold, dry winter, warm summer         [90 120 220]
    23: 'Dwc',  # Cold, dry winter, cold summer         [75 80 180]
    24: 'Dwd',  # Cold, dry winter, very cold winter    [50 0 135]
    25: 'Dfa',  # Cold, no dry season, hot summer       [0 255 255]
    26: 'Dfb',  # Cold, no dry season, warm summer      [55 200 255]
    27: 'Dfc',  # Cold, no dry season, cold summer      [0 125 125]
    28: 'Dfd',  # Cold, no dry season, very cold winter [0 70 95]
    29: 'ET',   # Polar, tundra                         [178 178 178]
    30: 'EF',   # Polar, frost                          [102 102 102]
}

KOPPEN_COLORS = {
    'BSh': 'rgb(245, 165, 0)',
    'BSk': 'rgb(255, 220, 100)',
    'BWh': 'rgb(255, 0, 0)',
    'Cfa': 'rgb(200, 255, 80)',
    'Cfb': 'rgb(100, 255, 80)',
    'Csa': 'rgb(255, 255, 0)',
    'Csb': 'rgb(200, 200, 0)', 
}

COLOR_DEFECTO = "rgb(120, 120, 120)"
 
KOPPEN_DESC = {
    'BSh': 'Árido, estepario, cálido',
    'BSk': 'Árido, estepario, frío',
    'BWh': 'Árido, desértico, cálido',
    'Cfa': 'Templado, sin estación seca, verano caluroso',
    'Cfb': 'Templado, sin estación seca, verano templado',
    'Csa': 'Templado, verano seco, verano caluroso',
    'Csb': 'Templado, verano seco, verano templado',
}

VARS_MEDIDA = [
    "tmed", "tmax", "tmin", "velmedia", "racha", "hrMedia", "prec",
    "presMax", "presMin", "presMedia", "mslp_media", "mslp_max", "mslp_min",
]


# https://rasterio.readthedocs.io/en/latest/index.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
def etiquetar(df_estaciones, path=KOPPEN_TIF,
              col_lon=COL_LON, col_lat=COL_LAT):
    df = df_estaciones.copy()
    lons = df[col_lon].to_numpy(dtype="float64")
    lats = df[col_lat].to_numpy(dtype="float64")
 
    with rasterio.open(path) as src:
        # print(src.crs) # https://epsg.io/4326 en mapa interactivo x=longitud e y=latitud
        # print(src.bounds) # izquierda: -180, derecha: 180, arriba: 90, abajo: -90
        koppen = src.read(1)
        transform = src.transform

    # Foreground son las celdas invalidas, la función da índices de la celda background (válida) más cercana
    invalid = (koppen <= 0)
    idx = distance_transform_edt(invalid, return_distances=False, return_indices=True) 
    koppen_nearest = koppen[tuple(idx)]
    
    # https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html#rasterio.transform.rowcol
    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    rows, cols = np.asarray(rows), np.asarray(cols)
    codes = koppen_nearest[rows, cols].astype("int32")

    # invalid = codes[codes <= 0]
    # print(f"Códigos no establecidos: {len(invalid)}")
    # if len(invalid) > 0:
    #     print(f"Error: {df.loc[codes <= 0]}")
 
    df["koppen_code"] = codes
    df["koppen_class"] = [KOPPEN_LEGEND.get(c, "sin_dato") for c in codes]
    df["koppen_main"] = df["koppen_class"].str[0]
    return df


def mapa_koppen(df_estaciones, zoom=4.5, height=1080, width=1080, inner_size=7, outer_size=9):
    df = df_estaciones.copy()

    df["clima"] = df["koppen_class"].map(
        lambda c: f"{c}:- {KOPPEN_DESC.get(c, 'sin descripción')}"
    )
    col_nombre = "nombre" if "nombre" in df.columns else "indicativo"
    orden = [c for c in KOPPEN_COLORS if c in set(df["koppen_class"])]

    min_lat, max_lat = df['latitud'].min(), df['latitud'].max()
    min_lon, max_lon = df['longitud'].min(), df['longitud'].max()

    centro_lat = (min_lat + max_lat) / 2
    centro_lon = (min_lon + max_lon) / 2

    # scattermap principal
    fig = px.scatter_map(
        df,
        lat="latitud",
        lon="longitud",
        color="koppen_class",
        color_discrete_map=KOPPEN_COLORS,
        category_orders={"koppen_class": orden},
        hover_name=col_nombre,
        hover_data={
            "clima": True,
            "koppen_class": False,
            "latitud": False,
            "longitud": False,
            **({"indicativo": True} if "indicativo" in df.columns else {}),
        },
        zoom=zoom,
        height=height,
        width=width,
        # title="Estaciones meteorológicas por clima (Köppen-Geiger)",
    )

    # tamaño de los puntos de color
    fig.update_traces(marker={"size": inner_size})

    # borde negro simulado, capa de puntos negros mayores por debajo
    fig.add_scattermap(
        lat=df["latitud"], lon=df["longitud"],
        mode="markers",
        marker={"size": outer_size, "color": "black"},
        hoverinfo="skip", showlegend=False,
    )
    fig.data = (fig.data[-1],) + fig.data[:-1] # manda el borde al fondo

    fig.update_layout(
        map_style="carto-positron-nolabels",
        legend_title_text="Clima (Köppen)",
        margin={"r": 5, "t": 5, "l": 5, "b": 5},
        map_center={"lat": centro_lat, "lon": centro_lon},
    )
    return fig