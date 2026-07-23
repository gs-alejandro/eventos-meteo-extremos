import numpy as np
import pandas as pd
from config import CONFIG

"""
NOTA IMPORTANTE:
Las transformaciones que miran valores pasados (shift, ewma, ...) se hacen por registros, no por fechas
    - Se consideran como operaciones por días porque los registros se toman con frecuencia diaria
    - Se usan porque se comprobó anteriormente (libreta 3) que hay muy pocos huecos en comparación con el tamaño
        del conjunto de datos de España (en CLM no hay huecos por la imputación realizada)
    - Dependen de registros ordenados por indicativo y luego fecha
Los métodos de pandas en objetos GroupBy operan solo con registros dentro de cada grupo, no se meten de otras estaciones
"""


def load_data(path):
    df = pd.read_parquet(path)
    print(f"Cargados datos desde {path}:")
    print(f"    {df.shape[0]} registros de {df['indicativo'].nunique()} estaciones")
    df = df.sort_values(["indicativo", "fecha"]).reset_index(drop=True)
    return df


def register(reg, var, view):
    # print(f"    Registrado en {view}: {var}")
    reg.append({
        "var": var,
        "view": view
    })


def impute_flag_nans(df):
    for flag in [f"extreme_{col}" for col in CONFIG["ex_target_cols"]]:
        num = df[flag].isna().sum()
        df[flag] = df[flag].fillna(False).astype("int8")
        print(f"Imputado False en {num} valores perdidos de {flag}")
    return df


def assign_split(df):
    def _split(anio):
        if anio <= CONFIG["train_max_year"]:    return "train"
        elif anio <= CONFIG["val_max_year"]:    return "val"
        else:                                   return "test"

    df["split"] = df["anio"].map(_split)
    print(
        df.groupby("split", observed=True)
        .agg(filas=("anio", "size"), anio_min=("anio", "min"), anio_max=("anio", "max"))
        .sort_values("anio_min")
    )
    return df


def _build_target(df, var, N):
    flag = f"extreme_{var}"
    col = f"target_{var}_{N}"
    if col not in df.columns:
        g = df.groupby("indicativo")[flag]
        df[col] = False
        for n in range(1, N+1):
            df[col] |= g.shift(-n).fillna(False) # NaN tiene truth value True
        df[col] = df[col].astype("int8")
    return df


def build_target(df):
    for col in CONFIG["ex_target_cols"]:
        for horizon in CONFIG["ex_horizons"]:
            df = _build_target(df, col, horizon)
    return df


# https://pandas.pydata.org/docs/dev/reference/api/pandas.api.typing.SeriesGroupBy.shift.html
def _lags(df, vars, ns, reg, view):
    g = df.groupby("indicativo")
    for var in vars:
        for n in ns:
            col = f"{var}_lag_{n}"
            if col not in df.columns:
                df[col] = g[var].shift(n)
            register(reg, col, view)
    return df


# https://pandas.pydata.org/docs/dev/reference/api/pandas.api.typing.SeriesGroupBy.ewm.html
def _ewm(df, vars, ns, reg, view):
    g = df.groupby("indicativo")
    for var in vars:
        for n in ns:
            col = f"{var}_ewma_{n}"
            if col not in df.columns:
                df[col] = g[var].ewm(span=n).mean().reset_index(level=0, drop=True)
            register(reg, col, view)
    return df


# https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.api.typing.DataFrameGroupBy.rolling.html
def _roll(df, vars, ns, ops, reg, view):
    g = df.groupby("indicativo")
    for var in vars:
        for op in ops:
            for n in ns:
                col = f"{var}_roll_{n}_{op}"
                if col not in df.columns:
                    df[col] = g[var].rolling(window=n).agg(op).reset_index(level=0, drop=True)
                register(reg, col, view)
    return df


# https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.api.typing.SeriesGroupBy.diff.html
def _diff(df, vars, ns, reg, view):
    g = df.groupby("indicativo")
    for var in vars:
        for n in ns:
            col = f"{var}_diff_{n}"
            if col not in df.columns:
                df[col] = g[var].diff(n)
            register(reg, col, view)
    return df


# Codificación 2D del ciclo orbital de la tierra (sacado del día del año)
# https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#trigonometric-features
def _cos_sin(df, reg, view):
    col_sin = "sin_doy"
    col_cos = "cos_doy"
    if col_sin not in df.columns or col_cos not in df.columns:
        angle = df["diaAnio"] / 365.25 * 2 * np.pi
        df[col_sin], df[col_cos] = np.sin(angle), np.cos(angle)
    register(reg, col_sin, view)
    register(reg, col_cos, view)
    return df


def _extreme_today(df, vars, reg, view):
    for var in vars:
        flag = f"extreme_{var}"
        col = f"{var}_ex_today"
        if col not in df.columns:
            df[col] = df[flag]
        register(reg, col, view)
    return df


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.typing.SeriesGroupBy.ffill.html
def _extreme_history(df, vars, ns, reg, view):
    for var in vars:
        flag = f"extreme_{var}"
        aux = f"_aux_{flag}"
        col = f"days_since_{var}_ex"
        # Se guardan los índices de los días extremos, en el resto se pone NA 
        if col not in df.columns:
            df[aux] = df.index
            df.loc[~df[flag].astype(bool), aux] = pd.NA 

    g = df.groupby("indicativo")
    for var in vars:
        flag = f"extreme_{var}"
        aux = f"_aux_{flag}"
        # DÍAS DESDE EL ÚLTIMO EXTREMO
        col = f"days_since_{var}_ex"
        if col not in df.columns:
            # Se rellenan los NA con el índice del último extremo (forward fill) por grupo
            # se resta el índice del último extremo al índice actual y el resultado son registros
            # desde el último extremo (vease nota de arriba)   
            df[col] = df.index - g[aux].ffill()
        df = df.drop(columns=aux)
        register(reg, col, view)

        df = _lags(df, [flag], ns, reg, view)
    return df


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.typing.SeriesGroupBy.ffill.html
def _since_prec(df, reg, view):
    var = "prec"
    aux = f"_aux_{var}"
    col = f"days_since_{var}"
    # Mismo proceso que para días desde extremo pero con días en los que precipitación no sea 0
    if col not in df.columns:
        df[aux] = df.index
        df.loc[~df[var].astype(bool), aux] = pd.NA
        g = df.groupby("indicativo")
        df[col] = df.index - g[aux].ffill()
        df = df.drop(columns=aux)
    register(reg, col, view)
    return df


# suma de precipitación en [n1, n2, ...] días
def _sum_prec(df, ns, reg, view):
    df = _roll(df, ["prec"], ns, ["sum"], reg, view)
    return df


# Número negativo = se acerca al umbral, número positivo = lo sobrepasa
def add_dist_to_ex(df, reg, views):
    df = df.copy()
    for view in views:
        for var in CONFIG["ex_target_cols"]:
            col = f"{var}_dist_to_ex"
            if col not in df.columns:
                if CONFIG["extreme_tail"][var] == "lower":
                    df[col] = df[f"{var}_threshold"] - df[var] 
                else: 
                    df[col] = df[var] - df[f"{var}_threshold"]
            register(reg, col, view)
        print(f"{view}: añadidas distancias a extremos de {CONFIG["ex_target_cols"]}")
    return df


# Solo se registran, ya están en el df
def add_cont_vars(df, reg, views):
    df = df.copy()
    for view in views:
        for col in CONFIG["cont_cols"]:
            register(reg, col, view)
        print(f"{view}: añadidas {CONFIG['cont_cols']} (continuas)")
    return df


# Solo se registran, ya están en el df
def add_cat_vars(df, reg, views):
    df = df.copy()
    for view in views:
        for col in CONFIG["cat_cols"]:
            register(reg, col, view)
        print(f"{view}: añadidas {CONFIG['cat_cols']} (categóricas)")
    return df


def add_time_vars(df, reg, views):
    df = df.copy()
    for view in views:
        df = _cos_sin(df, reg, view)
        register(reg, "anio", view)
        print(f"{view}: añadidas variables temporales")
    return df


def add_lags(df, reg, views):
    df = df.copy()
    for view in views:
        df = _lags(df, CONFIG["lag_cols"], CONFIG["lag_windows"], reg, view)
        print(f"{view}: añadidos lags {CONFIG['lag_windows']} de {CONFIG['lag_cols']}")
    return df


def add_ewma(df, reg, views):
    df = df.copy()
    for view in views:
        df = _ewm(df, CONFIG["ewma_cols"], CONFIG["ewma_spans"], reg, view)
        print(f"{view}: añadidas EWMA con spans {CONFIG['ewma_spans']} de {CONFIG['ewma_cols']}")
    return df


def add_diffs(df, reg, views):
    df = df.copy()
    for view in views:
        df = _diff(df, CONFIG["diff_cols"], CONFIG["diff_spans"], reg, view)
        print(f"{view}: añadidas diferencias con spans {CONFIG['diff_spans']} de {CONFIG['diff_cols']}")
    return df


def add_rolls(df, reg, views):
    df = df.copy()
    for view in views:
        df = _roll(df, CONFIG["roll_cols"], CONFIG["roll_windows"], CONFIG["roll_ops"], reg, view)
        print(f"{view}: añadidas operaciones {CONFIG['roll_ops']} sobre ventanas {CONFIG['roll_windows']} de {CONFIG['roll_cols']}")
    return df


def add_ex_today(df, reg, views):
    df = df.copy()
    for view in views:
        df = _extreme_today(df, CONFIG["ex_target_cols"], reg, view)
        print(f"{view}: añadidos indicadores de extremos de {CONFIG["ex_target_cols"]}")
    return df


def add_ex_history(df, reg, views):
    df = df.copy()
    for view in views:
        df = _extreme_history(df, CONFIG["ex_target_cols"], CONFIG["ex_lags"], reg, view)
        print(f"{view}: añadidos historiales de extremos de {CONFIG["ex_target_cols"]}")
    return df


def add_prec_vars(df, reg, views):
    df = df.copy()
    for view in views:
        df = _sum_prec(df, CONFIG["roll_windows"], reg, view)
        df = _since_prec(df, reg, view)
        print(f"{view}: añadidas variables de precipitación")
    return df


def add_aux(df, reg, views):
    for view in views:
        for var in CONFIG["aux_cols"]:
            register(reg, var, view)
        print(f"{view}: añadidas variables auxiliares")
    return df

def save_data(df, data_path, reg, reg_path):
    df.to_parquet(data_path, index=False)
    print(f"Datos guardados en {data_path}")
    print(f"    Tamaño de datos: {df.shape}")
    reg.sort_values(["view", "var"]).to_csv(reg_path, index=False)
    print(f"Registro de características guardado en {reg_path}")
    print(f"    Longitud vista completa: {len(reg[reg['view']=='curated'])}")
    print(f"    Longitud vista mínima: {len(reg[reg['view']=='minimal'])}")

