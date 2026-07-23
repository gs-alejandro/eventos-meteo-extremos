import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf, pacf

import matplotlib.pyplot as plt
import seaborn as sns

DIRECCION = {"tmax": ">", "prec": ">", "velmedia": ">", "racha": ">", "tmin": "<"}

THRESHOLD_COLS = [
    'tmax_threshold', 
    'tmin_threshold',
    'prec_threshold',
    'velmedia_threshold',
    'racha_threshold',
    'global_tmax_threshold',
    'global_tmin_threshold',
    'global_prec_threshold',
    'global_velmedia_threshold',
    'global_racha_threshold'
]

LOCAL_THRESHOLD_COLS, GLOBAL_THRESHOLD_COLS = THRESHOLD_COLS[:5], THRESHOLD_COLS[5:]

LABEL_COLS = [
    'extreme_tmax',
    'extreme_tmin',
    'extreme_prec',
    'extreme_velmedia',
    'extreme_racha',
    'extreme_tmax_global',
    'extreme_tmin_global',
    'extreme_prec_global',
    'extreme_velmedia_global',
    'extreme_racha_global'
]

LOCAL_LABEL_COLS, GLOBAL_LABEL_COLS = LABEL_COLS[:5], LABEL_COLS[5:]


def _grupos(df, group_var, subset=None):
    if group_var is not None and subset:
        df = df[df[group_var].isin(subset)]
    if group_var is None:
        return [("global", df)]
    return list(df.groupby(group_var, observed=True))


def longitudes_episodios(serie_bool):
    s = serie_bool.fillna(False).astype(int).values
    longs, c = [], 0
    for x in s:
        if x:
            c += 1
        elif c:
            longs.append(c); c = 0
    if c:
        longs.append(c)
    return longs


def extreme_probs(df, vars, group_var=None):
    filas = []
    for val, g in _grupos(df, group_var):
        for v in vars:
            flag = f"extreme_{v}"

            gr = pd.DataFrame({
                "hoy": g.groupby("indicativo")[flag].shift(0),
                "man": g.groupby("indicativo")[flag].shift(-1)
            }).dropna(subset=["man"])

            base  = gr["man"].mean()
            cond  = gr.loc[gr["hoy"] == True, "man"].mean()

            # longitud media de extremos consecutivos (episodios agrupados por estación)
            longs = []
            for _, s in g.groupby("indicativo")[flag]:
                longs += longitudes_episodios(s)
            long_media = np.mean(longs) if longs else 0

            fila = {"variable": v, "P(mañana)": base,
                    "P(mañana|hoy extremo)": cond, "lift": cond / base,
                    "long_media_episodio": long_media}
            if group_var is not None:
                fila[group_var] = val
            filas.append(fila)

    idx = ["variable"] if group_var is None else [group_var, "variable"]
    return pd.DataFrame(filas).set_index(idx)


def target_futuro(serie_flag, n):
    idx = pd.api.indexers.FixedForwardWindowIndexer(window_size=n)
    return serie_flag.astype("float").shift(-1).rolling(window=idx, min_periods=n).max()


def prevalencias_target(df, vars, horizontes, group_var=None):
    filas = []
    for val, g in _grupos(df, group_var):
        for v in vars:
            flag = f"extreme_{v}"
            for n in horizontes:
                tgt = g.groupby("indicativo", group_keys=False)[flag].apply(
                    lambda s: target_futuro(s, n))
                sub = tgt.dropna()
                fila = {"variable": v, "N": n,
                        "prevalencia": sub.mean(), "n_filas": len(sub)}
                if group_var is not None:
                    fila[group_var] = val
                filas.append(fila)
    idx = ["variable", "N"] if group_var is None else [group_var, "variable", "N"]
    return pd.DataFrame(filas).set_index(idx)


def climatologia(df, var, group_var=None):
    filas = []
    for val, g in _grupos(df, group_var):
        pe = (g.groupby(["indicativo", "diaAnio"])[var]
                .agg(p05=lambda s: s.quantile(.05), p50="median",
                     p95=lambda s: s.quantile(.95)))
        clim_val = pe.groupby("diaAnio").mean().reset_index()
        clim_val["grupo"] = val
        filas.append(clim_val)
    return pd.concat(filas, ignore_index=True)


def overlay_climatologia(df, var, group_var=None, banda=False, subset=None, title=True, fsize=(10, 3), line_alpha=1.0, band_alpha=0.1):
    if group_var is not None and subset:
        df = df[df[group_var].isin(subset)]
    clim = climatologia(df, var, group_var)
    orden = clim.groupby("grupo")["p50"].mean().sort_values().index.tolist()
    colores = sns.color_palette("deep", len(orden))
    
    fig, ax = plt.subplots(figsize=fsize)
    for c, val in zip(colores, orden):
        sub = clim[clim["grupo"] == val]
        if banda:
            ax.fill_between(sub["diaAnio"], sub["p05"], sub["p95"], color=c, alpha=0.15)
        ax.plot(sub["diaAnio"], sub["p50"], lw=1.5, color=c, label=val, alpha=line_alpha)
    ax.set_xlabel("Día del año"); ax.grid(alpha=band_alpha)
    por = "" if group_var is None else f" por {group_var}"
    
    if title:
        ax.set_title(f"Climatología de {var} - mediana{por} (media por estación)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


def plot_tendencia(
    df, var, 
    group_var=None,
    xlim=None,
    ylim=None,
    title=True,
    fsize=(10, 4),
    lowess=False,
    ci=95
    ):
    por_est = df.groupby(["indicativo", "anio"])[var].mean().reset_index()

    if group_var is None:
        por_est["grupo"] = "global"
    else:
        mapa = df.drop_duplicates("indicativo").set_index("indicativo")[group_var]
        por_est["grupo"] = por_est["indicativo"].map(mapa)
    anual = por_est.groupby(["grupo", "anio"], observed=True)[var].mean().reset_index()

    g = sns.lmplot(
        data=anual,
        x="anio",
        y=var,
        hue="grupo",
        ci=ci,
        lowess=lowess,
        height=fsize[1],
        aspect=fsize[0] / fsize[1],
        palette="muted",
        scatter_kws={"s": 18, "alpha": 0.4},
        line_kws={"lw": 2},
        legend=group_var is not None,
    )

    ax = g.ax
    if title:
        ax.set_title(f"Tendencia de {var} (medias anuales por estación, train)")
    ax.set_xlabel("año")
    ax.set_ylabel(var)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    g.tight_layout()
    plt.show()


def amplitud_estacional(df, var, group_var=None):
    clim = climatologia(df, var, group_var)
    return (clim.groupby("grupo")["p50"].agg(lambda s: s.max() - s.min())
            .rename("amplitud_anual").sort_values(ascending=False))


def acf_pacf(df, var, group_var=None, lags=31, desestacionalizar=False, subset=None, title=True, ylabel=True, s_count=True):
    if group_var is not None and subset:
        df = df[df[group_var].isin(subset)]
    sufijo = " (anomalías)" if desestacionalizar else " (cruda)"

    def serie_estacion(g_s):
        # serie diaria de una estación
        s = (g_s.groupby("fecha")[var].mean().sort_index()
               .asfreq("D").interpolate(limit=3).dropna())
        if desestacionalizar:
            s = s - s.groupby(s.index.dayofyear).transform("mean")
        return s

    def acf_pacf_medio(g):
        # promedia los coeficientes ACF/PACF y sus errores sobre las estaciones
        acfs, pacfs, errs_acf, errs_pacf = [], [], [], []
        for _, g_s in g.groupby("indicativo", observed=True):
            s = serie_estacion(g_s)
            
            # Coeficientes e intervalos de confianza (de que es ruido) al 95% (alpha=0.05)
            # acf los devuelve centrados en el valor ACF, no como plot_acf
            ac, ac_conf = acf(s, nlags=lags, alpha=0.05) 
            pac, pac_conf = pacf(s, nlags=lags, alpha=0.05, method='ywm') 
            acfs.append(ac)
            pacfs.append(pac)
            
            # La amplitud del error es la distancia desde el valor central al límite
            # confint[:, 0] es el límite inferior, confint[:, 1] el superior
            errs_acf.append(ac - ac_conf[:, 0])
            errs_pacf.append(pac - pac_conf[:, 0])
            
        if not acfs:
            return None
        return np.mean(acfs, 0), np.mean(pacfs, 0), np.mean(errs_acf, 0), np.mean(errs_pacf, 0), len(acfs)

    # Un resultado por grupo
    resultados = {}
    if group_var is None:
        r = acf_pacf_medio(df)
        if r: resultados["global"] = r
    else:
        for val, g in df.groupby(group_var, observed=True):
            r = acf_pacf_medio(g)
            if r: resultados[val] = r

    var_vals = sorted(resultados.keys())
    n = len(var_vals)

    fig, axes = plt.subplots(n, 2, figsize=(10, 2.5*n), squeeze=False)
    
    for i, val in enumerate(var_vals):
        ac, pac, err_ac, err_pac, n_est = resultados[val]
        for ax, coef, err in zip([axes[i, 0], axes[i, 1]], [ac, pac], [err_ac, err_pac]):
            lags_range = range(1, len(coef))
            
            ax.fill_between(lags_range, -err[1:], err[1:], alpha=0.12)
            ax.stem(lags_range, coef[1:], basefmt=" ")
            ax.axhline(0, color="k", lw=0.8)
            
            # ax.set_xlim(left=1.0)
            
        axes[i, 0].set_title("ACF" if i == 0 else "")
        axes[i, 1].set_title("PACF" if i == 0 else "")
        if ylabel:
            if s_count:
                axes[i, 0].set_ylabel(f"{val}\n({n_est} est.)", fontsize=9)
            else:
                axes[i, 0].set_ylabel(f"{val}", fontsize=9)

    titulo_por = f" por {group_var}" if group_var is not None else ""
    if title:
        fig.suptitle(f"ACF / PACF de {var}{sufijo}{titulo_por}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.show()


def serie_vs_umbral(df, var, group_var=None, inicio=pd.to_datetime('2010-01-01'),
                    final=pd.to_datetime('2018-12-31'), subset=None, ncol=2, xsize=7, sharey=True):
    thr = f"{var}_threshold"
    signo = DIRECCION[var]

    base = df[(df["fecha"] >= inicio) & (df["fecha"] <= final)
              & (df["split"].isin(["train", "val"]))]
    if group_var is not None and subset:
        base = base[base[group_var].isin(subset)]

    grupos = _grupos(base, group_var)
    var_vals = [val for val, _ in grupos]
    n = len(var_vals)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(xsize*ncol, 3*nrow),
                             sharex=True, squeeze=False, sharey=sharey)
    axes = axes.ravel()
    for ax, (val, g) in zip(axes, grupos):
        diaria = g.groupby("fecha")[[var, thr]].mean() # media diaria del grupo (visual)
        ax.plot(diaria.index, diaria[var], lw=.5, alpha=.6, label=var)
        ax.plot(diaria.index, diaria[thr], color="red", lw=1, label="umbral (<=2018)")
        ax.set_title(str(val), fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    por = "" if group_var is None else f" por {group_var}"
    fig.suptitle(f"{var} vs umbral fijo{por}", fontweight="bold")
    fig.supxlabel("fecha"); fig.supylabel(var)

    plt.tight_layout(); plt.show()

    # tasa de superación a nivel estación-día (cada registro vs su propio umbral)
    for val, g in grupos:
        print(f"{val}:")
        for sp in [
            "train",
            # "val"
            ]:
            s = g[g["split"] == sp]
            if not s.empty:
                cruces = (s[var] > s[thr]) if signo == ">" else (s[var] < s[thr])
                print(f"  {sp}: tasa de superación del umbral = {cruces.mean():.2%}")
            else:
                print(f"  {sp}: No hay datos para este split.")

