import joblib
import numpy as np
import pandas as pd
from time import time
import scipy.stats as st

import os
import json
from config import CONFIG

from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    accuracy_score
)

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base               import clone
from sklearn.pipeline           import Pipeline
from sklearn.impute             import SimpleImputer
from sklearn.compose            import ColumnTransformer
from sklearn.model_selection    import PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing      import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.naive_bayes        import GaussianNB
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier

from lightgbm                   import LGBMClassifier

SEED = 42
np.random.seed(SEED)


# Función personalizada y simplificada inspirada en StandardScaler de scikit-learn:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://github.com/scikit-learn/scikit-learn/blob/cc50648cc/sklearn/preprocessing/_data.py#L742
# y por la web: 
# https://towardsdatascience.com/customizing-scikit-learn-pipelines-write-your-own-transformer-fdaaefc5e5d7/
class GroupedTimeWindowStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, date_col, window, cols):
        self.group_col = group_col
        self.date_col = date_col
        self.cols = cols
        self.window = window


    def fit(self, X, y=None):
        offsets = np.tile(np.arange(-self.window, self.window+1), X.shape[0])
        offsets = pd.to_timedelta(offsets, unit="D")
        
        # Duplica el df por tantos registros como haya en la ventana y asigna un centro en el que se agregan los registros de cada ventana
        repeated = X.copy().loc[X.index.repeat((2*self.window)+1), [self.group_col] + self.cols + [self.date_col]]
        repeated["__center"] = pd.to_datetime(repeated[self.date_col]) + offsets
        repeated["__c_month"] = repeated["__center"].dt.month
        repeated["__c_day"] = repeated["__center"].dt.day
        
        repeated = repeated.drop(columns=[self.date_col, "__center"])

        self.mean_df_ = repeated.groupby([self.group_col, "__c_month", "__c_day"]).mean().add_suffix("_mean")
        self.std_df_ = repeated.groupby([self.group_col, "__c_month", "__c_day"]).std().add_suffix("_std")
        self.std_df_ = self.std_df_.fillna(1.0).replace(0.0, 1.0) # Evitar división por 0 en variables que no cambien
        return self
    

    def transform(self, X):
        X = X.copy()
        X["__center"] = pd.to_datetime(X[self.date_col])
        X["__c_month"] = X["__center"].dt.month
        X["__c_day"] = X["__center"].dt.day 
        X = X.merge(self.mean_df_.reset_index(), on=[self.group_col, "__c_month", "__c_day"], how="left")
        X = X.merge(self.std_df_.reset_index(), on=[self.group_col, "__c_month", "__c_day"], how="left")
        
        mean_cols = [col + "_mean" for col in self.cols]
        std_cols = [col + "_std" for col in self.cols]

        # Estandarizado (sin el to_numpy sale solo NaNs)
        X[self.cols] -= X[mean_cols].to_numpy()
        X[self.cols] /= X[std_cols].to_numpy()

        # Limpieza de columnas auxiliares
        cols_to_drop = ["__center", "__c_month", "__c_day"] + mean_cols + std_cols
        X = X.drop(columns=cols_to_drop)
        return X


# Función personalizada y simplificada inspirada en MinMaxScaler de scikit-learn (rango [-1, 1]):
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# https://github.com/scikit-learn/scikit-learn/blob/cc50648cc/sklearn/preprocessing/_data.py#L305
class GroupedMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, cols, feature_range=(-1, 1)):
        self.group_col = group_col
        self.cols = cols
        self.feature_range = feature_range
    
    def fit(self, X, y=None):
        self.max_df_ = X.groupby(self.group_col)[self.cols].max().add_suffix("_max")
        self.min_df_ = X.groupby(self.group_col)[self.cols].min().add_suffix("_min")
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.merge(self.max_df_.reset_index(), on=self.group_col, how="left")
        X = X.merge(self.min_df_.reset_index(), on=self.group_col, how="left")

        # Escalado min max
        min_cols = [col + "_min" for col in self.cols]
        max_cols = [col + "_max" for col in self.cols]
        _range = (X[max_cols].to_numpy() - X[min_cols].to_numpy())
        _scale = (self.feature_range[1] - self.feature_range[0]) / np.where(_range == 0, 1.0, _range)
        _min = self.feature_range[0] - X[min_cols].to_numpy() * _scale

        X[self.cols] *= _scale
        X[self.cols] += _min
        cols_to_drop = min_cols + max_cols
        X = X.drop(columns=cols_to_drop)
        return X


class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None):
        self.fitted_ = True # si no, da error NotFitted al ejecutar el pipeline
        return self
    def transform(self, X): return X.drop(columns=[col for col in self.cols if col in X.columns])


def load_data(path, reg_path):
    df = pd.read_parquet(path)
    df = df.sort_values(["indicativo", "fecha"]).reset_index(drop=True)
    print(f"Datos cargados desde {path}: {df.shape}")
    print(f"    Número de estaciones de datos: {df['indicativo'].nunique()}")
    reg = pd.read_csv(reg_path)
    print(f"Registro de características cargado desde {reg_path}")
    print(f"    Longitud vista completa: {len(reg[reg['view'] == 'curated'])}")
    print(f"    Longitud vista mínima: {len(reg[reg['view'] == 'minimal'])}")
    return df, reg


def feature_view(reg, view):
    cols = [col for col in reg.loc[reg["view"] == view, "var"]]
    return cols


def column_types(df, cols):
    num = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    cat = [col for col in cols if isinstance(df[col].dtype, pd.CategoricalDtype)]
    return num, cat


def keep_mask(df, N):
    keep = np.ones(len(df), dtype=bool)
    splits = df["split"]
    g = df.groupby("indicativo")["split"]
    for n in range(1, N+1):
        keep &= (splits == g.shift(-n)).to_numpy()
    return keep


def load_experiment(df, var, N, view, reg):
    cols = feature_view(reg, view)
    target = f"target_{var}_{N}"
    keep = keep_mask(df, N)
    sub = df[keep]
    out = {"cols": cols, "target": target, "N": N}
    for split in ["train", "val", "test"]:
        sub_split = sub[sub["split"] == split]
        out[split] = {
            "X": sub_split[cols], "y": sub_split[target].astype("int8")
        }
    return out


def conditional_probability_baseline(df, var, N, reg):
    """
    El baselie se basa en calcular la probabilidad condicional de positivo en la clase objetivo según
    la condición de extremo del día actual. Devuelve un array con las probabilidades aprendidas en train:
    1. P(target|extremo) si hay extremo el dia actual
    2. P(target|no extremo) si no hay extremo
    Tener probabilidades como baseline es necesario para usar métricas como average precision o ROC-AUC,
    además, al optimizar el umbral según el f1 score se comporta como persistencia normal (se predice
    target positivo si hay extremo hoy y negativo en caso contrario).
    """
    e = load_experiment(df, var, N, "minimal", reg)
    ex_today = f"{var}_ex_today"
    X, y = e["train"]["X"], e["train"]["y"]
    p1 = y.loc[X[ex_today] == 1].mean()
    p0 = y.loc[X[ex_today] == 0].mean()
    out = {}
    for split in ["train", "val", "test"]:
        cond = e[split]["X"][ex_today] == 1
        out[split] = np.where(cond, p1, p0)
    return out


def _tune_threshold(y_true, y_score):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = np.asarray([f1_score(y_true, (y_score >= threshold).astype(int)) for threshold in thresholds])
    best_threshold = thresholds[np.argmax(f1s)]
    return best_threshold


def evaluate(y_true, y_score, threshold=None):
    if threshold is None: threshold = _tune_threshold(y_true, y_score)
    out                 = {}
    out["prevalence"]   = float(np.mean(y_true))
    out["AP"]           = average_precision_score(y_true, y_score)
    out["ROC_AUC"]      = roc_auc_score(y_true, y_score)
    out["threshold"]    = threshold
    y_pred              = (y_score >= threshold).astype(int)
    out["F1"]           = f1_score(y_true, y_pred)
    out["precision"]    = precision_score(y_true, y_pred)
    out["recall"]       = recall_score(y_true, y_pred)
    out["accuracy"]     = accuracy_score(y_true, y_pred)
    out["confusion"]    = confusion_matrix(y_true, y_pred)
    return out


def make_scalers(cols, core_only):
    scaler_steps = []
    # El escarado nucleo agrupa por estación y ventana temporal
    # La intención es escalar la variación del tiempo "normal" para la estación en el momento
    # Para mayor generalización se podría cambiar a agrupado por clase climática
    core_scaler = GroupedTimeWindowStandardScaler("indicativo", "fecha", CONFIG["core_half_window"], CONFIG["core_scale_cols"])
    scaler_steps.append(("core_scaler", core_scaler))
    if not core_only: 
        # Minmax agrupado para escalar a el rango de la estación
        # Evita que valores extremos de algunas estaciones queden enmascarados 
        other = [col for col in cols if col not in CONFIG["core_scale_cols"] and col not in CONFIG["static_scale_cols"]]
        grouped_minmax = GroupedMinMaxScaler("indicativo", other)
        scaler_steps.append(("grouped_minmax_scaler", grouped_minmax))
        # Minmax normal para features estáticas en registros de la misma estación
        # Se escala a rango del df para comparar entre estaciones
        minmax_transformer = ColumnTransformer(
            [("static_minmax_scaler", MinMaxScaler(feature_range=(-1, 1)), CONFIG["static_scale_cols"])], 
            verbose_feature_names_out=False, remainder="passthrough"
        ).set_output(transform="pandas")
        scaler_steps.append(("static_minmax_transformer", minmax_transformer))
    return scaler_steps


def make_preprocessor(df, cols, passthrough, core_only):
    num, _ = column_types(df, cols)
    steps = []
    if passthrough:
        steps.extend(make_scalers(num, core_only))
    else:
        impute_encode_transformer = ColumnTransformer(
            [("impute", SimpleImputer(strategy="median", add_indicator=False), num),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CONFIG["cat_cols"])],
            verbose_feature_names_out=False, remainder="passthrough"
        ).set_output(transform="pandas")
        steps.append(("impute_encode", impute_encode_transformer))
        steps.extend(make_scalers(num, core_only))
    steps.append(("drop_aux", DropCols(["indicativo", "fecha"])))
    pipeline = Pipeline(steps)
    return pipeline


def build_model(name):
    if name == "NaiveBayes":
        pipe = Pipeline([
            ("classifier", GaussianNB())
        ])
        grid = {"classifier__var_smoothing": st.loguniform(1e-9, 1e-1)}
        return pipe, grid
    
    elif name == "LogReg":
        pipe = Pipeline([
            ("classifier", LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=SEED, max_iter=4000))
        ])
        grid = {
            "classifier__C": st.loguniform(1e-3, 1e-1),
        }
        return pipe, grid
    
    elif name == "DecisionTree":
        pipe = Pipeline([
            ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))
        ])
        grid = {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [5, 10, 15],
            "classifier__min_samples_leaf": [100, 200],
            "classifier__max_features": [None, "sqrt"],
        }
        return pipe, grid
    
    elif name == "LGBM":
        pipe = Pipeline([
            ("classifier", LGBMClassifier(n_estimators=800, random_state=SEED, n_jobs=1, verbose=-1, is_unbalance=True, subsample_freq=1))
        ])
        grid = {
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__num_leaves": [32, 64],
            "classifier__lambda_l1": [0.0, 0.5, 1.0],
            "classifier__lambda_l2": [0.0, 0.5, 1.0],
            "classifier__bagging_fraction": [0.5, 0.75, 1.0],
            "classifier__feature_fraction": [0.5, 0.75, 1.0]
        }
        return pipe, grid


def run_experiment(df, var, N, model_name, view, reg, region, verbose=True):
    t0 = time()
    e = load_experiment(df, var, N, view=view, reg=reg)
    X_train, X_val, X_test = e["train"]["X"], e["val"]["X"], e["test"]["X"]
    y_train, y_val, y_test = e["train"]["y"], e["val"]["y"], e["test"]["y"]
    best_params, model_path = {}, None
    cols = X_train.columns

    if model_name == "Baseline":
        b = conditional_probability_baseline(df, var, N, reg=reg)
        y_score_train  = b["train"]
        y_score_val    = b["val"]
        y_score_test   = b["test"]

    else:
        # Se hace preprocesamiento una vez por experimento, reduce tiempo de ejecución y no introduce 
        # fuga de datos al tener folds train, val, test fijos
        passthrough = False
        core_only = True
        if model_name == "LGBM": passthrough = True
        if model_name == "LogReg": core_only = False
        preprocessor = make_preprocessor(df, cols, passthrough, core_only)
        X_train = preprocessor.fit_transform(X_train)
        X_val   = preprocessor.transform(X_val)
        X_test  = preprocessor.transform(X_test)
        pipe, grid = build_model(model_name)
        X_train_val = pd.concat([X_train, X_val], ignore_index=True) # Salen del preprocesamiento con índices reseteados
        y_train_val = pd.concat([y_train, y_val], ignore_index=True)
        split_idx = np.zeros(X_train_val.shape[0])
        split_idx[:X_train.shape[0]] = -1 # ignorados en val en PredefinedSplit
        
        # print(X_train_val.isna().sum().to_dict())
        n_iters = CONFIG["search_iters"][model_name]
        t_s = time()
        search = RandomizedSearchCV(
            pipe, grid, n_iter=n_iters, scoring=CONFIG["primary_metric"],
            cv=PredefinedSplit(split_idx), random_state=SEED, refit=False, n_jobs=CONFIG["n_jobs"],
            error_score="raise", verbose=3
        )
        search.fit(X_train_val, y_train_val)
        best_params = search.best_params_
        if verbose:
            print(f"    Búsqueda de {n_iters} sobre {len(X_train)} filas en {time() - t_s:.4f} segundos"
                      f" | mejor AP en val: {search.best_score_:.4f}", flush=True)
        # Entrenamiento del mejor modelo en train para métricas
        m_train = clone(pipe).set_params(**best_params)
        m_train.fit(X_train, y_train)
        y_score_train = m_train.predict_proba(X_train)[:, 1]
        y_score_val = m_train.predict_proba(X_val)[:, 1]
        # Reentrenamiento del mejor modelo en todo, es como refit=True en RandomizedSearchCV
        m_train_val = clone(pipe).set_params(**best_params)
        m_train_val.fit(X_train_val, y_train_val)
        y_score_test = m_train_val.predict_proba(X_test)[:, 1]

        os.makedirs(CONFIG["models_dir"], exist_ok=True)
        tag = f"_{view}_{region}"
        model_path = os.path.join(CONFIG["models_dir"], f"{model_name}_{var}_N{N}{tag}.joblib")
        joblib.dump({
                "model": m_train_val,
                "cols": cols, "var": var, "N": N,
                "model_name": model_name,
        }, model_path, compress=3)

    threshold   = _tune_threshold(y_val, np.asarray(y_score_val))
    res_train   = evaluate(y_train, y_score_train, threshold)
    res_val     = evaluate(y_val, y_score_val, threshold)
    res_test    = evaluate(y_test, y_score_test, threshold)
    row = {
        "variable": var, "N": N, "model_name": model_name, "view": view,
        "n_features": len(cols),
        "prevalence_train": res_train["prevalence"],
        "prevalence_val": res_val["prevalence"],
        "prevalence_test": res_test["prevalence"],
        "AP_train": res_train["AP"],    "ROC_AUC_train": res_train["ROC_AUC"],
        "AP_val": res_val["AP"],        "ROC_AUC_val": res_val["ROC_AUC"],
        "AP_test": res_test["AP"],      "ROC_AUC_test": res_test["ROC_AUC"],
        "precision_train": res_train["precision"],  "recall_train": res_train["recall"],
        "accuracy_train": res_train["accuracy"],    "f1_train": res_train["F1"],
        "precision_val": res_val["precision"],      "recall_val": res_val["recall"],
        "accuracy_val": res_val["accuracy"],        "f1_val": res_val["F1"],
        "precision_test": res_test["precision"],    "recall_test": res_test["recall"],
        "accuracy_test": res_test["accuracy"],      "f1_test": res_test["F1"],
        "best_params": json.dumps(best_params),     "model_path": model_path,
        "s": round(time() - t0, 1)
    }

    preds = {
        "variable": var, "N": N, "model_name": model_name, "view": view,
        "y": np.asarray(y_test, dtype="int8"), "p": np.asarray(y_score_test, dtype="float32"),
        "model_path": model_path,
        "threshold": threshold,
        "cols": list(cols)
    }

    if verbose:
        print(f"[{var}, N={N}, {model_name}]")
        for split in ["train", "val", "test"]:
            print(f"    Métricas en {split}")
            print(
                f"        AP={row[f'AP_{split}']:.3f}"
                f" | ROC_AUC={row[f'ROC_AUC_{split}']:.3f}"
            )
            print(
                f"        prev={row[f'prevalence_{split}']:.3f}"
                f" | P={row[f'precision_{split}']:.3f}"
                f" | R={row[f'recall_{split}']:.3f}"
                f" | F1={row[f'f1_{split}']:.3f}"
                f" | acc={row[f'accuracy_{split}']:.3f}"
            )
    
    return row, preds


def run_all_experiments(df, view, reg, region, verbose=True):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    EXPERIMENTS = [
        (v, N, m) for v in CONFIG["ex_target_cols"] for N in CONFIG["ex_horizons"] for m in CONFIG["models"]
    ]
    TOTAL = len(EXPERIMENTS)
    print(f"Ejecutando {TOTAL} experimentos "
          f"({len(CONFIG['ex_target_cols'])} vars x {len(CONFIG['ex_horizons'])} N x {len(CONFIG['models'])} modelos)")
    print(f"Número de iteraciones por modelo: {CONFIG['search_iters']} | jobs paralelos: {CONFIG['n_jobs']}\n")

    RESULTS, PREDS = [], []
    t_all, times = time(), []
    for k, (var, N, model) in enumerate(EXPERIMENTS, 1):
        te = time()
        print(f"[{k:2d}/{TOTAL}] {var} N={N} {model}", flush=True)
        row, preds = run_experiment(df, var, N, model, view, reg, region, verbose=verbose)
        RESULTS.append(row)
        PREDS.append(preds)
        times.append(time() - te)
        avg = float(np.mean(times))
        rem = (TOTAL - k) * avg
        print(f"    Hecho en {times[-1]:.0f}s | transcurrido {(time()-t_all)/60:.1f} min"
              f" | tiempo restante estimado ~{rem/60:.1f} min ({TOTAL-k} modelos restantes)\n", flush=True)

    results_path = f"{CONFIG['results_dir']}/reg_experiments_{view}_{region}.csv"
    results_df = pd.DataFrame(RESULTS)
    results_df.to_csv(results_path, index=False)
    predictions_path = f"{CONFIG['results_dir']}/reg_predictions_{view}_{region}.parquet"
    predictions_df = pd.DataFrame(PREDS)
    predictions_df.to_parquet(predictions_path, index=False)
    print(f"TOTAL: {len(RESULTS)} experimentos en {(time()-t_all)/60:.1f} min")
    print(f"Registro guardado en {results_path}")
    return results_df
