VARIABLES       = ["tmax", "tmin", "prec", "racha"]
VARIABLES_CONT  = ["tmax", "tmin", "prec", "velmedia", "racha", "mslp_max", "mslp_min", "hrMedia"]

FLAGS = [f"extreme_{v}" for v in VARIABLES]
HORIZONTES  = [1, 3, 7]

"""
Esta configuración controla el feature engineering, el modelado y la experimentación 
"""

CONFIG = {
    # ===========================================
    #  Rutas a datos
    # =========================================== 
    "data_path_spain": "../datasets/parquet/LABELED_CURATED_AEMET_KOPPEN.parquet",
    "data_path_clm": "../datasets_clm/main/LABELED_KOPPEN_DATA.parquet",
    "out_dir": "data/processed",
    "out_features_spain": "data/processed/features_df_spain.parquet",
    "out_features_clm": "data/processed/features_df_clm.parquet",
    "reg_path_spain": "data/processed/feature_reg_spain.csv",
    "reg_path_clm": "data/processed/feature_reg_clm.csv",

    # ===========================================
    #  Rutas a resultados
    # =========================================== 
    "results_dir":      "data/results",
    "models_dir":       "data/models",
    "results_path_spain":   "data/results/registro_experimentos_spain.csv",
    "results_path_clm":     "data/results/registro_experimentos_clm.csv",
    "best_path_spain":      "data/results/mejores_por_tipo_spain.csv",
    "best_path_clm":        "data/results/mejores_por_tipo_clm.csv",
    
    # ===========================================
    #  Dominio del problema
    # ===========================================
    "station_col": "indicativo",
    "date_col": "fecha",
    "year_min_spain": 1990,
    "year_min_clm": 2010,
    "ex_target_cols": ["tmax", "tmin", "prec", "racha"],
    "ex_horizons": [1, 3, 7],
    "extreme_tail": {"tmax": "upper", "tmin": "lower", "prec": "upper", "racha": "upper"},

    # ===========================================
    #  Modelos e hiperparámetros
    # ===========================================
    # Los modelos posibles son ["Baseline", "NaiveBayes", "LogReg", "DecisionTree", "LGBM"]
    "models": ["Baseline", "NaiveBayes", "LogReg", "DecisionTree", "LGBM"], 
    "search_iters": {"NaiveBayes": 10, "LogReg": 10, "DecisionTree": 10, "LGBM": 10},
    "primary_metric": "average_precision",
    "n_jobs": 1,

    # ===========================================
    #  Splits temporales
    # ===========================================
    "train_max_year": 2018,
    "val_max_year": 2021,
    # test: 2022+

    # ===========================================
    #  Feature engineering
    # ===========================================
    "ex_lags": [1],

    # Define las columnas normales que pasan al conjunto de datos de entrenamiento
    "cont_cols": ["tmax", "tmin", "prec", "racha", "tmed", "hrMedia", "mslp_max", "mslp_min", "mslp_media", "velmedia", "altitud", "latitud", "longitud"],
    "cat_cols": ["koppen_main", "koppen_class"],

    # Define las columnas sobre las que se aplican retardos y qué retardos se aplican
    "lag_cols": ["tmax", "tmin", "prec", "racha", "mslp_min", "mslp_max"],
    "lag_windows": [1, 3],

    # Define las columnas sobre las que se aplican operaciones de ventana rodante, qué operaciones y con qué ventanas
    "roll_cols": ["tmax", "tmin", "prec", "racha"],
    "roll_windows": [3, 7, 14],
    "roll_ops": ["mean", "max"],

    # Define las columnas sobre las que se aplica EWMA y sobre qué spans
    "ewma_cols": ["tmax", "tmin", "prec", "racha"],
    "ewma_spans": [3, 7, 14],

    # Define las columnas sobre las que se aplica diferencia y con qué spans
    "diff_cols": ["mslp_max", "mslp_min"] + ["tmax", "tmin", "prec", "racha"],
    "diff_spans": [1, 3],

    # NO CAMBIAR: SON NECESARIAS PARA EL PREPROCESAMIENTO
    "aux_cols": ["indicativo", "fecha"],

    # ===========================================
    #  Preprocesamiento
    # ===========================================
    # Define las columnas sobre las que se aplica el escalado estándar con ventana
    "core_scale_cols": ["tmax", "tmin", "prec", "racha", "tmed", "hrMedia", "mslp_max", "mslp_min", "mslp_media", "velmedia"],

    # Define los días adelante y atrás que se consultan para calcular media y desviación estándar en escalado estándar con ventana
    "core_half_window": 2, 

    # Define las columnas que permanecen estáticas en los registros pertenecientes a la misma estación
    "static_scale_cols": ["altitud", "latitud", "longitud"]
}
