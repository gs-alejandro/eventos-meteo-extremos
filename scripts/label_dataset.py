import pandas as pd
from dataset_labeling.labeler import build_daily_thresholds, label_rows, optimize_datatypes
from dataset_processing.processor import reduce_aemet_df_to_clm, match_era5_units, adjust_pressures
import gc

if __name__ == '__main__':
    print('Loading Spain dataframe')
    df = pd.read_parquet('datasets/parquet/CURATED_AEMET.parquet')

    print('Calculating thresholds for Spain')
    thresholds_spain = build_daily_thresholds(df)

    print('Labeling Spain rows')
    df_labeled = label_rows(df, thresholds_spain)
    df_labeled = optimize_datatypes(df_labeled)

    print('Saving Spain dataset')
    df_labeled.to_parquet('datasets/parquet/LABELED_CURATED_AEMET.parquet', index=False)

    print('Cleaning Spain objects')
    del df
    del thresholds_spain
    del df_labeled
    gc.collect()

    print('Loading CLM dataframe')
    df_clm = pd.read_parquet('datasets_clm/main/AEMET_ERA5_daily.parquet')

    print('Calculating CLM thresholds')
    thresholds_clm = build_daily_thresholds(df_clm)

    print('Labeling CLM rows')
    df_clm_labeled = label_rows(df_clm, thresholds_clm)

    print('Saving CLM dataset')
    df_clm_labeled = optimize_datatypes(df_clm_labeled)
    df_clm_labeled.to_parquet(
        'datasets_clm/main/LABELED_AEMET_ERA5.parquet',
        index=False
    )

    print('Cleaning CLM objects')
    del df_clm
    del thresholds_clm
    del df_clm_labeled
    gc.collect()

    print('Done')