import pandas as pd
import numpy as np
import seaborn as sns
import gc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import plotly.express as px
    

def plot_na_values(df, fsize=(16, 9), title='Proporción de valores perdidos por variable', orient='h', xlim=(0, 1)):
    '''
    Plots proportion of NA values per column given a DataFrame
    '''
    missing = (df.isna().sum() / len(df)).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print('No hay valores NA en ninguna columna')
        return
    plt.figure(figsize=fsize)
    ax = sns.barplot(x=missing.values, y=missing.index, orient=orient)
    ax.bar_label(
        ax.containers[0],
        labels=[f'{v*100:.1f}%' for v in missing.values],
        padding=3
    )
    plt.title(title)
    plt.xlabel('Proporción')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.xlim(xlim)
    plt.show()


def plot_na_values_multiple(*datasets, dataset_names=None, dodge=True, horizontal=True, 
                            title=None, hide_zeros=True, fsize=(16, 9), legend_title=''):
    '''
    Performs same function as plot_na_values but for several dataframes with the same columns
    '''
    if not datasets:
        print('No dataframes provided')
        return

    if dataset_names is None or len(dataset_names) != len(datasets):
        if dataset_names is not None:
            print('Dataset names length does not match amount of dataframes, using default names')
        dataset_names = [f'Dataset_{i+1}' for i in range(len(datasets))]

    if title is None:
        title = 'Proporción de valores perdidos por variable y dataset'

    all_missing = []
    for name, df in zip(dataset_names, datasets):
        missing = (df.isna().sum() / len(df)).sort_values(ascending=False)\
            .reset_index(name='NaN_proportion')
        missing['dataset'] = name
        if hide_zeros:
            missing = missing[missing['NaN_proportion'] > 0]
        all_missing.append(missing)
    combined = pd.concat(all_missing, ignore_index=True)

    if combined.empty:
        print('No NA values')
        return

    plt.figure(figsize=fsize)
    if horizontal:
        ax = sns.barplot(data=combined, x='NaN_proportion', y='index',
                         hue='dataset', orient='h', dodge=dodge)
        plt.xlabel('Proporción'); plt.ylabel('Variable')
    else:
        ax = sns.barplot(data=combined, x='index', y='NaN_proportion',
                         hue='dataset', orient='v', dodge=dodge)
        plt.xticks(rotation=90)
        plt.xlabel('Variable'); plt.ylabel('Proporción')

    # Etiqueta de porcentaje por cada grupo de barras (dataset)
    for container in ax.containers:
        ax.bar_label(container,
                     labels=[f'{v*100:.1f}%' if pd.notna(v) and v > 0 else ''
                             for v in container.datavalues],
                     padding=3, fontsize=8)

    plt.title(title)
    plt.legend(title=legend_title)
    plt.tight_layout()
    plt.show()


def plot_na_distributions(by_id, cols, fsize=(16,9), ylim=(0, 1.02)):
    '''
    Takes df gof NaN proportions grouped by station id, 
    sorts by first column and plots NaN proportions for each station (each col is a line)
    '''
    plt.figure(figsize=fsize)

    by_id = by_id.copy()
    by_id = by_id.sort_values(by=cols[0]).reset_index()
    by_id['percentil'] = np.linspace(0, 100, len(by_id))

    # print(by_id)

    for col in cols:
        sns.lineplot(data=by_id, x='percentil', y=col, label='NaN - '+col, alpha=0.5)

    plt.ylabel('Proporción de NaNs')
    plt.xlabel(f'Percentil de estación en NaNs de {cols[0]}')
    plt.legend()
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(0, 100)
    plt.show()


def plot_na_distributions_interactive(by_id, cols, fsize=(16,9), fix_y_100=True):
    '''
    Same functionality as plot_na_distributions, but shows more detailed data when
    hovering cursor over it. Toggle fix_y_100 on to fix y axis scale from 0% to 100%
    '''
    by_id = by_id.copy()
    by_id = by_id.sort_values(by=cols[0]).reset_index(drop=True)
    by_id['rango'] = by_id[cols[0]].rank(ascending=True)

    df_melted = by_id.melt(
        id_vars=['indicativo', 'rango', 'n_dias_teoricos', 'altitud', 'provincia'], 
        value_vars=cols, 
        var_name='Variable', 
        value_name='Proporcion_NaNs'
    )

    df_melted['hover_header'] = df_melted.apply(
        lambda x: f"<b>Rango:</b> {int(x['rango'])}<br><b>Estación:</b> {x['indicativo']}<br><b>Provincia:</b> {x['provincia']}<br><b>Altitud:</b> {int(x['altitud'])}<br><b>Días:</b> {x['n_dias_teoricos']}", 
        axis=1
    )

    fig = px.line(
        df_melted, 
        x='hover_header', 
        y='Proporcion_NaNs', 
        color='Variable',
        title=f'Distribución de Nulos (ordenado por {cols[0]})'
    )

    fig.update_traces(
        mode="lines",
        hovertemplate="<b>%{data.name}</b>: %{y:.2%}<extra></extra>" 
    )

    yaxis_config = dict(
        title="% Nulos", 
        tickformat=".1%"
    )
    
    if fix_y_100:
        yaxis_config['range'] = [0, 1]

    fig.update_layout(
        hovermode='x unified',
        template='plotly_white',
        width=fsize[0]*80,
        height=fsize[1]*80,
        hoverlabel=dict(bgcolor="white", font_size=13),
        xaxis=dict(
            title="",              
            showticklabels=False, 
            showgrid=False         
        ),
        yaxis=yaxis_config
    )

    fig.show()


def plot_nan_evolution(df, var, xlim=(1990, 2024), ylim=None, title=True, fsize=(10, 4)):
    pct_missing = (
        df
        .groupby('anio')[var]
        .apply(lambda x: x.isna().mean() * 100)
        .reset_index(name='pct_missing')
    )

    fig, ax = plt.subplots(figsize=fsize)

    sns.lineplot(
        data=pct_missing,
        x='anio',
        y='pct_missing',
        marker='o',
        ax=ax
    )

    if title:
        ax.set_title(f'Porcentaje de registros con {var} ausente por año')
    ax.set_xlabel('Año')
    ax.set_ylabel('Valores perdidos (%)')

    plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()


def plot_nan_evolution_compare(
    df1,
    df2,
    var,
    labels=('DF1', 'DF2'),
    xlim=(1990, 2024),
    ylim=None,
    figsize=(10, 4),
    title=True
):
    pct_missing_1 = (
        df1.groupby('anio')[var]
        .apply(lambda x: x.isna().mean() * 100)
        .reset_index(name='pct_missing')
    )

    pct_missing_2 = (
        df2.groupby('anio')[var]
        .apply(lambda x: x.isna().mean() * 100)
        .reset_index(name='pct_missing')
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=pct_missing_1,
        x='anio',
        y='pct_missing',
        marker='o',
        label=labels[0],
        ax=ax
    )

    sns.lineplot(
        data=pct_missing_2,
        x='anio',
        y='pct_missing',
        marker='o',
        label=labels[1],
        ax=ax
    )

    if title:
        ax.set_title(f'Porcentaje de registros con {var} perdido por año')
    ax.set_xlabel('Año')
    ax.set_ylabel('Valores perdidos (%)')
    ax.set_xlim(xlim)
    ax.legend()

    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()


def plot_nan_evolution_day_of_year(df, var, xlim=(1, 365), title=True, fsize=(10, 4)):
    pct_missing = (
        df.groupby('diaAnio')[var]
        .apply(lambda x: x.isna().mean() * 100)
        .reset_index(name='pct_missing')
    )

    fig, ax = plt.subplots(figsize=fsize)

    # sns.lineplot(
    #     data=pct_missing,
    #     x='diaAnio',
    #     y='pct_missing',
    #     marker='o',
    #     ax=ax
    # )

    sns.regplot(
        data=pct_missing,
        x="diaAnio",
        y='pct_missing',
        lowess=True,
        ax=ax,
    )

    if title:
        ax.set_title(f'Porcentaje de registros con {var} ausente por día del año')
    ax.set_xlabel('Día del año')
    ax.set_ylabel('Valores ausentes (%)')

    plt.xlim(xlim)

    plt.tight_layout()
    plt.show()