import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px


def create_histogram(df):
    '''Create histogram plot for Pandas dataframe

    Parameters:
        df: Pandas dataframe

    Returns:
        None

    '''
    plt.figure(figsize=(15,50))
    columns = df.columns
    num_rows = len(columns)
    for col in range(0,num_rows):
        ax = plt.subplot(int(num_rows/2),2,col+1)
        sns.histplot(df[columns[col]], kde=True, ax=ax)
        plt.xlabel(columns[col])
        plt.ylabel('count')
    plt.show()

def create_boxplot(df):
    '''Create boxplot plot for Pandas dataframe

    Parameters:
        df: Pandas dataframe

    Returns:
        None

    '''
    plt.figure(figsize=(15,50))
    columns = df.columns
    num_rows = len(columns)
    for col in range(0,num_rows):
        ax = plt.subplot(int(num_rows/2),2,col+1)
        sns.boxplot(df[columns[col]], ax=ax)
        plt.xlabel(columns[col])
    plt.show()

def timeseries_price(df, column, unit, title):
    '''Create timeseries chart for Pandas dataframe

    Parameters:
        df: Pandas dataframe,
        column: the name of columns (str)
        unit: unit of measurement (str)
        title: title of the plot (str)

    Returns:
        None

    '''
    plt.figure(figsize=(18,8))
    plt.plot(df[column])
    plt.xlabel('hourly date')
    plt.ylabel(unit)
    plt.title(title)
    plt.show()

def create_correlation_map(df, cmap, annotation=True):
    '''Create heatmap for Pandas dataframe

    Parameters:
        df: Pandas dataframe
        cmap: color map reference (https://matplotlib.org/stable/gallery/color/colormap_reference.html) (str)
        annotation: whether to display annot or not (bool)

    Returns:
        None

    '''
    plt.figure(figsize=(15,12.5))

    # .corr heatmap of df to visualize correlation & show plot
    sns.heatmap(round(df.corr(),1), annot=annotation, cmap=cmap, linewidth=0.9)
    plt.show()


def create_price_per_total_load_chart(df):
    '''Create scatter plot for price actual per total load
    displayed per season chart

    Parameters:
        df: Pandas dataframe

    Returns:
        None

    '''
    fig = px.scatter(df,x='total_load_actual',
                    y='price_actual',
                    facet_col='season',
                    opacity=0.1,
                    title='Price Per MW Hour Compaired To Total Energy Genereated Per Season',)
                    #animation_frame=df.index.year)

    # Figure customizations
    fig.update_traces(marker=dict(size=12,
                                line=dict(width=2,
                                            color='darkslateblue')),
                    selector=dict(mode='markers'))
    fig.show()


if __name__ == '__main__':
    '''
    this is main function to test the codes above

    '''
    pass
    from data_import import *
    df = import_merged_data()
    create_correlation_map(df, 'bwr', annotation=False)
