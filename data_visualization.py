import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px


def create_histogram(df):
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
    plt.figure(figsize=(15,50))
    columns = df.columns
    num_rows = len(columns)
    for col in range(0,num_rows):
        ax = plt.subplot(int(num_rows/2),2,col+1)
        sns.boxplot(df[columns[col]], ax=ax)
        plt.xlabel(columns[col])
    plt.show()

def timeseries_price(df, column, title):
    plt.figure(figsize=(18,8))
    plt.plot(df[column])
    plt.xlabel('hourly date')
    plt.ylabel('euro/MWh')
    plt.title(title)
    plt.show()

def create_correlation_map(df):
    plt.figure(figsize=(15,12.5))

    # .corr heatmap of df to visualize correlation & show plot
    sns.heatmap(round(df.corr(),1), annot=True, cmap='Purples', linewidth=0.9)
    plt.show()


def create_price_per_total_load_chart(df):
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
    from data_import import *
    df = import_clean_energy_data()
    # # create_histogram(df)
    # create_boxplot(df)
    # timeseries_price(df, 'price_actual', 'my title')
    # create_correlation_map(df)
    # Sort index
    # df = df.sort_index()

    # # Set conditional satements for filtering times of month to season value
    # condition_winter = (df.index.month>=1)&(df.index.month<=3)
    # condtion_spring = (df.index.month>=4)&(df.index.month<=6)
    # condition_summer = (df.index.month>=7)&(df.index.month<=9)
    # condition_automn = (df.index.month>=10)@(df.index.month<=12)

    # # Create column in dataframe that inputs the season based on the conditions created above
    # df['season'] = np.where(condition_winter,'winter',
    #                         np.where(condtion_spring,'spring',
    #                                 np.where(condition_summer,'summer',
    #                                         np.where(condition_automn,'autumn',np.nan))))
    # create_price_per_total_load_chart(df)
    df2 = import_clean_weather_data()
    print(df2.head())
