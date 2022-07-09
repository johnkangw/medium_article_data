"""Capacity Dash Application"""

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np
from sktime.forecasting.naive import NaiveForecaster

app = Dash(__name__)

# Import over data to analyze
# Read in and save the raw data
bike_data = pd.read_csv(r'C:\Users\JKANG1\OneDrive - Cox Automotive\Personal\EdX\Medium Dash articles\bike_data.csv')
# Clean the data using Pandas methods that are chained together
bike_data_clean = bike_data.\
    drop(['Unnamed: 0', 'Inspect ID', 'Style Code',
          'Order Number', 'User Code', 'Defect Found', 'Maint Date'], axis='columns').\
    rename(columns={'Line Number': 'Product', 'Quantity': 'Quantity Sold'}).\
    assign(created_date = lambda x: pd.to_datetime(x['Created Date'], format ='%Y%m%d')).\
    drop(['Created Date'], axis='columns')


# Groupby month and plot quantities sold
bike_data_clean.groupby(pd.Grouper(key='created_date', freq='M')).sum()['Quantity Sold'].plot()
clean_data = bike_data.assign(**{'Created Date': lambda x: pd.to_datetime(x['Created Date'], format="%Y%m%d"),
                                 'Maint Date': lambda x: pd.to_datetime(x['Created Date'], format="%Y%m%d")})
# Prepare the data for time series forecasting
forecast_ready_2 = bike_data_clean.set_index('created_date').\
    sort_index().\
    groupby([pd.Grouper(freq='1M'),'Product']).\
    sum().\
    reset_index().\
    set_index('created_date')
# Split data into products
cool_kids_bike_2 = forecast_ready_2.loc[forecast_ready_2['Product'] == 'Cool Kids Bike'].\
    drop(labels=['Product', "single_digit_year"], axis='columns').\
    assign(created_date = lambda df: df.index.to_period('M')).\
    set_index('created_date')

forecasting_horizon = np.arange(1, 25)
forecaster = NaiveForecaster(
    strategy='mean',
    sp=12
)
forecaster.fit(cool_kids_bike_2)
y_pred = forecaster.predict(forecasting_horizon)
y_pred.index.name = "Date"
y_pred.rename(columns={y_pred.columns[0]: "Forecast"}, inplace=True)

y_plot = y_pred.reset_index()
y_plot.Date = y_plot.Date.astype('datetime64[ns]')

fig = px.line(y_plot, x="Date", y="Forecast")

app.layout = html.Div(children=[
    html.H1(children='Naive Demand Forecast for next year plotted'),

    html.Div(children='''
        Plot
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
