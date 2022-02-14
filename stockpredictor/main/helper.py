import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd



# retrieve stock ticker object and history of past 5 years
def get_ticker(ticker):
    stock = yf.Ticker(ticker)
    return stock

def get_past(stock):
    df = stock.history(period="5y")
    df.reset_index(inplace=True)
    return df

# retrieve candlestick chart of df
def get_candlestick(df):
    candlesticks = go.Candlestick(x=df['Date'],
                            open=df['Open'],
                            close=df['Close'],
                            high=df['High'],
                            low=df['Low'],
                            name="Candlestick")
    
    fig = go.Figure()
    fig.add_trace(candlesticks)
    fig.update_layout(
        title="Candlestick Chart",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgb(0,0,0,0)", 
        autosize=True,
        width=500,
        height=375,
        margin=dict(
            l=30,
            r=30,
            b=60,
            t=50,
            pad=4
        ),
    )
    candle_div = plot(fig, output_type='div')
    return candle_div

# retrieve the 
def get_scatter(df):
    scatter = go.Scatter(x=df['Date'],
                        y=df['Close'],
                        name="Closing Price")
    
    fig = go.Figure()
    fig.add_trace(scatter)
    fig.update_layout(
        title="Closing Price Scatter Chart",
        yaxis_title="Closing Price ($)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgb(0,0,0,0)", 
        autosize=True,
        width=500,
        height=375,
        margin=dict(
            l=30,
            r=30,
            b=60,
            t=50,
            pad=4
        ),
    )
    scatter_div = plot(fig, output_type='div')
    return scatter_div

# TBC: replace this method with prediction methods
# This is just to send a dummy graph for UI design purposes
def placeholder_plot():
    x_data = [0,1,2,3]
    y_data = [x**2 for x in x_data]
    random_graph = go.Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='green')
    fig = go.Figure()
    fig.add_trace(random_graph)
    fig.update_layout(
        title="Closing Price Scatter Chart",
        yaxis_title="Closing Price ($)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgb(0,0,0,0)", 
        autosize=True,
        width=500,
        height=375,
        margin=dict(
            l=30,
            r=30,
            b=60,
            t=50,
            pad=4
        ),
    )
    plot_div = plot(fig,
               output_type='div')
    return plot_div