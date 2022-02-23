from django.shortcuts import render
from .helper import get_ticker, get_past, get_candlestick, get_scatter, placeholder_plot

# Create your views here.
def home(request):
    return render(request, 'main/home.html', {})

def stock(request, ticker):

    print("Stock view called")
    stock = get_ticker(ticker)
    df = get_past(stock)
    scatter_div = get_scatter(df)
    candle_div = get_candlestick(df)
    prediction = None
    
    request.session["stock"] = stock
    request.session["scatter_div"] = scatter_div
    request.session["candle_div"] = candle_div

    return render(request, 'main/stock.html', {"scatter":scatter_div, "candlestick": candle_div, "stock": stock, "prediction": prediction})

def predict(request, ticker):
    
    print("Predict view called")
    # stock = request.session["stock"]
    stock = get_ticker(ticker)
    scatter_div = request.session["scatter_div"]
    candle_div = request.session["candle_div"]


    predict_div = placeholder_plot()

    return render(request, 'main/stock.html', {"scatter":scatter_div, "candlestick": candle_div, "stock": stock, "prediction": predict_div})
