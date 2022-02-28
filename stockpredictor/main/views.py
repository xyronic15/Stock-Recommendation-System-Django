from django.shortcuts import render, redirect
from .helper import get_ticker, get_past, get_candlestick, get_scatter, placeholder_plot

from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login

from .forms import SignUpForm


# Create your views here.
@login_required(login_url='/accounts')
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

    return render(request, 'main/stock.html',
                  {"scatter": scatter_div, "candlestick": candle_div, "stock": stock, "prediction": prediction})


def predict(request):
    print("Predict view called")
    stock = request.session["stock"]
    scatter_div = request.session["scatter_div"]
    candle_div = request.session["candle_div"]

    predict_div = placeholder_plot()

    return render(request, 'main/stock.html',
                  {"scatter": scatter_div, "candlestick": candle_div, "stock": stock, "prediction": predict_div})


def account(request):
    user = None
    form = None

    if request.method == 'POST':
        if request.POST.get('submit') == 'signin':
            user = authenticate(username=request.POST['username'],
                                password=request.POST['password'])
            if user is not None:
                login(request, user)
        elif request.POST.get('submit') == 'signup':
            form = SignUpForm(request.POST)
            print(form)
            if form.is_valid():
                form.save()

        if user is not None or form.is_valid():
            return redirect('/home')

    data = {
        'login': AuthenticationForm(),
        'register': SignUpForm()
    }
    return render(request, 'main/account.html', data)
