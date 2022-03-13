from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import Favourite
from .helper import get_ticker, get_past, get_candlestick, get_scatter, placeholder_plot
from .predictor import predictor
from django.contrib.auth.models import User
from django.db.models import Q

PAST_PERIOD = "5Y"

from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login

from .forms import SignUpForm


# Create your views here.
@login_required(login_url='/accounts')
def home(request):

    # get user's name
    fname, lname = request.user.first_name, request.user.last_name

    # retrieve all the favourites associated with the user
    favourites = Favourite.objects.filter(userID=request.user)

    msg = None

    return render(request, 'main/home.html', {"fname":fname, "lname": lname, "favourites": favourites, "msg": msg})

@login_required(login_url='/accounts')
def stock(request, ticker):
    print("Stock view called")

    stock = get_ticker(ticker)
    if stock.info['regularMarketPrice'] == None:
        # get user's name
        fname, lname = request.user.first_name, request.user.last_name

        # retrieve all the favourites associated with the user
        favourites = Favourite.objects.filter(userID=request.user)
        
        msg = "No such ticker as " + ticker
        return render(request, 'main/home.html', {"fname": fname, "lname": lname, "favourites": favourites, "msg": msg}, status=301)

    # check if favourite exists for this user
    curr_fav = False
    favourite = None
    try:
        favourite = Favourite.objects.filter(Q(ticker__iexact=ticker) & Q(userID=request.user))
    except:
        favourite = None
    
    if favourite:
        curr_fav = True
    
    # print(favourite)
    # curr_fav = Favourite.objects.exists(Q(ticker__iexact=ticker) & Q(userID=request.user))
    # print(curr_fav)
    # print(request.user)

    df = get_past(stock, PAST_PERIOD)
    scatter_div = get_scatter(df)
    candle_div = get_candlestick(df)
    prediction = None

    request.session["stock"] = stock
    request.session["curr_fav"] = curr_fav
    request.session["scatter_div"] = scatter_div
    request.session["candle_div"] = candle_div

    return render(request, 'main/stock.html',
                  {"scatter": scatter_div, "candlestick": candle_div, "stock": stock, "favourite": curr_fav, "prediction": prediction})

@login_required(login_url='/accounts')
def predict(request, ticker):
    
    print("Predict view called")
    stock = request.session["stock"]
    curr_fav = request.session["curr_fav"]
    scatter_div = request.session["scatter_div"]
    candle_div = request.session["candle_div"]

    # Create a predictor object for the stock and retrieve the recommendations
    predict_stock = predictor(ticker)
    recommendation_list = predict_stock.recommendation()
    predict_div = predict_stock.get_div(predict_stock.predict_fig)

    # predict_div = placeholder_plot()

    return render(request, 'main/stock.html', {"scatter":scatter_div, "candlestick": candle_div, "stock": stock, "favourite": curr_fav, "prediction": predict_div, "recommendation_list": recommendation_list})

def add_favourite(request):
    if request.method == "POST":
        ticker = request.POST.get('ticker')

        favourite = Favourite(userID=request.user, ticker=ticker)
        print(ticker)
        favourite.save()
        # request.user.favourite.add(favourite)

        print("Added " + ticker + " to favourites")

        return HttpResponse("Added " + ticker + " to favourites")
    else:
        return HttpResponse("Could not add ticker to favourites")

def remove_favourite(request):
    if request.method == "POST":
        ticker = request.POST.get('ticker')

        try:
            print("Ticker to be removed + Current user:")
            print(ticker)
            print(request.user)

            f = Favourite.objects.get(Q(ticker__iexact=ticker) & Q(userID=request.user))
            f.delete()

            print("Removed " + ticker + " from favourites")

            return HttpResponse("Removed " + ticker + " from favourites")
        except:
            return HttpResponse("Failed to remove from favourites")


def account(request):
    user = None
    form = None
    print(request.POST)
    if request.method == 'POST':
        if request.POST.get('submit') == 'signin':
            user = authenticate(username=request.POST['username'],
                                password=request.POST['password'])
            if user is not None:
                login(request, user)
        elif request.POST.get('submit') == 'signup':
            form = SignUpForm(request.POST)
            if form.is_valid():
                form.save()

        if user is not None or (form and form.is_valid()):
            return redirect('/home')

    data = {
        'login': AuthenticationForm(),
        'register': SignUpForm()
    }
    return render(request, 'main/account.html', data)
