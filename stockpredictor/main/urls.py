from django.urls import path, include
from django.contrib import admin
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path("home/", views.home, name="home"),
    path("stock/", views.stock, name="stock"),
    path("predict/<ticker>", views.predict, name="predict"),
    path('accounts/', views.account, name="accounts"),
    path('favourite/', views.add_favourite, name="favourite"),
    path('unfavourite/', views.remove_favourite, name="unfavourite")
]