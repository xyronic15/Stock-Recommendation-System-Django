from django.urls import path
from . import views

urlpatterns = [
    path("home", views.home, name="home"),
    path("stock/<ticker>", views.stock, name="stock"),
    path("predict/", views.predict, name="predict"),

    path('accounts/', views.account, name="accounts")
]