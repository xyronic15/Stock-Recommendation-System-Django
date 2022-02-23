from django.urls import path
from . import views

urlpatterns = [
    path("home", views.home, name="home"),
    path("stock/<ticker>", views.stock, name="stock"),
    path("predict/<ticker>", views.predict, name="predict"),
]