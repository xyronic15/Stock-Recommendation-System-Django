from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Favourite(models.Model):
    userID = models.ForeignKey(User, on_delete=models.CASCADE)
    ticker  = models.CharField(max_length=200)

    def __str__(self):
        return self.ticker


class User(models.Model):
    first_name = models.CharField(max_length=200)
    last_name = models.CharField(max_length=200)
    email = models.EmailField()
    username = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
