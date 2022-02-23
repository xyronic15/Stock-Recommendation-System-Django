from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Favourite(models.Model):
    userID = models.ForeignKey(User, on_delete=models.CASCADE)
    ticker  = models.CharField(max_length=200)

    def __str__(self):
        return self.ticker