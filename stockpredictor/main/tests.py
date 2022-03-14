from django.test import TestCase
from django.test import Client
from django.contrib.auth.models import User
from .models import Favourite
from django.db.models import Q


from django.test import TestCase
from django.test import Client
from django.contrib.auth.models import User


class LoginTest(TestCase):
    # testing if page displays when user visits same address
    def test_view_page(self):
        response = self.client.get('/accounts/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/account.html')

    # testing login of user
    def test_user_login_success(self):
        user = {
            'username': 'TestUser123',
            'password': '*G3Zd54(.ys8(nE_',
            'submit': 'signin'
        }

        new_user = User.objects.create(username=user['username'])
        new_user.set_password(user['password'])
        new_user.save()

        c = Client()
        response = c.post('/accounts/', username=user['username'], password=user['password'], submit=user['submit'])

        self.assertEqual(response.status_code, 200)

    # testing login of user that does not exist
    def test_user_login_does_not_exist(self):
        user = {
            'first_name': 'TestFirstName',
            'last_name': 'TestLastName',
            'username': 'TestUser123',
            'password': '*G3Zd54(.ys8(nE_',
            'submit': 'signup'
        }
        c = Client()
        response = c.post('/accounts/',
                          first_name=user['first_name'],
                          last_name=user['last_name'],
                          username=user['username'],
                          password1=user['password'],
                          password2=user['password'],
                          submit=user['submit']
                          )

        response = c.post('/accounts/',
                          "TestUsername",
                          "TestPassword123",
                          submit='signin'
                          )

        # self.assertEqual(response.status_code, 400)
        self.assertTemplateUsed(response, 'main/account.html')


class SignupTest(TestCase):
    # testing if page displays when user visits same address
    def test_user_signup_success(self):
        user = {
            'first_name': 'TestFirstName',
            'last_name': 'TestLastName',
            'username': 'TestUser123',
            'password': '*G3Zd54(.ys8(nE_',
            'submit': 'signup'
        }

        c = Client()
        response = c.post('/accounts/',
                          first_name=user['first_name'],
                          last_name=user['last_name'],
                          username=user['username'],
                          password1=user['password'],
                          password2=user['password'],
                          submit=user['submit']
                          )

        self.assertEqual(response.status_code, 200)

    # testing if user enters non-matching passwords
    def test_not_same_pass(self):
        user = {
            'first_name': 'TestFirstName',
            'last_name': 'TestLastName',
            'username': 'TestUser123',
            'password': '*G3Zd54(.ys8(nE_',
            'submit': 'signup'
        }

        c = Client()
        response = c.post('/accounts/',
                          first_name=user['first_name'],
                          last_name=user['last_name'],
                          username=user['username'],
                          password1=user['password'],
                          password2='random',
                          submit=user['submit']
                          )

        # self.assertEqual(response.status_code, 400)
        self.assertTemplateUsed(response, 'main/account.html')


class StockTest(TestCase):

    # setup the test by making a new user
    def setUp(self):
        self.user = {
            'username': 'TestUser123',
            'password': '*G3Zd54(.ys8(nE_',
            'first_name': "tester",
            'last_name': "123",
            'submit': 'signin'
        }

        self.new_user = User.objects.create(username=self.user['username'], first_name=self.user['first_name'], last_name=self.user['last_name'])
        self.new_user.set_password(self.user['password'])
        self.new_user.save()

        self.client.login(username=self.user['username'], password=self.user['password'])

    def test_valid_stock_passed(self):

        response = self.client.post('/stock/aapl')
        # print(response.status_code)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/stock.html')
        self.assertIsNotNone(response.context['scatter'])
        self.assertIsNotNone(response.context['candlestick'])
        self.assertIsNotNone(response.context['stock'])
        self.assertIsNotNone(response.context['favourite'])
        self.assertIsNone(response.context['prediction'])
    
    def test_invalid_stock_passed(self):
        
        response = self.client.post('/stock/35634')
        # print(response.status_code)

        favourites = Favourite.objects.filter(userID=response.context['user'])

        self.assertEqual(response.status_code, 301)
        self.assertTemplateUsed(response, 'main/home.html')
        self.assertEqual(response.context['msg'], "No such ticker as 35634")
        self.assertEqual(response.context['fname'], self.user['first_name'])
        self.assertEqual(response.context['lname'], self.user['last_name'])
        self.assertEqual(list(response.context['favourites']), list(favourites))

    def test_add_favourite(self):

        ticker = "aapl"
        response = self.client.post("/favourite/", {"ticker": ticker})
        # print(response.content.decode("utf-8") == "Added " + ticker + " to favourites")
        # print(response.content == "Added " + ticker + " to favourites")

        favourite = Favourite.objects.filter(userID=self.new_user)[0]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode("utf-8"), "Added " + ticker + " to favourites")
        self.assertEqual(favourite.ticker, ticker)

    def test_remove_valid_favourite(self):
    
        ticker = "aapl"
        self.client.post("/favourite/", {"ticker": ticker})
        response = self.client.post("/unfavourite/", {"ticker": ticker})
        # print(response.content.decode("utf-8") == "Added " + ticker + " to favourites")
        # print(response.content == "Added " + ticker + " to favourites")

        favourites = Favourite.objects.filter(userID=self.new_user)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode("utf-8"), "Removed " + ticker + " from favourites")
        self.assertTrue(len(favourites) == 0)

    def test_remove_invalid_favourite(self):

        ticker = "aapl"
        self.client.post("/favourite/", {"ticker": ticker})
        response = self.client.post("/unfavourite/", {"ticker": "1234"})

        favourites = Favourite.objects.filter(userID=self.new_user)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode("utf-8"), "Failed to remove from favourites")
        self.assertFalse(len(favourites) == 0)

class PredictTest(TestCase):

    # setup the test by making a new user
    def setUp(self):
        self.user = {
            'username': 'TestUser123',
            'password': '*G3Zd54(.ys8(nE_',
            'first_name': "tester",
            'last_name': "123",
            'submit': 'signin'
        }

        new_user = User.objects.create(username=self.user['username'], first_name=self.user['first_name'], last_name=self.user['last_name'])
        new_user.set_password(self.user['password'])
        new_user.save()

        self.client.login(username=self.user['username'], password=self.user['password'])
    
    def test_prediction(self):

        self.client.post('/stock/aapl')
        response = self.client.post("/predict/aapl")
        # print(response.status_code)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/stock.html')
        self.assertIsNotNone(response.context['scatter'])
        self.assertIsNotNone(response.context['candlestick'])
        self.assertIsNotNone(response.context['stock'])
        self.assertIsNotNone(response.context['favourite'])
        self.assertIsNotNone(response.context['prediction'])
        self.assertIsNotNone(response.context["recommendation_list"])