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
            'username': 'RandomUser',
            'password': 'RandomPass',
            'submit': 'signin'
        }
        c = Client()
        response = c.post('/accounts/',
                          username=user['username'],
                          password=user['password'],
                          submit=user['submit']
                          )

        self.assertEqual(response.status_code, 400)


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

        self.assertEqual(response.status_code, 400)

class StockTest(TestCase):

    user = {
            'username': 'TestUser123',
            'password': '*G3Zd54(.ys8(nE_',
            'submit': 'signin'
        }

    # setup the test by making a new user
    def setUp(self):
        new_user = User.objects.create(username=self.user['username'])
        new_user.set_password(self.user['password'])
        new_user.save()
        pass

    def test_valid_stock_passed(self):

        c = Client()
        print(self.user['username'])
        c.login(username=self.user['username'], password=self.user['password'])
        response = c.post('/stock/aapl')
        print(response.status_code)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/stock.html')


        

