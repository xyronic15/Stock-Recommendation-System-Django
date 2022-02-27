from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.auth.models import User


class SignUpForm(UserCreationForm):
    first_name = forms.CharField(required=True, max_length=200)
    last_name = forms.CharField(required=True, max_length=200)
    email = forms.CharField(required=True)
    username = forms.CharField(required=True, max_length=200)

    class Meta:
        model = User
        fields = [
            'first_name',
            'last_name',
            'email',
            'username'
        ]

        def save(self, commit=True):
            new_user = super(SignUpForm, self).save(commit=False)
            new_user.firstname = self.cleaned_data['first_name']
            new_user.lastname = self.cleaned_data['last_name']
            new_user.email = self.cleaned_data['email']
            new_user.username = self.clean_data['username']

            if commit:
                new_user.save()

            return new_user
