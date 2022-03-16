# Generated by Django 4.0.3 on 2022-03-16 14:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_user_alter_favourite_userid'),
    ]

    operations = [
        migrations.AddField(
            model_name='favourite',
            name='logo_url',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AddField(
            model_name='favourite',
            name='longName',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AlterField(
            model_name='favourite',
            name='ticker',
            field=models.CharField(default='', max_length=200),
        ),
    ]