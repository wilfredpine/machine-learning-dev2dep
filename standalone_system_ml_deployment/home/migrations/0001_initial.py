# Generated by Django 5.1.3 on 2024-11-20 06:41

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='RiceDisease',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('disease_name', models.CharField(max_length=50)),
                ('description', models.TextField(null=True)),
                ('symptoms', models.TextField(null=True)),
                ('treatment', models.TextField(null=True)),
            ],
        ),
    ]