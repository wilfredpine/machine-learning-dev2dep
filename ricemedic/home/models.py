from django.db import models

# Create your models here.
class RiceDisease(models.Model):
    disease_name = models.CharField(max_length=50)
    description = models.TextField(null=True)
    symptoms = models.TextField(null=True)
    treatment = models.TextField(null=True)