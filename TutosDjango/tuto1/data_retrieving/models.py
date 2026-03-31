from django.db import models


class Matrix(models.Model):
    matrix=models.CharFields(max_length=30)
    colum_sums=models.CharFields(max_length=30)
    
