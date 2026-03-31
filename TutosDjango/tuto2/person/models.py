from django.db import models

class Person (models.Model):
    first_name=models.CharField(max_length=30)
    last_name=models.CharField(max_length=30)
    YEAR_IN_SCHOOL_CHOICES=[
        ("FR","Freshman"),
        ("SO","Sophomore"),
        ("JR","Junior"),
        ("SR","Senior"),
        ("GR","Graduate"),
    ]
    statut=models.CharField(max_length=2,choices=YEAR_IN_SCHOOL_CHOICES)

class Musician(models.Model):
    first_name=models.CharField("first name",max_length=50)
    last_name=models.CharField("last name",max_length=50)
    instrument=models.CharField("instrument",max_length=100)

class Album(models.Model):
    artist=models.ForeignKey(Musician,on_delete=models.CASCADE,verbose_name="related artists")
    editor=models.ForeignKey(Person,verbose_name="related editors",on_delete=models.CASCADE)
    name=models.CharField(max_length=50)
    release_date=models.DateField()
    num_stars=models.IntegerField()

class Runner(models.Model):
    MedalType=models.TextChoices("MetalType","GOLD SILVER BRONZE")
    name=models.CharField(max_length=60)
    medal=models.CharField(blank=True,choices=MedalType,max_length=10)

    


