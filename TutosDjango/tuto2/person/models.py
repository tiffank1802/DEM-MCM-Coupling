from ast import mod
from typing import override
from django.db import models
from django.db.models import constraints

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
    SHIRT_SIZE={
    "S":"Small",
    "M":"Medium",
    "L":"Large",
    }
    shirt_size=models.CharField(max_length=1,choices=SHIRT_SIZE)

class Musician(models.Model):
    first_name=models.CharField("first name",max_length=50)
    last_name=models.CharField("last name",max_length=50)
    instrument=models.CharField("instrument",max_length=100)

class Album(models.Model):
    artist=models.ForeignKey(Musician,on_delete=models.CASCADE,verbose_name="related artists")
    editor=models.ForeignKey(Person,verbose_name="related editors",on_delete=models.CASCADE,null=True)
    name=models.CharField(max_length=50)
    release_date=models.DateField()
    num_stars=models.IntegerField()

class Runner(models.Model):
    MedalType=models.TextChoices("MetalType","GOLD SILVER BRONZE")
    name=models.CharField(max_length=60)
    medal=models.CharField(blank=True,choices=MedalType,max_length=10)

class Poll(models.Model):
    question_text=models.CharField(max_length=30)

class Site(models.Model):
    site_name=models.CharField(max_length=100)

class Place(models.Model):
    place_name=models.CharField(max_length=50)

class Investigation(models.Model):
    poll=models.ForeignKey(Poll,on_delete=models.CASCADE,verbose_name="the related poll")
    sites=models.ManyToManyField(Site,verbose_name="list of sites")
    place=models.OneToOneField(Place,on_delete=models.CASCADE,verbose_name="related place")
    

class Manufacturer(models.Model):
    pass

class Car(models.Model):
    Manufacturer=models.ForeignKey(Manufacturer,on_delete=models.CASCADE)

class Topping(models.Model):
    pass

class Pizza(models.Model):
    toppings=models.ManyToManyField(Topping) # il est important de mettre un champ models.ManyToManyField() dans un seul modèle pas dans les deux

###########################################
# Définition des champs supplémentaires dans une relationo ManyToManyField
###########################################
class  Personae(models.Model):
    name=models.CharField(max_length=128)

    
    def __str__(self):
        return self.name

class Group(models.Model):
    name=models.CharField(max_length=128)
    members=models.ManyToManyField(Personae,
    through="Membership" # Déclaration du modèle intermediare Membership. Cettte déclaration ne se fait que sur une seule classe
    #through permet de définir la clé étrangère qui est la classe intermediare
    )

    
    def __str__(self):
        return self.name

class Membership(models.Model):  # Modèle intermediare dans la relation ManyToMany
    # Définition des clés étrangères vers les modèles impliqués dans la relation ManyToMany 
    person=models.ForeignKey(Personae,on_delete=models.CASCADE)
    group=models.ForeignKey(Group,on_delete=models.CASCADE)


    date_joined=models.DateField()
    invite_reason=models.CharField(max_length=64)

    class Meta:
        constraints=[
            models.UniqueConstraint(
                fields=["person","group"],name="unique_person_group"
            )
        ]
    