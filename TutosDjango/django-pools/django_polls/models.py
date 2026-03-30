from django.contrib import admin
import datetime
from django.db import models
from django.utils import timezone
import numpy as np


class Question(models.Model):
    question_text=models.CharField(max_length=200)
    pub_date=models.DateTimeField("date published")
    
    def __str__(self):
        return self.question_text
    @admin.display(
        boolean=True,
        ordering="pub_date",
        description="Published Recently?",
    )
    def was_published_recently(self):
        now=timezone.now()
        return now-datetime.timedelta(days=1)<=self.pub_date<=now
    def sum(self):
        A=np.zeros((3,3))+2*np.eye(3)
        b=np.ones(3)
        b[0]=0
        return np.linalg.solve(A,b).sum()
    
    
class Choice(models.Model):
    question=models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text=models.CharField(max_length=200)
    votes=models.IntegerField(default=0)
    
    def __str__(self):
        return self.choice_text
