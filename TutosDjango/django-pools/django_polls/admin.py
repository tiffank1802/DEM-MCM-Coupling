from django.contrib import admin
from .models import Question,Choice


class ChoiceInline(admin.TabularInline): 
    """Cette classe permet d'afficher plus d'option de création des questions d'un  choix donné"""
    model=Choice
    extra=3
   

    
class QuestionAdmin(admin.ModelAdmin):
    # fields=["pub_date","question_text"]
    fieldsets=[
        ("Text",{"fields":["question_text"]}),
        ("Date Informations",{"fields":["pub_date"],"classes":["collapse"]}),
    ]
    inlines=[ChoiceInline]
    list_display=["question_text","pub_date","was_published_recently","sum"]
    list_filter=["pub_date"]

class ChoiceAdmin(admin.ModelAdmin):
    fieldsets=[
        ("Question",{"fields":["question"]}),
        ("Choices",{"fields":["choice_text"]}),
        ("Votes",{"fields":["votes"]})
    ]

admin.site.register(Question,QuestionAdmin)
admin.site.register(Choice,ChoiceAdmin)
