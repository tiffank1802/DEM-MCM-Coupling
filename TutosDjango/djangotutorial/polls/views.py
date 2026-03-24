from django.shortcuts import render,get_object_or_404
from django.http import HttpResponse
from django.http import Http404
# from django.template import loader #une fois qu'on utilise le render, on plus besoin d'utiliser le loader
from .models import Question,Choice

def index(request):
    latest_question_list=Question.objects.order_by("-pub_date")[:5]
    # template=loader.get_template("polls/index.html") # on a plus besoin d'utiliser le loader une fois qu'on utilise le render
    # output=",".join([q.question_text for q in latest_question_list])
    context={"latest_question_list": latest_question_list}
    # return HttpResponse(template.render(context,request)) # est une version ancienne qui se remplace par render
    return render(request,"polls/index.html")

def detail(request,question_id):
    # try:
    #     question=Question.objects.get(pk=question_id)
    # except Question.DoesNotExist:
    #     raise Http404("Question does not exist.")
    question=get_object_or_404(Question,pk=question_id) # la fonction get_object_or_404 permet de faire ce que fait le try except en une seule ligne
    return render(request, "polls/detail.html",{"question": question})

def results(request,question_id):
    response=f"You're looking at the results of question {question_id}."
    return HttpResponse(response)

def vote(request,question_id):
    return HttpResponse(f"You're voting on question {question_id}.")