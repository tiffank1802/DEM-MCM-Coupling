from django.db.models import F
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render,get_object_or_404
from django.urls import reverse
# from django.http import Http404
# from django.template import loader #une fois qu'on utilise le render, on plus besoin d'utiliser le loader
from django.views import generic
from django.utils import timezone
from .models import Question,Choice


class IndexView(generic.ListView):
    template_name="polls/index.html"
    context_object_name="latest_question_list"
    model=Question
    
    def get_queryset(self):
        return self.model.objects.filter(pub_date__lte=timezone.now()).order_by("pub_date")[:5]

class DetailView(generic.DetailView):
    model=Question
    template_name="polls/results.html"
    def get_queryset(self):
        return self.model.objects.filter(pub_date__lte=timezone.now())
    
class ResultsView(generic.DetailView):
    model=Question
    template_name="polls/results.html"
    
    
def vote(request,question_id):
    question=get_object_or_404(Question,pk=question_id)
    try:
        selected_choice=question.choice_set.get(pk=request.POST["choice"])
    except(KeyError, Choice.DoesNotExist):
        return render(
            request,
            "polls/detail.html",
            {
                "question":question,
                "error_message":"You didn't select a choice.",
            },
        )
    else:
        selected_choice.votes=F("votes")+1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        #with POST data. This prevents data from being posted twice if a user hits the Back button.
        return HttpResponseRedirect(reverse("polls:results",args=question.id,))
    













# def results(request,question_id):
#     question=get_object_or_404(Question,pk=question_id)
#     return render(request,"polls/results.html",{"question":question})


# def index(request):
#     latest_question_list=Question.objects.order_by("-pub_date")[:5]
#     # template=loader.get_template("polls/index.html") # on a plus besoin d'utiliser le loader une fois qu'on utilise le render
#     # output=",".join([q.question_text for q in latest_question_list])
#     context={"latest_question_list": latest_question_list}
#     # return HttpResponse(template.render(context,request)) # est une version ancienne qui se remplace par render
#     return render(request,"polls/index.html")

# def detail(request,question_id):
#     # try:
#     #     question=Question.objects.get(pk=question_id)
#     # except Question.DoesNotExist:
#     #     raise Http404("Question does not exist.")
#     question=get_object_or_404(Question,pk=question_id) # la fonction get_object_or_404 permet de faire ce que fait le try except en une seule ligne
#     return render(request, "polls/detail.html",{"question": question})
