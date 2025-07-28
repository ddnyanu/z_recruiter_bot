from django.urls import path
from .views import ResumeParserAPIView,RegenerateSummaryAPIView


from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "ok"})

urlpatterns = [
     path('', health_check),
    path('parse/', ResumeParserAPIView.as_view(), name='parse-resume'),
    path('regenerate-summary/', RegenerateSummaryAPIView.as_view()), 
]
