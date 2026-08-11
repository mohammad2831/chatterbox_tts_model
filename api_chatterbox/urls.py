from django.urls import path
from .views import GenerateTTSView

urlpatterns = [
    path('generate/', GenerateTTSView.as_view(), name='tts-generate'),
]
