from django.urls import path

from .views import hdr_predict

urlpatterns = [
    path("", hdr_predict, name="hdr_predict_num")
]