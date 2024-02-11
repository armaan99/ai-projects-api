from django.urls import path

from .views import apply_facemesh, apply_lens

urlpatterns = [
    path("/facemesh", apply_facemesh, name="facemesh"),
    path("/try-lens", apply_lens, name="try_lens")
]