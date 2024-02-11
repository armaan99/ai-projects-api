# from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # path('admin/', admin.site.urls),
    # path('api/hdr', include("hdr_api.urls")),
    # path('api/hdr_LeNet', include("hdr_LeNet_api.urls")),
    path('api/lens', include("lens_tryon_api.urls")),
]
