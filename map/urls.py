from django.urls import path
from . import views

urlpatterns = [
    path('', views.mapApi, name='index'),
    path('detail/', views.detail, name='detail'),
    path('detail/homeinfo/', views.homeinfo, name='homeinfo'),
    path('process/', views.process_pdf, name='process_pdf'),
]