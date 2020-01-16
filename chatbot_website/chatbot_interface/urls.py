from django.conf.urls import url
# from django.urls import path
from . import views

from .chatbotmanager import ChatbotManager

# test =  url(r'^$', 'index', name='index')
# print("test", test)
# print("main_view", views.mainView)
urlpatterns = [
    url(r'^$', views.mainView),
    url('Homer/', views.homer, name='homer'),
    url('Marge/', views.marge, name="marge"),
    url('Bart/', views.bart, name='bart'),
    url('Lisa/', views.lisa, name="lisa"),
]

