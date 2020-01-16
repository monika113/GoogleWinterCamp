from django.conf.urls import url
# from django.urls import path
from . import views

from .chatbotmanager import ChatbotManager

# test =  url(r'^$', 'index', name='index')
# print("test", test)
# print("main_view", views.mainView)
urlpatterns = [
    url(r'^$', views.mainView),
    url('homer/', views.detail, name='homer'),
    url('marge/', views.marge, name="marge"),
    url('bart/', views.bart, name='bart'),
    url('lisa/', views.lisa, name="lisa"),
]

