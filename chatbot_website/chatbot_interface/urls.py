from django.conf.urls import url
# from django.urls import path
from . import views

from .chatbotmanager import ChatbotManager

# test =  url(r'^$', 'index', name='index')
# print("test", test)
print("main_view", views.mainView)
urlpatterns = [
    url(r'^$', views.mainView),
    url('homer/', views.detail, name='homer'),
    # url(r'^$', views.detail, name="new"),
]
print("url1", url(r'^$', views.mainView))
print("url2", url(r'^$', views.detail))
