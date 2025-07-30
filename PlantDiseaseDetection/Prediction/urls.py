from django.urls import path
from .import views


urlpatterns = [
    path('',views.predict,name='predict'),
    path('about/',views.about,name='about'),
    path('contact/',views.contact,name='contact'),
    path('news/',views.news,name='news'),
    path('hindi_news/',views.hindi_news,name='hindi_news'),
    path('plant_info/',views.plant_info,name='plant_info'),
    path('quick_links/',views.quick_links,name='quick_links')
]
