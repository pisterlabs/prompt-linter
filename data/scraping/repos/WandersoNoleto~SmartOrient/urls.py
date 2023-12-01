from django.urls import path

from guidances import views

urlpatterns = [
    path('register-guidance-page/student', views.register_guidance_page, name="register_guidance_page"),
    path('register-guidance-save/', views.register_guidance_save, name="register_guidance_save"),
    path('pending-guidances/', views.guidances_pending_page, name="guidances_pending_page"),
    path('accept-guidance/<int:id>/', views.pending_guidance_accept, name="pending_guidance_accept"),
    path('delete-guidance/<int:id>/', views.delete_guidance, name="delete_guidance"),
    path('open-guidance/<int:id>/', views.open_guidance, name="open_guidance")
]