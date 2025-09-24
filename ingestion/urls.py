from django.urls import path
from ingestion.views import UploadView

urlpatterns = [
    path('', UploadView.as_view(), name='ingest-upload'),
]


