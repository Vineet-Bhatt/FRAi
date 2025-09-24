from django.db import models

# Create your models here.
from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    file_type = models.CharField(max_length=16, choices=[('csv','csv'),('xml','xml'),('bin','bin')])
    created_at = models.DateTimeField(auto_now_add=True)

class Dataset(models.Model):
    source = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='datasets')
    metadata = models.JSONField(default=dict, blank=True)  # e.g., sample rate, vendor, notes
    created_at = models.DateTimeField(auto_now_add=True)

class ModelVersion(models.Model):
    name = models.CharField(max_length=64)                 # e.g., 'classifier' or 'autoencoder'
    version = models.CharField(max_length=32)              # e.g., '1.0.0'
    path = models.CharField(max_length=512)                # SavedModel dir
    metrics = models.JSONField(default=dict, blank=True)   # accuracy, loss, etc.
    preprocessing = models.JSONField(default=dict, blank=True)  # grid, mean, std
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

class Prediction(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='predictions')
    model_version = models.ForeignKey(ModelVersion, on_delete=models.PROTECT, related_name='predictions')
    predicted_class = models.IntegerField(null=True, blank=True)
    probabilities = models.JSONField(default=dict, blank=True)   # class → prob
    saliency_path = models.CharField(max_length=512, blank=True) # optional saved .npy
    created_at = models.DateTimeField(auto_now_add=True)