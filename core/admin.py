from django.contrib import admin
from core.models import UploadedFile, Dataset, ModelVersion, Prediction


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ("id", "file", "file_type", "created_at")
    search_fields = ("file", "file_type")
    list_filter = ("file_type", "created_at")


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "created_at")
    search_fields = ("id",)
    list_filter = ("created_at",)


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "version", "is_active", "created_at")
    list_editable = ("is_active",)
    search_fields = ("name", "version")
    list_filter = ("name", "is_active", "created_at")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "model_version", "predicted_class", "created_at")
    search_fields = ("id",)
    list_filter = ("created_at",)

from django.contrib import admin

# Register your models here.
