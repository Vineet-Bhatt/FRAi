from rest_framework import serializers

class UploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    file_type = serializers.CharField(required=False)  # optional
    baseline = serializers.BooleanField(required=False, default=False)  # if baseline upload
