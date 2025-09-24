from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction

from ingestion.serializer import UploadSerializer
from ingestion.parser import parse_csv, parse_xml
from core.models import UploadedFile, Dataset


class UploadView(APIView):
    def post(self, request):
        serializer = UploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        uploaded = serializer.validated_data['file']
        file_type = serializer.validated_data.get('file_type')

        # Infer file type by extension if not provided
        if not file_type and hasattr(uploaded, 'name'):
            lower = uploaded.name.lower()
            if lower.endswith('.csv'):
                file_type = 'csv'
            elif lower.endswith('.xml'):
                file_type = 'xml'
            elif lower.endswith('.bin') or lower.endswith('.dat'):
                file_type = 'bin'

        if file_type not in {'csv', 'xml', 'bin'}:
            return Response({'detail': 'Unsupported or unknown file_type.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with transaction.atomic():
                uf = UploadedFile(file_type=file_type)
                # Save file into FileField storage
                uf.file.save(uploaded.name, uploaded, save=True)

                ds = Dataset.objects.create(source=uf, metadata={})

        except Exception as exc:
            return Response({'detail': f'Failed to save file: {exc}'}, status=status.HTTP_400_BAD_REQUEST)

        # Try parsing for a small preview (non-fatal if parse fails)
        preview = None
        try:
            if file_type == 'csv':
                freqs, mags = parse_csv(uf.file.path)
                n = min(5, len(freqs))
                preview = {
                    'points': [[float(freqs[i]), float(mags[i])] for i in range(n)]
                }
            elif file_type == 'xml':
                freqs, mags = parse_xml(uf.file.path)
                n = min(5, len(freqs))
                preview = {
                    'points': [[float(freqs[i]), float(mags[i])] for i in range(n)]
                }
            else:
                preview = {'note': 'Binary preview not implemented'}
        except Exception:
            preview = {'note': 'Preview unavailable'}

        return Response({
            'uploaded_file_id': uf.id,
            'dataset_id': ds.id,
            'file_name': uf.file.name,
            'file_type': uf.file_type,
            'preview': preview,
        }, status=status.HTTP_201_CREATED)


