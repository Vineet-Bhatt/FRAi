from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import numpy as np
import uuid

# Use non-interactive backend for server-side rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ingestion.parser import parse_csv, parse_xml
from ingestion.preprocessoring import resample_to_grid, smooth, normalize
from ml.inference import load_model, predict_classifier, saliency_map


_MODEL = None
_MODEL_INPUT_LEN = None


def _ensure_model_loaded():
    global _MODEL, _MODEL_INPUT_LEN
    if _MODEL is not None:
        return
    # Try default classifier path under MODEL_DIR
    model_dir = getattr(settings, 'MODEL_DIR', None)
    if not model_dir:
        return
    candidate = os.path.join(model_dir, 'classifier_savedmodel')
    if os.path.exists(candidate):
        try:
            _MODEL = load_model(candidate)
            # Expect input shape (None, L, 1)
            _MODEL_INPUT_LEN = int(_MODEL.input_shape[1]) if _MODEL.input_shape else None
        except Exception:
            _MODEL = None
            _MODEL_INPUT_LEN = None


def home(request):
    _ensure_model_loaded()
    if request.method == 'GET':
        return render(request, 'dashboard/home.html', {
            'model_loaded': _MODEL is not None,
        })

    # POST: handle upload and analyze
    uploaded = request.FILES.get('file')
    if not uploaded:
        return render(request, 'dashboard/home.html', {
            'error': 'Please select a file to upload.',
            'model_loaded': _MODEL is not None,
        })

    # Save file to media/uploads
    rel_path = os.path.join('uploads', uploaded.name)
    abs_path = default_storage.save(rel_path, ContentFile(uploaded.read()))
    full_path = os.path.join(settings.MEDIA_ROOT, abs_path)

    # Detect type and parse
    lower = uploaded.name.lower()
    freqs, mags = None, None
    try:
        if lower.endswith('.csv'):
            freqs, mags = parse_csv(full_path)
        elif lower.endswith('.xml'):
            freqs, mags = parse_xml(full_path)
        else:
            # Fallback: not supported, just return
            return render(request, 'dashboard/result.html', {
                'file_name': uploaded.name,
                'note': 'Unsupported file type for parsing. Upload CSV or XML.',
            })
    except Exception as exc:
        return render(request, 'dashboard/result.html', {
            'file_name': uploaded.name,
            'note': f'Failed to parse file: {exc}',
        })

    # Basic preprocessing to a fixed grid if we have a model
    prediction = None
    probs = None
    saliency = None
    model_note = None
    plot_url = None

    try:
        if _MODEL is not None and _MODEL_INPUT_LEN:
            # Build uniform grid across observed freqs
            fmin = float(np.min(freqs))
            fmax = float(np.max(freqs))
            target_grid = np.linspace(fmin, fmax, _MODEL_INPUT_LEN)
            mags_on_grid = resample_to_grid(freqs, mags, target_grid)
            mags_on_grid = smooth(mags_on_grid)
            mags_on_grid = normalize(mags_on_grid)

            pred_idx, pred_probs = predict_classifier(_MODEL, mags_on_grid)
            prediction = int(pred_idx)
            probs = [float(x) for x in pred_probs]
            try:
                sal = saliency_map(_MODEL, mags_on_grid, class_index=pred_idx)
                saliency = [float(x) for x in sal]
            except Exception:
                saliency = None
        else:
            model_note = 'Model not loaded. Place SavedModel at MODEL_DIR/classifier_savedmodel.'
    except Exception as exc:
        model_note = f'Inference failed: {exc}'

    # Prepare small preview of parsed data
    n = min(10, len(freqs))
    preview_points = [[float(freqs[i]), float(mags[i])] for i in range(n)]

    # Create and save plot image under media/plots
    try:
        plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        img_name = f"fra_{uuid.uuid4().hex}.png"
        img_path = os.path.join(plots_dir, img_name)

        plt.figure(figsize=(8, 4.5), dpi=150)
        try:
            # Try semilog-x; fall back to linear if values invalid
            plt.semilogx(freqs, mags, color='#1f77b4')
        except Exception:
            plt.plot(freqs, mags, color='#1f77b4')
        plt.title('Frequency Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        # Build URL for template
        plot_url = os.path.join(settings.MEDIA_URL, 'plots', img_name)
    except Exception:
        plot_url = None

    return render(request, 'dashboard/result.html', {
        'file_name': uploaded.name,
        'preview_points': preview_points,
        'prediction': prediction,
        'probabilities': probs,
        'saliency': saliency,
        'model_loaded': _MODEL is not None,
        'model_note': model_note,
        'plot_url': plot_url,
    })


def landing(request):
    return render(request, 'dashboard/index.html')