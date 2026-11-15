"""Local Windows try-on service with DeCo-VTON and an explicit legacy option."""
import os
import sys
import io
import secrets
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('AIFIT_EXISTING_PROJECT', r'E:\Comsats\8th Semester\Fyp 2\Virtual-Try-On-main'))
DECO_CHECKPOINT = ROOT / '.models' / 'deco'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(ROOT / '.preprocess-deps'))
sys.path.insert(0, str(PROJECT / 'CatVTON'))
sys.path.insert(0, str(PROJECT / 'ai_engine' / 'clothes'))
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, request, jsonify, send_file
from huggingface_hub import snapshot_download
from clothing_mask import ClothingMask, restore_crop
from preprocessing import PhotoPreprocessor, InputRejected

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024
lock = threading.Lock()
pipeline = None
clothing_mask = None
preprocessor = None
backend_name = None
infer_size = (768, 1024)
keypath = ROOT / 'instance' / 'gpu.key'
keypath.parent.mkdir(exist_ok=True)
if not keypath.exists():
    with keypath.open('x') as f:
        f.write(secrets.token_urlsafe(40))
KEY = keypath.read_text().strip()


def _deco_paths():
    vendor = str(SERVICE / 'vendor')
    for path in (vendor,):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_clothing_mask():
    global clothing_mask
    checkpoint = snapshot_download('zhengchong/CatVTON', local_files_only=True)
    clothing_mask = ClothingMask(Path(checkpoint) / 'SCHP/exp-schp-201908261155-lip.pth')
    return checkpoint


def _load_deco():
    global pipeline, backend_name, infer_size
    weights = DECO_CHECKPOINT / 'unet' / 'diffusion_pytorch_model.safetensors'
    if not weights.exists():
        raise RuntimeError('DeCo-VTON weights are missing. Run gpu_service/download_deco.py first.')
    original_path = list(sys.path)
    try:
        _deco_paths()
        from deco_backend import DeCoBackend
        pipeline = DeCoBackend(DECO_CHECKPOINT)
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        backend_name = 'deco'
        infer_size = DeCoBackend.resolution
        print('Model ready: DeCo-VTON 384 x 512, 50 steps, safety checker enabled.', flush=True)
        return True
    except Exception:
        sys.path[:] = original_path
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        raise


def _load_catvton(checkpoint):
    global pipeline, backend_name, infer_size
    from model.pipeline import CatVTONPipeline
    from safetensors.torch import load_file
    pipeline = CatVTONPipeline(
        base_ckpt='runwayml/stable-diffusion-inpainting', attn_ckpt=checkpoint,
        attn_ckpt_version='mix', weight_dtype=torch.float16, device='cuda', use_tf32=True)
    trained = load_file(str(Path(checkpoint) / 'mix-48k-1024/attention/model.safetensors'))
    current = pipeline.attn_modules.state_dict()
    unexpected = set(trained) - set(current)
    mismatches = [
        name for name in trained
        if name in current and not torch.equal(current[name].detach().cpu(), trained[name].to(dtype=current[name].dtype))
    ]
    aliases = dict(pipeline.attn_modules.named_parameters(remove_duplicate=False))
    covered = {id(aliases[name]) for name in trained if name in aliases}
    uncovered = {id(p) for p in pipeline.attn_modules.parameters()} - covered
    if uncovered or unexpected or mismatches:
        raise RuntimeError('CatVTON attention weights do not match the loaded pipeline.')
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    original_checker = pipeline.run_safety_checker

    def checked(*args, **kwargs):
        result, flags = original_checker(*args, **kwargs)
        pipeline.last_safety_blocked = bool(flags is not None and any(flags))
        return result, flags

    pipeline.run_safety_checker = checked
    backend_name = 'catvton'
    infer_size = (768, 1024)
    print('Model ready: CatVTON 768 x 1024, 50 steps, fp16, safety checker enabled.', flush=True)


def initialize():
    global clothing_mask, preprocessor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    print('Loading try-on weights on ' + torch.cuda.get_device_name(0), flush=True)
    torch.set_num_threads(4)
    preprocessor = PhotoPreprocessor(ROOT / '.models/preprocessing')
    checkpoint = _load_clothing_mask()
    engine = os.environ.get('AIFIT_ENGINE', 'deco')
    if engine == 'deco':
        _load_deco()
    elif engine == 'catvton':
        _load_catvton(checkpoint)
    else:
        raise ValueError('AIFIT_ENGINE must be deco or catvton')


@app.get('/health')
def health():
    return jsonify(
        ready=pipeline is not None and preprocessor is not None and clothing_mask is not None,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        backend=backend_name,
        resolution=list(infer_size),
        pose_validation=preprocessor is not None,
        pose_preservation="landmark-protected inpainting and output check",
        garment_segmentation=preprocessor is not None,
    ), 200 if pipeline is not None and preprocessor is not None and clothing_mask is not None else 503


@app.post('/infer')
def infer():
    from deco_backend import OutputBlocked
    if not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + KEY):
        return jsonify(error='Unauthorized'), 401
    category = request.form.get('category', 'upper')
    if category not in ('upper', 'lower', 'overall'):
        return jsonify(error='Invalid category'), 400
    if not lock.acquire(False):
        return jsonify(error='GPU busy. Retry shortly.'), 429
    try:
        if pipeline is None or preprocessor is None or clothing_mask is None:
            return jsonify(error='The try-on service is still starting. Retry shortly.'), 503
        images = []
        for name in ('person', 'garment'):
            file = request.files.get(name)
            if not file:
                return jsonify(error='Missing image'), 400
            im = Image.open(file.stream)
            if im.format not in ('JPEG', 'PNG') or im.width * im.height > 20_000_000:
                return jsonify(error='Invalid image'), 400
            im.load()
            images.append(ImageOps.exif_transpose(im).convert('RGBA' if name == 'garment' else 'RGB'))
        try:
            landmarks, foreground = preprocessor.validate_person(images[0], category, include_foreground=True)
            clean_garment = preprocessor.clean_garment(images[1])
            if preprocessor.garment_has_person(clean_garment):
                clean_garment = clothing_mask.garment_only(clean_garment, category)
            person, mask, crop_box, protected = clothing_mask.prepare(
                images[0], category, infer_size, landmarks=landmarks, include_protection=True, foreground=foreground)
        except ValueError as error:
            return jsonify(error=str(error)), 422
        garment = ImageOps.pad(clean_garment, infer_size, method=Image.Resampling.LANCZOS, color='white')
        with torch.inference_mode():
            torch.manual_seed(42)
            generator = torch.Generator(device='cuda').manual_seed(42)
            output = pipeline(
                image=person, condition_image=garment, mask=mask,
                width=infer_size[0], height=infer_size[1],
                num_inference_steps=50, guidance_scale=2.5, generator=generator)[0]
        if getattr(pipeline, 'last_safety_blocked', False):
            raise OutputBlocked('The generated image could not be returned. Try a different photo or garment.')
        output = restore_crop(output, images[0], mask, crop_box, protected=protected)
        if category == 'upper':
            output = clothing_mask.restore_exposed_skin(images[0], output)
        preprocessor.check_output_pose(landmarks, output, category)
        result = io.BytesIO()
        output.save(result, format='JPEG', quality=95)
        result.seek(0)
        return send_file(result, mimetype='image/jpeg')
    except (OutputBlocked, InputRejected) as error:
        return jsonify(error=str(error)), 422
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify(error='The image could not be read. Upload a valid JPEG or PNG.'), 400
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return jsonify(error='GPU memory exhausted. Close other GPU applications.'), 503
    except RuntimeError as error:
        # Some CUDA kernels report allocation failure as RuntimeError rather
        # than torch.cuda.OutOfMemoryError. Keep other failures as HTTP 500.
        message = str(error).lower()
        if 'cuda' in message and 'out of memory' in message:
            torch.cuda.empty_cache()
            return jsonify(error='GPU memory exhausted. Close other GPU applications and retry.'), 503
        app.logger.exception('Inference runtime failed')
        return jsonify(error='Inference failed; inspect local GPU log'), 500
    except Exception:
        app.logger.exception('Inference failed')
        return jsonify(error='Inference failed; inspect local GPU log'), 500
    finally:
        lock.release()


if __name__ == '__main__':
    initialize()
    app.run(host='127.0.0.1', port=8001, threaded=True)
