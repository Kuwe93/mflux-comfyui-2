import random
import json
import os
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
 
# ---------------------------------------------------------------------------
# mflux 0.16.x Imports
# Alle Imports die über FLUX.1 hinausgehen werden defensiv behandelt,
# da mflux regelmäßig neue Modellklassen hinzufügt und die Pfade sich ändern.
# ---------------------------------------------------------------------------
 
# ── Immer verfügbar (FLUX.1-Kern) ──────────────────────────────────────────
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.utils.image_util import ImageUtil
from mflux.models.flux.variants.controlnet.controlnet_util import ControlnetUtil
 
# ── FLUX.1-Varianten (seit ~0.9) ───────────────────────────────────────────
try:
    from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
    HAS_FILL = True
except ImportError:
    HAS_FILL = False
    print("[Mflux] Flux1Fill not available")
 
try:
    from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
    HAS_DEPTH = True
except ImportError:
    HAS_DEPTH = False
    print("[Mflux] Flux1Depth not available")
 
try:
    from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
    HAS_REDUX = True
except ImportError:
    HAS_REDUX = False
    print("[Mflux] Flux1Redux not available")
 
try:
    from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
    HAS_KONTEXT = True
except ImportError:
    HAS_KONTEXT = False
    print("[Mflux] Flux1Kontext not available")
 
# ── FLUX.2-Familie (zukünftig) ──────────────────────────────────────────────
try:
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    HAS_FLUX2 = True
except ImportError:
    HAS_FLUX2 = False
    print("[Mflux] Flux2Klein not available")
 
# ── Z-Image-Familie ─────────────────────────────────────────────────────────
try:
    from mflux.models.z_image.variants.z_image import ZImage
    HAS_ZIMAGE = True
except (ImportError, AttributeError):
    HAS_ZIMAGE = False
    print("[Mflux] ZImage not available")
 
# ── FLUX.2 Edit ──────────────────────────────────────────────────────────────
try:
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    HAS_FLUX2_EDIT = True
except (ImportError, AttributeError):
    HAS_FLUX2_EDIT = False
    print("[Mflux] Flux2KleinEdit not available")
 
# ── FLUX.1 In-Context ────────────────────────────────────────────────────────
try:
    from mflux.models.flux.variants.in_context.flux_in_context_dev import FluxInContextDev
    from mflux.models.flux.variants.in_context.flux_in_context_fill import FluxInContextFill
    HAS_IN_CONTEXT = True
except (ImportError, AttributeError):
    HAS_IN_CONTEXT = False
    print("[Mflux] FluxInContext not available")
 
# ── Fibo (Bria) ──────────────────────────────────────────────────────────────
try:
    from mflux.models.fibo.variants.txt2img.fibo import Fibo
    HAS_FIBO = True
except (ImportError, AttributeError):
    HAS_FIBO = False
    print("[Mflux] Fibo not available")
 
# ── SeedVR2 (Upscaler) ───────────────────────────────────────────────────────
try:
    from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
    HAS_SEEDVR2 = True
except (ImportError, AttributeError):
    HAS_SEEDVR2 = False
    print("[Mflux] SeedVR2 not available")
 
# ── Qwen-Familie ────────────────────────────────────────────────────────────
try:
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("[Mflux] Qwen not available")
 
from .Mflux_Pro import MfluxControlNetPipeline
 
# ---------------------------------------------------------------------------
# Modell-Familien-Zuordnung
# ---------------------------------------------------------------------------
MODEL_FAMILY_MAP = {
    # FLUX.1 – immer verfügbar
    "schnell":        ("flux1", "schnell"),
    "dev":            ("flux1", "dev"),
    "krea-dev":       ("flux1", "krea-dev"),
    "kontext-dev":    ("flux1", "kontext-dev"),
    "flux2-klein-4b": ("flux2", "flux2-klein-4b"), 
    "flux2-klein-9b": ("flux2", "flux2-klein-9b"),
}
 
if HAS_ZIMAGE:
    MODEL_FAMILY_MAP.update({
        "z-image-turbo": ("zimage", "z-image-turbo"),
        "z-image-base":  ("zimage", "z-image-base"),
    })
 
if HAS_QWEN:
    MODEL_FAMILY_MAP["qwen-image"] = ("qwen", "qwen-image")
 
if HAS_FIBO:
    MODEL_FAMILY_MAP["fibo"] = ("fibo", "fibo")
 
if HAS_SEEDVR2:
    MODEL_FAMILY_MAP["seedvr2"] = ("seedvr2", "seedvr2")
 
# Distilled FLUX.2-Modelle benötigen guidance=1.0
FLUX2_DISTILLED_MODELS = {"flux2-klein-4b", "flux2-klein-9b"}
 
# Modelle mit nativer negative_prompt Unterstützung (CFG-basiert)
# Z-Image-Turbo ist bewusst ausgeschlossen (distilliert, kein CFG)
NEGATIVE_PROMPT_MODELS = {"zimage", "qwen"}
 
# Exports für __init__.py und Mflux_Air.py
__all_flags__ = [
    "HAS_FILL", "HAS_DEPTH", "HAS_REDUX", "HAS_KONTEXT", "HAS_QWEN",
    "HAS_FLUX2", "HAS_FLUX2_EDIT", "HAS_ZIMAGE", "HAS_IN_CONTEXT",
    "HAS_FIBO", "HAS_SEEDVR2",
]
 
# Alle Quantisierungsoptionen
ALL_QUANTIZE_OPTIONS = ["None", "3", "4", "5", "6", "8"]
 
# ---------------------------------------------------------------------------
# Model-Cache
# ---------------------------------------------------------------------------
_model_cache: dict = {}
 
 
def _evict_and_store(key, instance):
    _model_cache.clear()
    _model_cache[key] = instance
    return instance
 
 
# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
 
def get_lora_info(Loras):
    if Loras:
        return Loras.lora_paths, Loras.lora_scales
    return [], []
 
 
def resolve_model_alias(model: str, local_path: str) -> tuple[str, str]:
    if local_path:
        name_lower = local_path.lower()
        for alias, (family, _) in MODEL_FAMILY_MAP.items():
            if alias in name_lower or alias.replace("-", "_") in name_lower:
                return family, alias
        name = os.path.basename(local_path).lower()
        if "flux2" in name or "klein" in name:
            return ("flux2" if HAS_FLUX2 else "flux1"), model
        if "z-image" in name or "zimage" in name:
            return ("zimage" if HAS_ZIMAGE else "flux1"), model
        if "qwen" in name:
            return ("qwen" if HAS_QWEN else "flux1"), model
        return "flux1", model
    if model in MODEL_FAMILY_MAP:
        return MODEL_FAMILY_MAP[model]
    return "flux1", model
 
 
def _tensor_from_image(generated) -> torch.Tensor:
    if hasattr(generated, "image"):
        arr = np.array(generated.image).astype(np.float32) / 255.0
    elif isinstance(generated, np.ndarray):
        arr = generated.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    else:
        arr = np.array(generated).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    t = torch.from_numpy(arr)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t
 
 
def _save_temp_image(tensor: torch.Tensor) -> str:
    import tempfile
    arr = (tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name
 
 
# ---------------------------------------------------------------------------
# Modell laden / aus Cache holen
# ---------------------------------------------------------------------------
 
def load_or_create_model(model, quantize, local_path, lora_paths, lora_scales,
                         variant=""):
    family, alias = resolve_model_alias(model, local_path)
    q = None if quantize == "None" else int(quantize)
    path = local_path if local_path else None
    key = (family, alias, q, path or "", tuple(lora_paths), tuple(lora_scales), variant)
 
    print(f"[Mflux] --- MODEL ---")
    if key in _model_cache:
        print(f"[Mflux] Using CACHED model: {family} ({alias})")
        return _model_cache[key]
    
    if path:
        print(f"[Mflux] Loading LOCAL model from path: {path}")
    else:
        print(f"[Mflux] Loading INTEGRATED model using alias: {alias}")
    
    print(f"[Mflux] Model Family: {family} | Quantization: {quantize}bit | Variant: {variant if variant else 'standard'}")
    print(f"[Mflux] --- MODEL END ---")
    
    # Basis-Konfiguration
    m_config = ModelConfig.from_name(alias)
 
    # Wir bauen die Argumente dynamisch zusammen
    # mflux 0.16.x: Flux1 nutzt local_path, ZImage/Flux2 nutzen model_path
    common_args = {
        "model_config": m_config,
        "quantize": q,
        "lora_paths": lora_paths,
        "lora_scales": lora_scales,
    }
 
    if family == "flux1":
        # Flux1 erwartet 'model_path'
        args = {**common_args, "model_path": path}
        if variant == "controlnet":
            inst = Flux1Controlnet(**args)
        elif variant == "fill":
            if not HAS_FILL:
                raise RuntimeError("Flux1Fill not available.")
            inst = Flux1Fill(quantize=q, model_path=path, lora_paths=lora_paths, lora_scales=lora_scales)
        elif variant == "depth":
            if not HAS_DEPTH:
                raise RuntimeError("Flux1Depth not available.")
            inst = Flux1Depth(quantize=q, model_path=path, lora_paths=lora_paths, lora_scales=lora_scales)
        elif variant == "redux":
            if not HAS_REDUX:
                raise RuntimeError("Flux1Redux not available.")
            inst = Flux1Redux(quantize=q, model_path=path, lora_paths=lora_paths, lora_scales=lora_scales)
        elif variant == "kontext":
            if not HAS_KONTEXT:
                raise RuntimeError("Flux1Kontext not available.")
            inst = Flux1Kontext(**args)
        else:
            inst = Flux1(**args)
 
    elif family == "flux2":
        if not HAS_FLUX2:
            raise RuntimeError("Flux2Klein not available.")
        flux2_args = {**common_args, "model_path": path}
        inst = Flux2Klein(**flux2_args)
 
    elif family == "zimage":
        # ZImage erwartet ebenfalls 'model_path'
        args = {**common_args, "model_path": path}
        inst = ZImage(**args)
 
    elif family == "qwen":
        args = {"quantize": q, "model_path": path, "lora_paths": lora_paths, "lora_scales": lora_scales}
        if variant == "edit":
            inst = QwenImageEdit(**args)
        else:
            inst = QwenImage(**args)
    elif family == "fibo":
        if not HAS_FIBO:
            raise RuntimeError("Fibo not available in this mflux version.")
        inst = Fibo(**{**common_args, "model_path": path})
 
    elif family == "seedvr2":
        if not HAS_SEEDVR2:
            raise RuntimeError("SeedVR2 not available in this mflux version.")
        inst = SeedVR2(**{**common_args, "model_path": path})
 
    else:
        # Fallback auf Flux1
        inst = Flux1(**{**common_args, "model_path": path})
 
    return _evict_and_store(key, inst)
 
 
# ---------------------------------------------------------------------------
# Generierungsfunktionen
# ---------------------------------------------------------------------------
 
def generate_image(prompt, model, seed, width, height, steps, guidance,
                   quantize="None", metadata=True, Local_model="",
                   image=None, Loras=None, ControlNet=None,
                   negative_prompt=""):
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux] seed={seed}")
    lora_paths, lora_scales = get_lora_info(Loras)
    family, alias = resolve_model_alias(model, Local_model)
    image_path = image.image_path if image else None
    image_strength = image.image_strength if image else None
    use_controlnet = ControlNet is not None and isinstance(ControlNet, MfluxControlNetPipeline)
 
    if alias in FLUX2_DISTILLED_MODELS:
        guidance = 1.0
 
    if use_controlnet:
        from mflux.post_processing.array_util import ArrayUtil
        import mlx.core as mx
        from tqdm import tqdm
        import comfy.utils as utils
        inst = load_or_create_model(model, quantize, Local_model, lora_paths, lora_scales,
                                    variant="controlnet")
        from mflux.config.config import ConfigControlnet
        from mflux.config.runtime_config import RuntimeConfig
        config = RuntimeConfig(
            config=ConfigControlnet(
                num_inference_steps=steps, height=height, width=width,
                guidance=guidance, controlnet_strength=ControlNet.control_strength,
            ),
            model_config=inst.model_config,
        )
        time_steps = tqdm(range(config.num_inference_steps))
        control_image = ImageUtil.load_image(ControlNet.control_image_path)
        control_image = ControlnetUtil.scale_image(config.height, config.width, control_image)
        control_image = ControlnetUtil.preprocess_canny(control_image)
        controlnet_cond = ImageUtil.to_array(control_image)
        controlnet_cond = inst.vae.encode(controlnet_cond)
        controlnet_cond = (controlnet_cond / inst.vae.scaling_factor) + inst.vae.shift_factor
        controlnet_cond = ArrayUtil.pack_latents(controlnet_cond, config.height, config.width)
        latents = FluxLatentCreator.create(seed=seed, height=config.height, width=config.width)
        t5_tokens = inst.t5_tokenizer.tokenize(prompt)
        clip_tokens = inst.clip_tokenizer.tokenize(prompt)
        prompt_embeds = inst.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = inst.clip_text_encoder.forward(clip_tokens)
        pbar = None
        for gen_step, t in enumerate(time_steps, 1):
            if gen_step == 2:
                pbar = utils.ProgressBar(total=steps)
            cb_samples, cb_single = inst.transformer_controlnet.forward(
                t=t, prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents, controlnet_cond=controlnet_cond, config=config,
            )
            noise = inst.transformer.predict(
                t=t, prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents, config=config,
                controlnet_block_samples=cb_samples,
                controlnet_single_block_samples=cb_single,
            )
            latents += noise * (config.sigmas[t + 1] - config.sigmas[t])
            if pbar:
                pbar.update(1)
            mx.eval(latents)
        if pbar:
            pbar.update(1)
        latents = ArrayUtil.unpack_latents(latents, config.height, config.width)
        decoded = inst.vae.decode(latents)
        arr = ImageUtil._to_numpy(ImageUtil._denormalize(decoded))
        tensor = torch.from_numpy(arr)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return (tensor,)
 
    inst = load_or_create_model(model, quantize, Local_model, lora_paths, lora_scales)
 
    gen_kwargs = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
        "guidance": guidance,
    }
 
    # negative_prompt nur für Modelle mit CFG-Unterstützung
    # Bei zimage zusätzlich alias prüfen: Turbo hat kein CFG, Base schon
    if negative_prompt and negative_prompt.strip():
        if family == "qwen" or (family == "zimage" and alias == "z-image-base"):
            gen_kwargs["negative_prompt"] = negative_prompt.strip()
 
    # img2img params
    if image_path:
        gen_kwargs["init_image_path"] = image_path
        gen_kwargs["init_image_strength"] = image_strength
 
    generated = inst.generate_image(**gen_kwargs)
    return (_tensor_from_image(generated),)
 
 
def generate_fill(prompt, seed, width, height, steps, guidance, quantize,
                  image_path, mask_path, Local_model="", Loras=None):
    if not HAS_FILL:
        raise RuntimeError("Fill/Inpainting needs Flux1Fill, not available in this mflux-version.")
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("dev", quantize, Local_model, lora_paths, lora_scales, variant="fill")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path, mask_path=mask_path,
    )
    return (_tensor_from_image(generated),)
 
 
def generate_depth(prompt, seed, width, height, steps, guidance, quantize,
                   image_path, Local_model="", Loras=None):
    if not HAS_DEPTH:
        raise RuntimeError("Depth Conditioning needs Flux1Depth, not available in this mflux-Version.")
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("dev", quantize, Local_model, lora_paths, lora_scales, variant="depth")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path,
    )
    return (_tensor_from_image(generated),)
 
 
def generate_redux(seed, width, height, steps, guidance, quantize,
                   image_path, Local_model=""):
    if not HAS_REDUX:
        raise RuntimeError("Redux needs Flux1Redux, not available in this mflux-Version.")
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    inst = load_or_create_model("dev", quantize, Local_model, [], [], variant="redux")
    generated = inst.generate_image(
        seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path,
    )
    return (_tensor_from_image(generated),)
 
 
def generate_kontext(prompt, seed, width, height, steps, guidance, quantize,
                     image_path, Local_model="", Loras=None):
    if not HAS_KONTEXT:
        raise RuntimeError("Kontext needs Flux1Kontext, not available in this mflux-Version.")
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("kontext-dev", quantize, Local_model, lora_paths, lora_scales,
                                variant="kontext")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path,
    )
    return (_tensor_from_image(generated),)
 
 
def generate_qwen(prompt, negative_prompt, seed, width, height, steps, guidance,
                  quantize, Local_model="", Loras=None):
    if not HAS_QWEN:
        raise RuntimeError("Qwen needs QwenImage, not available in this mflux-Version.")
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("qwen-image", quantize, Local_model, lora_paths, lora_scales)
    generated = inst.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else " ",
        seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
    )
    return (_tensor_from_image(generated),)
 
 
def generate_qwen_edit(prompt, seed, width, height, steps, guidance, quantize,
                       image_paths: list, Local_model="", Loras=None):
    if not HAS_QWEN:
        raise RuntimeError("Qwen Edit needs QwenImageEdit, not available in this mflux-Version.")
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("qwen-image", quantize, Local_model, lora_paths, lora_scales,
                                variant="edit")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_paths=image_paths,
    )
    return (_tensor_from_image(generated),)
 
 
# ---------------------------------------------------------------------------
# Metadaten speichern
# ---------------------------------------------------------------------------
 
def save_images_with_metadata(
    images, prompt, model, quantize, Local_model,
    seed, height, width, steps, guidance,
    lora_paths, lora_scales, image_path, image_strength,
    filename_prefix="Mflux", full_prompt=None, extra_pnginfo=None,
    extra_meta=None,
):
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
    )
    mflux_output_folder = os.path.join(full_output_folder, "MFlux")
    os.makedirs(mflux_output_folder, exist_ok=True)
 
    existing_counters = [
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(mflux_output_folder)
        if f.startswith(filename_prefix) and f.endswith(".png")
    ]
    counter = max(existing_counters, default=0) + 1
 
    results = []
    for image in images:
        i = 255.0 * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
 
        png_meta = None
        if full_prompt is not None or extra_pnginfo is not None:
            png_meta = PngInfo()
            if full_prompt is not None:
                png_meta.add_text("full_prompt", json.dumps(full_prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    png_meta.add_text(x, json.dumps(extra_pnginfo[x]))
 
        image_file = f"{filename_prefix}_{counter:05}.png"
        img.save(os.path.join(mflux_output_folder, image_file), pnginfo=png_meta, compress_level=4)
        results.append({"filename": image_file, "subfolder": subfolder, "type": "output"})
 
        family, alias = resolve_model_alias(model, Local_model)
        json_dict = {
            "prompt": prompt, "model": model, "model_family": family,
            "model_alias": alias, "quantize": quantize, "seed": seed,
            "height": height, "width": width, "steps": steps,
            "guidance": 1.0 if alias in FLUX2_DISTILLED_MODELS else guidance,
            "Local_model": Local_model, "init_image_path": image_path,
            "init_image_strength": image_strength,
            "lora_paths": lora_paths, "lora_scales": lora_scales,
        }
        if extra_meta:
            json_dict.update(extra_meta)
 
        with open(os.path.join(mflux_output_folder, f"{filename_prefix}_{counter:05}.json"), "w") as f:
            json.dump(json_dict, f, indent=4)
        counter += 1
 
    return {"ui": {"images": results}, "counter": counter}
 