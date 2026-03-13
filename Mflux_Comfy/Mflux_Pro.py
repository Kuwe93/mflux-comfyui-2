import os
import tempfile
import numpy as np
import torch
from PIL import Image
import folder_paths

from mflux.models.flux.variants.controlnet.controlnet_util import ControlnetUtil


# ---------------------------------------------------------------------------
# Pipeline-Classes
# ---------------------------------------------------------------------------

class MfluxImg2ImgPipeline:
    def __init__(self, image_path: str, image_strength: float):
        self.image_path = image_path
        self.image_strength = image_strength

    def clear_cache(self):
        self.image_path = None
        self.image_strength = None


class MfluxLorasPipeline:
    def __init__(self, lora_paths: list, lora_scales: list):
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

    def clear_cache(self):
        self.lora_paths = []
        self.lora_scales = []


class MfluxControlNetPipeline:
    def __init__(self, model_selection: str, control_image_path: str,
                 control_strength: float, save_canny: bool = False):
        self.model_selection = model_selection
        self.control_image_path = control_image_path
        self.control_strength = control_strength
        self.save_canny = save_canny

    def clear_cache(self):
        self.model_selection = None
        self.control_image_path = None
        self.control_strength = None
        self.save_canny = False


class MfluxFillPipeline:
    def __init__(self, image_path: str, mask_path: str):
        self.image_path = image_path
        self.mask_path = mask_path

    def clear_cache(self):
        self.image_path = None
        self.mask_path = None


class MfluxImageRefPipeline:
    def __init__(self, image_paths: list):
        self.image_paths = image_paths  # immer eine Liste

    @property
    def image_path(self):
        return self.image_paths[0] if self.image_paths else None

    def clear_cache(self):
        self.image_paths = []


# ---------------------------------------------------------------------------
# Hilfsfunktion: ComfyUI-Tensor → temp PNG
# ---------------------------------------------------------------------------

def _tensor_to_temp_path(tensor: torch.Tensor) -> str:
    arr = (tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Node: MfluxImg2Img
# ---------------------------------------------------------------------------

class MfluxImg2Img:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image_file": (sorted(files), {
                    "image_upload": True,
                    "default": sorted(files)[0] if files else "example.png",
                    "tooltip": "Upload an image file. Ignored when image_tensor is connected.",
                }),
                "image_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image_tensor": ("IMAGE", {
                    "tooltip": "Connect any IMAGE output here to override the uploaded file.",
                }),
            },
        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("MfluxImg2ImgPipeline", "INT", "INT")
    RETURN_NAMES = ("img2img", "width", "height")
    FUNCTION = "load_and_process"

    def load_and_process(self, image_file, image_strength, image_tensor=None):
        if image_tensor is not None:
            image_path = _tensor_to_temp_path(image_tensor)
            print("[MfluxImg2Img] Using tensor from another node.")
        else:
            if not folder_paths.exists_annotated_filepath(image_file):
                raise ValueError(
                    f"[MfluxImg2Img] Image file not found: {image_file}. "
                    "Please upload an image or connect an IMAGE tensor."
                )
            image_path = folder_paths.get_annotated_filepath(image_file)
            print("[MfluxImg2Img] Using uploaded image file.")

        with Image.open(image_path) as img:
            width, height = img.size

        return MfluxImg2ImgPipeline(image_path, image_strength), width, height

    @classmethod
    def IS_CHANGED(cls, image_file, image_strength, image_tensor=None):
        return (hash(image_file), round(float(image_strength), 2))


# ---------------------------------------------------------------------------
# Node: MfluxLorasLoader
# ---------------------------------------------------------------------------

class MfluxLorasLoader:
    @classmethod
    def INPUT_TYPES(cls):
        loras_relative = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "Lora1":  (loras_relative,),
                "scale1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Lora2":  (loras_relative,),
                "scale2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Lora3":  (loras_relative,),
                "scale3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline",),
                "hf_lora": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace LoRA: 'author/repo' oder 'author/repo:filename.safetensors'",
                }),
            },
        }

    RETURN_TYPES = ("MfluxLorasPipeline",)
    RETURN_NAMES = ("Loras",)
    FUNCTION = "lora_stacker"
    CATEGORY = "MFlux/Pro"

    def lora_stacker(self, Loras=None, hf_lora="", **kwargs):
        lora_base_path = folder_paths.models_dir
        lora_models = [
            (os.path.join(lora_base_path, "loras", kwargs[f"Lora{i}"]), kwargs[f"scale{i}"])
            for i in range(1, 4)
            if kwargs.get(f"Lora{i}") not in (None, "None")
        ]

        # HuggingFace-LoRA direkt als Pfad-String übergeben
        if hf_lora and hf_lora.strip():
            lora_models.append((hf_lora.strip(), 1.0))

        # Verkettete LoRAs anhängen
        if Loras is not None and isinstance(Loras, MfluxLorasPipeline):
            if Loras.lora_paths and Loras.lora_scales:
                lora_models.extend(zip(Loras.lora_paths, Loras.lora_scales))

        if lora_models:
            lora_paths, lora_scales = zip(*lora_models)
        else:
            lora_paths, lora_scales = [], []

        return (MfluxLorasPipeline(list(lora_paths), list(lora_scales)),)


# ---------------------------------------------------------------------------
# Node: MfluxControlNetLoader  (Canny)
# ---------------------------------------------------------------------------

CONTROLNET_MODELS = ["InstantX/FLUX.1-dev-Controlnet-Canny"]


class MfluxControlNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "model_selection": (CONTROLNET_MODELS, {"default": CONTROLNET_MODELS[0]}),
                "image":           (sorted(files), {"image_upload": True}),
                "control_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "save_canny":      ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
            }
        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("MfluxControlNetPipeline", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("ControlNet", "width", "height", "preprocessed_image")
    FUNCTION = "load_and_select"

    def load_and_select(self, model_selection, image, control_strength, save_canny):
        control_image_path = folder_paths.get_annotated_filepath(image)
        with Image.open(control_image_path) as img:
            width, height = img.size
            canny_image = ControlnetUtil.preprocess_canny(img)
            canny_np = np.array(canny_image).astype(np.float32) / 255.0

        canny_tensor = torch.from_numpy(canny_np)
        if canny_tensor.dim() == 3:
            canny_tensor = canny_tensor.unsqueeze(0)

        return (
            MfluxControlNetPipeline(model_selection, control_image_path, control_strength, save_canny),
            width, height, canny_tensor,
        )

    @classmethod
    def IS_CHANGED(cls, model_selection, image, control_strength, save_canny):
        return str(hash(image)) + model_selection + str(round(float(control_strength), 2)) + str(save_canny)

    @classmethod
    def VALIDATE_INPUTS(cls, image, model_selection, control_strength, save_canny):
        if model_selection not in CONTROLNET_MODELS:
            return f"Invalid model selection: {model_selection}"
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid control image file: {image}"
        if not isinstance(control_strength, (int, float)):
            return "Strength must be a number"
        if not isinstance(save_canny, bool):
            return "save_canny must be a boolean value"
        return True


# ---------------------------------------------------------------------------
# Node: MfluxFillLoader  (Inpainting)
# ---------------------------------------------------------------------------

class MfluxFillLoader:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True,
                                          "tooltip": "Original image, which gets the inpainting."}),
                "mask":  (sorted(files), {"image_upload": True,
                                          "tooltip": "Mask (white = generate, black = stays original)."}),
            },
            "optional": {
                "mask_tensor": ("MASK", {"tooltip": "alternative: mask as ComfyUI-Mask-Tensor."}),
            }
        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("MfluxFillPipeline", "INT", "INT")
    RETURN_NAMES = ("fill", "width", "height")
    FUNCTION = "load"

    def load(self, image, mask, mask_tensor=None):
        image_path = folder_paths.get_annotated_filepath(image)
        with Image.open(image_path) as img:
            width, height = img.size

        if mask_tensor is not None:
            # ComfyUI-Mask-Tensor → temporäre PNG
            mask_arr = (mask_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_arr, mode="L")
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            mask_img.save(tmp.name)
            mask_path = tmp.name
        else:
            mask_path = folder_paths.get_annotated_filepath(mask)

        return MfluxFillPipeline(image_path, mask_path), width, height

    @classmethod
    def IS_CHANGED(cls, image, mask, mask_tensor=None):
        return (hash(image), hash(mask))

    @classmethod
    def VALIDATE_INPUTS(cls, image, mask, mask_tensor=None):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        if mask_tensor is None and not folder_paths.exists_annotated_filepath(mask):
            return f"Invalid mask file: {mask}"
        return True


# ---------------------------------------------------------------------------
# Node: MfluxImageRefLoader  (Kontext / Depth / Redux / Qwen-Edit)
# ---------------------------------------------------------------------------

class MfluxImageRefLoader:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = ["None"] + [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image1": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "image2": (sorted(files), {"image_upload": True}),
                "image3": (sorted(files), {"image_upload": True}),
                "image4": (sorted(files), {"image_upload": True}),
                "image_tensor1": ("IMAGE", {"tooltip": "alternative: Image from other node."}),
                "image_tensor2": ("IMAGE", {"tooltip": "alternative: Image from other node."}),
            }
        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("MfluxImageRefPipeline", "INT", "INT")
    RETURN_NAMES = ("image_ref", "width", "height")
    FUNCTION = "load"

    def load(self, image1, image2="None", image3="None", image4="None",
             image_tensor1=None, image_tensor2=None):
        paths = []

        # Datei-Uploads
        for img_name in [image1, image2, image3, image4]:
            if img_name and img_name != "None":
                paths.append(folder_paths.get_annotated_filepath(img_name))

        # Tensor-Eingaben
        for tensor in [image_tensor1, image_tensor2]:
            if tensor is not None:
                paths.append(_tensor_to_temp_path(tensor))

        if not paths:
            raise ValueError("MfluxImageRefLoader: at least one Image required.")

        with Image.open(paths[0]) as img:
            width, height = img.size

        return MfluxImageRefPipeline(paths), width, height

    @classmethod
    def IS_CHANGED(cls, image1, **kwargs):
        return hash(image1)


# ---------------------------------------------------------------------------
# Node: MfluxScaleFactor 
# ---------------------------------------------------------------------------

class MfluxScaleFactor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 2.0, "min": 0.25, "max": 8.0, "step": 0.25,
                    "tooltip": "z.B. 2.0 = double resolution, 0.5 = half resolution",
                }),
                "round_to":     (["8", "16", "64"], {"default": "64",
                                  "tooltip": "Round to which basenumber? (FLUX needs at least 16)."}),
            }
        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "compute"

    def compute(self, image, scale_factor, round_to):
        _, h, w, _ = image.shape
        r = int(round_to)
        new_w = max(r, round(w * scale_factor / r) * r)
        new_h = max(r, round(h * scale_factor / r) * r)
        print(f"[MfluxScaleFactor] {w}×{h} × {scale_factor} → {new_w}×{new_h}")
        return new_w, new_h


# ---------------------------------------------------------------------------
# Node: MfluxMetadataLoader
# Lädt eine .metadata.json Sidecar-Datei und gibt alle Parameter als
# Node-Outputs weiter – direkt anschließbar an QuickMfluxNode.
# Einzelne Werte können überschrieben werden (z.B. anderer Prompt).
# ---------------------------------------------------------------------------

class MfluxMetadataLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata_json_path": ("STRING", {
                    "default": "~/image.metadata.json",
                    "tooltip": "Pfad zur .metadata.json Datei die mflux neben dem Bild ablegt "
                               "(z.B. image.metadata.json).",
                }),
            },
            "optional": {
                "override_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Überschreibt den Prompt aus der Metadata. Leer = Metadata-Prompt verwenden.",
                }),
                "override_seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Überschreibt den Seed. -1 = Seed aus der Metadata verwenden.",
                }),
                "override_steps": ("INT", {
                    "default": -1, "min": -1, "max": 200,
                    "tooltip": "Überschreibt die Steps. -1 = Steps aus der Metadata verwenden.",
                }),
                "override_guidance": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Überschreibt Guidance. -1.0 = Metadata-Wert verwenden.",
                }),
                "override_width": ("INT", {
                    "default": -1, "min": -1, "max": 4096,
                    "tooltip": "Überschreibt Width. -1 = Metadata-Wert verwenden.",
                }),
                "override_height": ("INT", {
                    "default": -1, "min": -1, "max": 4096,
                    "tooltip": "Überschreibt Height. -1 = Metadata-Wert verwenden.",
                }),
            },
        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT",   "INT",   "INT",   "FLOAT",    "INT")
    RETURN_NAMES = ("prompt", "model",  "quantize","seed", "width", "height","guidance", "steps")
    FUNCTION = "load_metadata"

    def load_metadata(self, metadata_json_path,
                      override_prompt="", override_seed=-1, override_steps=-1,
                      override_guidance=-1.0, override_width=-1, override_height=-1):
        import json as _json

        path = os.path.expanduser(metadata_json_path.strip())
        if not os.path.exists(path):
            raise ValueError(f"[MfluxMetadataLoader] Datei nicht gefunden: {path}")

        with open(path) as f:
            meta = _json.load(f)

        print(f"[MfluxMetadataLoader] Loaded: {path}")
        print(f"[MfluxMetadataLoader] Metadata: {_json.dumps(meta, indent=2)}")

        # Werte aus Metadata lesen (mit sinnvollen Fallbacks)
        prompt   = str(meta.get("prompt",   ""))
        model    = str(meta.get("model",    "schnell"))
        quantize = str(meta.get("quantize", "4"))
        seed     = int(meta.get("seed",     -1))
        width    = int(meta.get("width",    512))
        height   = int(meta.get("height",   512))
        guidance = float(meta.get("guidance", 3.5))
        steps    = int(meta.get("steps",    4))

        # Overrides anwenden
        if override_prompt and override_prompt.strip():
            prompt = override_prompt.strip()
            print(f"[MfluxMetadataLoader] Prompt overridden.")
        if override_seed != -1:
            seed = override_seed
        if override_steps != -1:
            steps = override_steps
        if override_guidance != -1.0:
            guidance = override_guidance
        if override_width != -1:
            width = override_width
        if override_height != -1:
            height = override_height

        print(f"[MfluxMetadataLoader] Final: model={model}, seed={seed}, "
              f"{width}×{height}, steps={steps}, guidance={guidance}")

        return (prompt, model, quantize, seed, width, height, guidance, steps)

    @classmethod
    def IS_CHANGED(cls, metadata_json_path, **kwargs):
        path = os.path.expanduser(metadata_json_path.strip())
        if os.path.exists(path):
            return os.path.getmtime(path)
        return float("nan")