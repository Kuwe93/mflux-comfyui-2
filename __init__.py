from .Mflux_Comfy.Mflux_Core import (
    HAS_FILL, HAS_DEPTH, HAS_REDUX, HAS_KONTEXT, HAS_QWEN,
)
from .Mflux_Comfy.Mflux_Air import (
    QuickMfluxNode,
    MfluxModelsLoader,
    MfluxModelsDownloader,
    MfluxCustomModels,
)
from .Mflux_Comfy.Mflux_Pro import (
    MfluxImg2Img,
    MfluxLorasLoader,
    MfluxControlNetLoader,
    MfluxFillLoader,
    MfluxImageRefLoader,
    MfluxScaleFactor,
    MfluxMetadataLoader,
)
from .Mflux_Comfy.Mflux_Train import (
    MfluxTrainConfigBuilder,
    MfluxTrainer,
)

# ---------------------------------------------------------------------------
# Basis-Nodes – immer verfügbar
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "QuickMfluxNode":           QuickMfluxNode,
    "MfluxModelsLoader":        MfluxModelsLoader,
    "MfluxModelsDownloader":    MfluxModelsDownloader,
    "MfluxCustomModels":        MfluxCustomModels,
    "MfluxImg2Img":             MfluxImg2Img,
    "MfluxLorasLoader":         MfluxLorasLoader,
    "MfluxControlNetLoader":    MfluxControlNetLoader,
    "MfluxFillLoader":          MfluxFillLoader,
    "MfluxImageRefLoader":      MfluxImageRefLoader,
    "MfluxScaleFactor":         MfluxScaleFactor,
    "MfluxMetadataLoader":      MfluxMetadataLoader,
    "MfluxTrainConfigBuilder":  MfluxTrainConfigBuilder,
    "MfluxTrainer":             MfluxTrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickMfluxNode":           "Quick MFlux Generation",
    "MfluxModelsLoader":        "MFlux Models Loader",
    "MfluxModelsDownloader":    "MFlux Models Downloader",
    "MfluxCustomModels":        "MFlux Custom Models",
    "MfluxImg2Img":             "MFlux Img2Img",
    "MfluxLorasLoader":         "MFlux LoRAs Loader",
    "MfluxControlNetLoader":    "MFlux ControlNet Loader",
    "MfluxFillLoader":          "MFlux Fill Loader",
    "MfluxImageRefLoader":      "MFlux Image Ref Loader",
    "MfluxScaleFactor":         "MFlux Scale Factor",
    "MfluxMetadataLoader":      "MFlux Metadata Loader",
    "MfluxTrainConfigBuilder":  "MFlux Train Config Builder",
    "MfluxTrainer":             "MFlux Trainer",
}

# ---------------------------------------------------------------------------
# Optionale Nodes – nur registrieren wenn mflux-Klasse verfügbar
# ---------------------------------------------------------------------------
if HAS_FILL:
    from .Mflux_Comfy.Mflux_Air import MfluxFillNode
    NODE_CLASS_MAPPINGS["MfluxFillNode"] = MfluxFillNode
    NODE_DISPLAY_NAME_MAPPINGS["MfluxFillNode"] = "MFlux Fill (Inpainting)"

if HAS_DEPTH:
    from .Mflux_Comfy.Mflux_Air import MfluxDepthNode
    NODE_CLASS_MAPPINGS["MfluxDepthNode"] = MfluxDepthNode
    NODE_DISPLAY_NAME_MAPPINGS["MfluxDepthNode"] = "MFlux Depth Conditioning"

if HAS_REDUX:
    from .Mflux_Comfy.Mflux_Air import MfluxReduxNode
    NODE_CLASS_MAPPINGS["MfluxReduxNode"] = MfluxReduxNode
    NODE_DISPLAY_NAME_MAPPINGS["MfluxReduxNode"] = "MFlux Redux (Image Variation)"

if HAS_KONTEXT:
    from .Mflux_Comfy.Mflux_Air import MfluxKontextNode
    NODE_CLASS_MAPPINGS["MfluxKontextNode"] = MfluxKontextNode
    NODE_DISPLAY_NAME_MAPPINGS["MfluxKontextNode"] = "MFlux Kontext (Image Editing)"

if HAS_FLUX2_EDIT:
    from .Mflux_Comfy.Mflux_Air import MfluxFlux2EditNode
    NODE_CLASS_MAPPINGS["MfluxFlux2EditNode"] = MfluxFlux2EditNode
    NODE_DISPLAY_NAME_MAPPINGS["MfluxFlux2EditNode"] = "MFlux FLUX.2 Edit (Multi-Image)"

if HAS_SEEDVR2:
    from .Mflux_Comfy.Mflux_Air import MfluxSeedVR2Node
    NODE_CLASS_MAPPINGS["MfluxSeedVR2Node"] = MfluxSeedVR2Node
    NODE_DISPLAY_NAME_MAPPINGS["MfluxSeedVR2Node"] = "MFlux SeedVR2 Upscaler"

if HAS_QWEN:
    from .Mflux_Comfy.Mflux_Air import MfluxQwenNode, MfluxQwenEditNode
    NODE_CLASS_MAPPINGS["MfluxQwenNode"] = MfluxQwenNode
    NODE_CLASS_MAPPINGS["MfluxQwenEditNode"] = MfluxQwenEditNode
    NODE_DISPLAY_NAME_MAPPINGS["MfluxQwenNode"] = "MFlux Qwen Image"
    NODE_DISPLAY_NAME_MAPPINGS["MfluxQwenEditNode"] = "MFlux Qwen Image Edit"