import os
import sys
import json
import subprocess
import tempfile

# ---------------------------------------------------------------------------
# Modell-Aliases die Training unterstützen
# ---------------------------------------------------------------------------
TRAIN_MODELS = [
    # Z-Image
    "z-image-base",
    "z-image-turbo",
    # FLUX.2 Klein Base (Edit-Training nur für diese)
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
]

TRAIN_QUANTIZE = ["None", "4", "8"]

OPTIMIZER_NAMES = ["AdamW", "Adam", "SGD"]

# Standardwerte je Modell
MODEL_DEFAULTS = {
    "z-image-base":        {"steps": 50, "guidance": 5.0, "quantize": "8"},
    "z-image-turbo":       {"steps": 9,  "guidance": 0.0, "quantize": "8"},
    "flux2-klein-base-4b": {"steps": 40, "guidance": 1.0, "quantize": "None"},
    "flux2-klein-base-9b": {"steps": 40, "guidance": 1.0, "quantize": "None"},
}

# LoRA-Targets je Modell (sinnvolle Defaults aus der mflux Dokumentation)
LORA_TARGETS_ZIMAGE = [
    {"module_path": "layers.{block}.attention.to_q", "blocks": {"start": 15, "end": 30}, "rank": 8},
    {"module_path": "layers.{block}.attention.to_k", "blocks": {"start": 15, "end": 30}, "rank": 8},
    {"module_path": "layers.{block}.attention.to_v", "blocks": {"start": 15, "end": 30}, "rank": 8},
]

LORA_TARGETS_FLUX2 = [
    {"module_path": "transformer_blocks.{block}.attn.to_q",          "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_k",          "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_v",          "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_out",        "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.add_q_proj",    "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.add_k_proj",    "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.add_v_proj",    "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_add_out",    "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff.linear_in",       "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff.linear_out",      "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff_context.linear_in",  "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff_context.linear_out", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj", "blocks": {"start": 0, "end": 20}, "rank": 16},
    {"module_path": "single_transformer_blocks.{block}.attn.to_out",          "blocks": {"start": 0, "end": 20}, "rank": 16},
]


def _get_lora_targets(model: str) -> list:
    if "z-image" in model:
        return LORA_TARGETS_ZIMAGE
    else:
        return LORA_TARGETS_FLUX2


# ---------------------------------------------------------------------------
# Node: MfluxTrainConfigBuilder
# Erzeugt eine train.json und gibt den Pfad weiter
# ---------------------------------------------------------------------------
class MfluxTrainConfigBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Basis ────────────────────────────────────────────────────
                "model": (TRAIN_MODELS, {
                    "default": "z-image-turbo",
                    "tooltip": "Model to train. Edit mode is auto-detected from data folder layout.",
                }),
                "data_path": ("STRING", {
                    "default": "/path/to/training/images",
                    "tooltip": "Folder with training images and .txt prompt files. "
                               "For edit mode: use *_in.png / *_out.png / *_in.txt pairs.",
                }),
                "output_path": ("STRING", {
                    "default": "~/training_output",
                    "tooltip": "Where to save checkpoints, loss plots and preview images.",
                }),
                "config_save_path": ("STRING", {
                    "default": "~/train_config.json",
                    "tooltip": "Where to save the generated train.json. "
                               "This path is passed to MfluxTrainer.",
                }),

                # ── Model settings ───────────────────────────────────────────
                "quantize": (TRAIN_QUANTIZE, {
                    "default": "4",
                    "tooltip": "Quantization for training. None = full precision (more VRAM).",
                }),
                "steps": ("INT", {"default": 4, "min": 1, "max": 200,
                                  "tooltip": "Inference steps used during training previews."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "max_resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64,
                                           "tooltip": "Maximum image resolution for training."}),
                "low_ram": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),

                # ── Training loop ────────────────────────────────────────────
                "num_epochs": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "timestep_low": ("INT", {"default": 1, "min": 0, "max": 1000,
                                         "tooltip": "Lower bound of timestep range for training."}),
                "timestep_high": ("INT", {"default": 4, "min": 0, "max": 1000,
                                          "tooltip": "Upper bound of timestep range for training."}),

                # ── Optimizer ────────────────────────────────────────────────
                "optimizer": (OPTIMIZER_NAMES, {"default": "AdamW"}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-7, "max": 1e-1,
                                            "step": 1e-5}),

                # ── Checkpoint & Monitoring ───────────────────────────────────
                "save_frequency": ("INT", {"default": 30, "min": 1,
                                           "tooltip": "Save a checkpoint every N steps."}),
                "plot_frequency": ("INT", {"default": 1,
                                           "tooltip": "Update loss plot every N steps."}),
                "generate_image_frequency": ("INT", {"default": 30,
                                                     "tooltip": "Generate preview image every N steps. 0 = disabled."}),

                # ── LoRA layers ───────────────────────────────────────────────
                "lora_rank": ("INT", {"default": 16, "min": 1, "max": 128,
                                      "tooltip": "LoRA rank for all target layers. "
                                                 "Higher = more parameters, more expressive."}),
                "lora_block_start": ("INT", {"default": 0, "min": 0, "max": 100,
                                             "tooltip": "First transformer block to apply LoRA to."}),
                "lora_block_end": ("INT", {"default": 5, "min": 1, "max": 100,
                                           "tooltip": "Last transformer block to apply LoRA to (exclusive)."}),
            },
            "optional": {
                "config_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Alternative: path to an existing train.json. "
                               "If set, all other inputs are ignored and this file is used directly.",
                }),
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a local model folder. Leave empty to use HuggingFace download.",
                }),
                "custom_lora_targets_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Override LoRA targets with custom JSON array. "
                               "Leave empty to use model-appropriate defaults.",
                }),
            },
        }

    RETURN_TYPES = ("MFLUX_TRAIN_CONFIG",)
    RETURN_NAMES = ("train_config",)
    CATEGORY = "MFlux/Train"
    FUNCTION = "build_config"
    OUTPUT_NODE = True

    def build_config(self, model, data_path, output_path, config_save_path,
                     quantize, steps, guidance, seed, max_resolution, low_ram,
                     num_epochs, batch_size, timestep_low, timestep_high,
                     optimizer, learning_rate, save_frequency, plot_frequency,
                     generate_image_frequency, lora_rank, lora_block_start,
                     lora_block_end, config_json_path="", model_path="",
                     custom_lora_targets_json=""):

        # ── Passthrough: existing JSON-Datei direkt verwenden ───────────────
        if config_json_path and config_json_path.strip():
            path = os.path.expanduser(config_json_path.strip())
            if not os.path.exists(path):
                raise ValueError(f"[MfluxTrain] config_json_path not found: {path}")
            print(f"[MfluxTrain] Using existing config: {path}")
            return (path,)

        # ── LoRA targets ────────────────────────────────────────────────────
        if custom_lora_targets_json and custom_lora_targets_json.strip():
            try:
                lora_targets = json.loads(custom_lora_targets_json.strip())
            except json.JSONDecodeError as e:
                raise ValueError(f"[MfluxTrain] Invalid custom_lora_targets_json: {e}")
        else:
            # Defaults aus der Dokumentation, rank-Override
            base_targets = _get_lora_targets(model)
            lora_targets = []
            for t in base_targets:
                target = dict(t)
                target["rank"] = lora_rank
                target["blocks"] = {"start": lora_block_start, "end": lora_block_end}
                lora_targets.append(target)

        # ── Config zusammenbauen ─────────────────────────────────────────────
        config = {
            "model": model,
            "data": os.path.expanduser(data_path),
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "quantize": None if quantize == "None" else int(quantize),
            "low_ram": low_ram,
            "max_resolution": max_resolution,
            "training_loop": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "timestep_low": timestep_low,
                "timestep_high": timestep_high,
            },
            "optimizer": {
                "name": optimizer,
                "learning_rate": learning_rate,
            },
            "checkpoint": {
                "output_path": os.path.expanduser(output_path),
                "save_frequency": save_frequency,
            },
            "monitoring": {
                "plot_frequency": plot_frequency,
                "generate_image_frequency": generate_image_frequency,
            },
            "lora_layers": {
                "targets": lora_targets,
            },
        }

        # Optionaler model_path
        if model_path and model_path.strip():
            config["model_path"] = os.path.expanduser(model_path.strip())

        # ── JSON speichern ───────────────────────────────────────────────────
        save_path = os.path.expanduser(config_save_path.strip())
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"[MfluxTrain] Config saved to: {save_path}")
        print(f"[MfluxTrain] Config:\n{json.dumps(config, indent=2)}")

        return (save_path,)


# ---------------------------------------------------------------------------
# Node: MfluxTrainer
# Startet mflux-train mit einer Config-JSON oder setzt ein Training fort
# ---------------------------------------------------------------------------
class MfluxTrainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["start", "resume"], {
                    "default": "start",
                    "tooltip": "'start' requires train_config. 'resume' requires checkpoint_path.",
                }),
            },
            "optional": {
                "train_config": ("MFLUX_TRAIN_CONFIG", {
                    "tooltip": "Connect MfluxTrainConfigBuilder output here, or use config_json_path.",
                }),
                "config_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Alternative: path to an existing train.json file. "
                               "Used if train_config is not connected.",
                }),
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a checkpoint .zip file for resuming training.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_log",)
    CATEGORY = "MFlux/Train"
    FUNCTION = "train"
    OUTPUT_NODE = True

    def train(self, mode, train_config=None, config_json_path="", checkpoint_path=""):
        if mode == "resume":
            if not checkpoint_path or not checkpoint_path.strip():
                raise ValueError(
                    "[MfluxTrainer] Resume mode requires a checkpoint_path "
                    "(path to a .zip checkpoint file)."
                )
            cp = os.path.expanduser(checkpoint_path.strip())
            if not os.path.exists(cp):
                raise ValueError(f"[MfluxTrainer] Checkpoint not found: {cp}")
            cmd = [sys.executable, "-m", "mflux.train", "--resume", cp]
            print(f"[MfluxTrainer] Resuming from: {cp}")

        else:  # start
            # Config-Pfad bestimmen: train_config hat Vorrang
            if train_config and str(train_config).strip():
                config_path = str(train_config).strip()
            elif config_json_path and config_json_path.strip():
                config_path = os.path.expanduser(config_json_path.strip())
            else:
                raise ValueError(
                    "[MfluxTrainer] Start mode requires either a connected train_config "
                    "or a config_json_path."
                )
            if not os.path.exists(config_path):
                raise ValueError(f"[MfluxTrainer] Config file not found: {config_path}")

            cmd = [sys.executable, "-m", "mflux.train", "--config", config_path]
            print(f"[MfluxTrainer] Starting training with config: {config_path}")

        print(f"[MfluxTrainer] Command: {' '.join(cmd)}")
        print("[MfluxTrainer] Training output will appear in the ComfyUI terminal.")
        print("[MfluxTrainer] This may take a long time depending on dataset size and epochs.")

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,   # Live-Output im ComfyUI Terminal
                text=True,
                check=True,
            )
            msg = f"[MfluxTrainer] Training completed successfully."
            print(msg)
            return (msg,)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"[MfluxTrainer] Training failed (exit {e.returncode}). "
                f"Check the ComfyUI terminal for details."
            ) from e
