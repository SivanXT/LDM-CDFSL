import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def state_dict_for_save(model):
    return unwrap_model(model).state_dict()


def load_checkpoint(model, checkpoint_path, device):
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    legacy_names = {"alpha_s", "alpha_q"}
    legacy_parts = (
        "cross_attention.text_projection.",
        "cross_attention.attention.",
        "cross_attention.conv_out.",
    )
    legacy_keys = sorted(
        key for key in checkpoint
        if key.rsplit(".", 1)[-1] in legacy_names
        or any(part in key for part in legacy_parts)
    )
    if legacy_keys:
        examples = ", ".join(legacy_keys[:3])
        raise RuntimeError(
            "This checkpoint uses a different Feature Refiner parameter structure "
            f"(detected: {examples}) and is incompatible with the current model. "
            "Run source-domain training with this code, then use the generated checkpoint."
        )
    model.load_state_dict(checkpoint)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def maybe_data_parallel(model, device):
    if (
        device.type == "cuda"
        and device.index is None
        and torch.cuda.device_count() > 1
    ):
        return torch.nn.DataParallel(model)
    return model


class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.intersection = 0
        self.union = 0

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        if self.num_classes > 1:
            for prediction, target in zip(predictions, gts):
                self.hist += self._fast_hist(prediction.flatten(), target.flatten())
        else:
            for prediction, target in zip(predictions, gts):
                self.intersection += np.logical_and(prediction.flatten(), target.flatten()).sum()
                self.union += np.logical_or(prediction.flatten(), target.flatten()).sum()

    def evaluate(self):
        if self.num_classes > 1:
            denominator = self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist)
            with np.errstate(divide="ignore", invalid="ignore"):
                iu = np.diag(self.hist) / denominator
            return np.nanmean(iu)
        return self.intersection / self.union if self.union > 0 else 0


@torch.no_grad()
def get_text_embedding(prompt, device, model):
    """Return frozen CLIP token features as [batch, tokens, width]."""
    import clip

    clip_model = unwrap_model(model)
    tokenized = clip.tokenize([prompt], truncate=True).to(device)
    dtype = clip_model.token_embedding.weight.dtype
    valid_tokens = tokenized.ne(0)
    if (valid_tokens.sum(dim=-1) < 2).any():
        raise RuntimeError("CLIP tokenization must preserve non-padding SOT and EOT tokens.")
    features = clip_model.token_embedding(tokenized).to(dtype)
    features = features + clip_model.positional_embedding.to(dtype)
    features = features.permute(1, 0, 2)
    features = clip_model.transformer(features)
    features = features.permute(1, 0, 2)
    features = clip_model.ln_final(features).float()
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    features = features.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
    return features


def _processor_to_device(processed, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return {
        key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        for key, value in processed.items()
    }


@torch.no_grad()
def batch_im_processor_pil_voc(image_paths, blip_processor, device):
    return batch_im_processor_pil(image_paths, blip_processor, device)


@torch.no_grad()
def batch_im_processor_pil(image_paths, blip_processor, device):
    processed_images = []
    for shot in image_paths:
        shot_images = []
        for image_path in shot:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            with Image.open(path) as image:
                processed = blip_processor(image.convert("RGB"), return_tensors="pt")
            shot_images.append(_processor_to_device(processed, device))
        processed_images.append(shot_images)
    return processed_images


@torch.no_grad()
def generate_and_embed(
    image_inputs,
    blip_processor,
    blip_generator,
    clip_encoder,
    device,
    domain_prompts,
    dataset,
    generate_kwargs,
):
    caption_model = unwrap_model(blip_generator)
    text_embeddings = []
    for shot in image_inputs:
        shot_embeddings = []
        for image_input in shot:
            generated = caption_model.generate(**image_input, **generate_kwargs)
            caption = blip_processor.decode(generated[0], skip_special_tokens=True)
            prompt = f"content: {caption}; domain attributes: {domain_prompts[dataset]}"
            prompt = prompt.replace("\n", " ").replace("\t", " ")
            shot_embeddings.append(get_text_embedding(prompt, device, clip_encoder))
        text_embeddings.append(torch.cat(shot_embeddings, dim=0))
    return text_embeddings
