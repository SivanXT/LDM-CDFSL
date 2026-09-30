import torch

from utils.config import pretrained_models
from utils.utils import maybe_data_parallel


def load_language_models(device):
    try:
        import clip
        from transformers import AutoProcessor, BlipForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Language-model dependencies are missing. Install requirements.txt first."
        ) from error

    processor = AutoProcessor.from_pretrained(pretrained_models["blip"])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    captioner = BlipForConditionalGeneration.from_pretrained(
        pretrained_models["blip"], torch_dtype=dtype
    ).to(device)
    text_encoder, _ = clip.load(pretrained_models["clip"], device=device)

    captioner.eval()
    text_encoder.eval()
    for model in (captioner, text_encoder):
        for parameter in model.parameters():
            parameter.requires_grad = False

    return processor, maybe_data_parallel(captioner, device), maybe_data_parallel(text_encoder, device)


def move_episode(img_s, mask_s, img_q, mask_q, device):
    img_s = img_s.permute(1, 0, 2, 3, 4)
    mask_s = mask_s.permute(1, 0, 2, 3)
    support_images = [images.to(device) for images in img_s]
    support_masks = [masks.to(device) for masks in mask_s]
    return support_images, support_masks, img_q.to(device), mask_q.to(device)


def restore_class_ids(prediction, target, class_ids, num_classes):
    if num_classes == 1:
        return prediction, target
    prediction = prediction.clone()
    target = target.clone()
    for index, class_id in enumerate(class_ids):
        label = int(class_id) + 1
        prediction[index][prediction[index] == 1] = label
        target[index][target[index] == 1] = label
    return prediction, target
