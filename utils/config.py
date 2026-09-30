domain_prompts = {
    "pascal": "natural images with diverse objects and backgrounds",
    "lung": "frontal chest X-ray images with grayscale anatomical structures",
    "deepglobe": "satellite images with overhead land-cover patterns",
    "fss": "natural images spanning diverse object categories and backgrounds",
    "isic": "dermoscopic skin-lesion images with clinical acquisition characteristics",
}


dataset_num_classes = {
    "fss": 1001,
    "deepglobe": 7,
    "isic": 4,
    "lung": 1,
}


generate_kwargs = {
    "max_length": 75,
    "num_beams": 20,
    "temperature": 0.7,
    "repetition_penalty": 100.0,
    "early_stopping": True,
}


pretrained_models = {
    "blip": "Salesforce/blip-image-captioning-base",
    "clip": "ViT-B/32",
}
