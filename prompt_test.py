import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from prompt_train import text_features
from utils.config import dataset_num_classes
from utils.runtime import load_language_models, move_episode, restore_class_ids
from utils.utils import count_params, load_checkpoint, mIOU, maybe_data_parallel, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CD-FSS checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="root containing target datasets")
    parser.add_argument("--dataset", choices=["fss", "deepglobe", "isic", "lung"], default="fss")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outdir/predictions"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--backbone-weights", type=Path, default=None)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def save_predictions(query_names, predictions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for query_name, prediction in zip(query_names, predictions.cpu().numpy()):
        source = Path(query_name)
        digest = hashlib.sha1(str(source).encode()).hexdigest()[:8]
        destination = output_dir / f"{source.stem}_{digest}_prediction.png"
        image = Image.fromarray(prediction.astype(np.uint16))
        image.save(destination)


@torch.no_grad()
def evaluate(model, dataloader, args, processor, captioner, text_encoder, device, output_dir):
    model.eval()
    metric = mIOU(dataset_num_classes[args.dataset])
    progress = tqdm(dataloader)
    for img_s, mask_s, img_q, mask_q, class_ids, support_names, query_names in progress:
        img_s, mask_s, img_q, mask_q = move_episode(img_s, mask_s, img_q, mask_q, device)
        support_text, query_text = text_features(
            support_names, query_names, processor, captioner, text_encoder, device, args.dataset
        )
        prediction = model(img_s, mask_s, img_q, mask_q, support_text, query_text)[0].argmax(1)
        prediction, mask_q = restore_class_ids(
            prediction, mask_q, class_ids, dataset_num_classes[args.dataset]
        )
        metric.add_batch(prediction.cpu().numpy(), mask_q.cpu().numpy())
        save_predictions(query_names, prediction, output_dir)
        progress.set_description(f"Test mIoU: {metric.evaluate() * 100:.2f}")
    return metric.evaluate() * 100


def main():
    args = parse_args()
    from data.dataset import FSSDataset
    from model.prompt_matching import Prompt_MatchingNet

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = resolve_device(args.device)
    processor, captioner, text_encoder = load_language_models(device)

    FSSDataset.initialize(img_size=400, datapath=str(args.data_root))
    testloader = FSSDataset.build_dataloader(
        args.dataset, args.batch_size, args.num_workers, 0, args.split, args.shot
    )
    model = Prompt_MatchingNet(
        backbone=args.backbone,
        refine=args.refine,
        shot=args.shot,
        backbone_weights=args.backbone_weights,
    )
    load_checkpoint(model, args.checkpoint, device)
    model = maybe_data_parallel(model.to(device), device)
    print(f"Loaded {args.checkpoint}; model parameters: {count_params(model):.1f}M")

    scores = []
    for run in range(args.runs):
        set_seed(args.seed + run)
        run_dir = args.output_dir / args.dataset / f"seed_{args.seed + run}"
        scores.append(evaluate(
            model, testloader, args, processor, captioner, text_encoder, device, run_dir
        ))
    average = sum(scores) / len(scores)
    print(f"Averaged mIoU over {args.runs} run(s): {average:.2f}")


if __name__ == "__main__":
    main()
