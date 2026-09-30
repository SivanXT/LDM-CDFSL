import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from prompt_train import evaluate, freeze_backbone_stem, freeze_batch_norm, text_features
from utils.runtime import load_language_models, move_episode
from utils.utils import (
    count_params,
    load_checkpoint,
    maybe_data_parallel,
    resolve_device,
    set_seed,
    state_dict_for_save,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot fine-tuning on a CD-FSS target domain")
    parser.add_argument("--data-root", type=Path, required=True, help="root containing target datasets")
    parser.add_argument("--dataset", choices=["fss", "deepglobe", "isic", "lung"], default="fss")
    parser.add_argument("--checkpoint", type=Path, required=True, help="stage-one checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outdir/models"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--backbone-weights", type=Path, default=None)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--episode", type=int, default=36000)
    parser.add_argument("--snapshot", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def main():
    args = parse_args()
    from data.dataset import FSSDataset
    from model.prompt_matching import Prompt_MatchingNet

    if args.episode < args.snapshot:
        raise ValueError("--episode must be greater than or equal to --snapshot")
    if not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir / args.dataset / "ifa"
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, captioner, text_encoder = load_language_models(device)
    FSSDataset.initialize(img_size=400, datapath=str(args.data_root))
    trainloader = FSSDataset.build_dataloader(
        f"{args.dataset}ifa", args.batch_size, args.num_workers, 0, "val", args.shot
    )
    testloader = FSSDataset.build_dataloader(
        args.dataset, args.batch_size, args.num_workers, 0, "val", args.shot
    )

    model = Prompt_MatchingNet(
        backbone=args.backbone,
        refine=args.refine,
        shot=args.shot,
        backbone_weights=args.backbone_weights,
    )
    load_checkpoint(model, args.checkpoint, device)
    freeze_backbone_stem(model)
    print(f"Loaded {args.checkpoint}; model parameters: {count_params(model):.1f}M")
    model = maybe_data_parallel(model.to(device), device)

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=5e-4,
    )
    total_iters = args.episode // args.batch_size
    lr_decay_iters = {total_iters // 3, total_iters * 2 // 3}
    epochs = args.episode // args.snapshot
    iterations = 0
    best_score = float("-inf")
    best_model = None

    for epoch in range(epochs):
        set_seed(args.seed + epoch)
        model.train()
        freeze_batch_norm(model)
        total_loss = 0.0
        progress = tqdm(trainloader)
        for index, (img_s, mask_s, img_q, mask_q, _, support_names, query_names) in enumerate(progress):
            img_s, mask_s, img_q, mask_q = move_episode(img_s, mask_s, img_q, mask_q, device)
            support_text, query_text = text_features(
                support_names, query_names, processor, captioner, text_encoder,
                device, args.dataset,
            )
            outputs = model(img_s, mask_s, img_q, mask_q, support_text, query_text)
            support_mask = torch.cat(mask_s, dim=0).long()
            if args.refine:
                loss = (criterion(outputs[0], mask_q) + criterion(outputs[1], mask_q)
                        + criterion(outputs[2], mask_q) + 0.2 * criterion(outputs[3], support_mask)
                        + 0.4 * criterion(outputs[4], support_mask)
                        + 0.1 * criterion(outputs[5], mask_q)
                        + 0.1 * criterion(outputs[6], support_mask)
                        + 0.1 * criterion(outputs[7], mask_q)
                        + 0.1 * criterion(outputs[8], support_mask))
            else:
                loss = (criterion(outputs[0], mask_q) + criterion(outputs[1], mask_q)
                        + 0.4 * criterion(outputs[2], support_mask))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iterations += 1
            if iterations in lr_decay_iters:
                optimizer.param_groups[0]["lr"] /= 10
            progress.set_description(f"Epoch {epoch + 1}/{epochs}, loss: {total_loss / (index + 1):.3f}")

        set_seed(args.seed)
        score = evaluate(model, testloader, args, processor, captioner, text_encoder, device)
        if score >= best_score:
            best_score = score
            best_model = deepcopy(model)
            path = output_dir / f"prompt_{args.backbone}_{args.shot}shot_{score:.2f}.pth"
            torch.save(state_dict_for_save(best_model), path)

    scores = []
    for offset in range(5):
        set_seed(args.seed + offset)
        scores.append(evaluate(best_model, testloader, args, processor, captioner, text_encoder, device))
    average = sum(scores) / len(scores)
    final_path = output_dir / f"prompt_{args.backbone}_{args.shot}shot_avg_{average:.2f}.pth"
    torch.save(state_dict_for_save(best_model), final_path)
    print(f"Averaged mIoU over 5 seeds: {average:.2f}")
    print(f"Saved checkpoint: {final_path}")


if __name__ == "__main__":
    main()
