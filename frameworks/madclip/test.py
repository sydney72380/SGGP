#!/usr/bin/env python3
"""Inference-only evaluation for the Table 1 MadCLIP checkpoints."""

import argparse
import json
import logging
import os
from pathlib import Path


DATASETS = (
    "Retina_OCT2017",
    "Histopathology",
    "Chest",
    "Brain",
    "Liver",
    "Retina_RESC",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a released MadCLIP Table 1 checkpoint")
    parser.add_argument("--method", choices=("baseline", "sggp"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing <dataset>_AD folders")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", default=None, help="Optional physical GPU id; prefer CUDA_VISIBLE_DEVICES")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Build the model and load the checkpoint only")
    return parser.parse_args()


def make_logger():
    logger = logging.getLogger("madclip_table1_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def run(cli):
    invocation_cwd = Path.cwd()
    checkpoint_path = Path(cli.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = invocation_cwd / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    data_root = Path(cli.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = invocation_cwd / data_root
    data_root = data_root.resolve()
    if cli.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    source_root = Path(__file__).resolve().parent
    os.chdir(source_root)

    import numpy as np
    import torch
    from torch.nn import functional as F
    from tqdm import tqdm

    from CLIP.clip import create_model
    from CLIP.madclip_adapter import MadCLIPInplanted, MadCLIPInplantedSGGP
    from dataset.medical_few import CLASS_INDEX, MedDataset
    from madclip_loss import MadCLIPDetectionLoss
    from madclip_prompt import PromptChooser, parse_token_positions
    from madclip_train_utils import finite_or_zero, normalize_np, safe_roc_auc, setup_seed

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_args = dict(checkpoint.get("args", {}))
    saved_method = "sggp" if bool(saved_args.get("use_sggp", False)) else "baseline"
    if saved_method != cli.method:
        raise ValueError(
            f"Checkpoint method mismatch: requested {cli.method}, checkpoint is {saved_method}"
        )

    saved_args.update(
        {
            "obj": cli.dataset,
            "data_path": str(data_root),
            "eval_batch_size": cli.eval_batch_size,
            "eval_num_workers": cli.num_workers,
            "pin_memory": True,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "amp": not cli.no_amp,
            "deterministic": True,
            "use_sggp": cli.method == "sggp",
            "method": cli.method,
            "split_json": None,
        }
    )
    defaults = {
        "model_name": "ViT-L-14-336",
        "pretrain": "openai",
        "img_size": 240,
        "features_list": [6, 12, 18, 24],
        "seed": 111,
        "shot": 4,
        "iterate": 0,
        "text_mood": "learnable_all",
        "contrast_mood": "yes",
        "dec_type": "mean",
        "loss_type": "sigmoid",
        "visionA": "MFCFC",
        "n_ctx": 8,
        "class_token_position": ["end", "front", "middle"],
        "nGroups": 11,
        "nMaxN": 5,
        "bBySum": 1,
        "prob": 0.25,
        "sggp_salience_mode": "global_energy",
        "sggp_anomaly_eps": 1e-6,
        "sggp_topk_ratio": 0.15,
        "pure_local_topk_ratio": 0.03,
        "sggp_fg_ratio": 0.8,
        "sggp_memory_max_patches": 4096,
        "sggp_local_mask_source": "gt_or_topk",
        "sggp_local_residual_alpha": 0.7,
        "sggp_cc_topk": 0,
    }
    for key, value in defaults.items():
        saved_args.setdefault(key, value)
    if isinstance(saved_args["class_token_position"], str):
        saved_args["class_token_position"] = parse_token_positions(
            saved_args["class_token_position"]
        )
    args = argparse.Namespace(**saved_args)

    setup_seed(args.seed, deterministic=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = make_logger()

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True,
    ).to(device)
    clip_model.eval()

    model_cls = MadCLIPInplantedSGGP if cli.method == "sggp" else MadCLIPInplanted
    model = model_cls(args, clip_model).to(device)
    text_chooser = PromptChooser(clip_model, args, device).to(device)
    loss_det = MadCLIPDetectionLoss(args.img_size, device, args.loss_type, args.dec_type).to(device)

    model.normal_det_adapters.load_state_dict(checkpoint["normal_det_adapters"])
    model.abnormal_det_adapters.load_state_dict(checkpoint["abnormal_det_adapters"])
    loss_det.load_state_dict(checkpoint["loss_det"])
    if args.text_mood == "fix":
        text_chooser.text_features_fix.copy_(checkpoint["text_features_fix"].to(device))
    elif args.text_mood == "learnable_all":
        text_chooser.prompt_maker_normal.load_state_dict(checkpoint["prompt_maker_normal"])
        text_chooser.prompt_maker_abnormal.load_state_dict(checkpoint["prompt_maker_abnormal"])
    else:
        text_chooser.prompt_maker_abnormal.load_state_dict(checkpoint["prompt_maker_abnormal"])
    model.eval()
    text_chooser.eval()
    loss_det.eval()

    metadata = {
        "backbone": "madclip",
        "method": cli.method,
        "dataset": cli.dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_auc": float(checkpoint["auc"]),
        "checkpoint_pauc": float(checkpoint["pauc"]),
    }
    if cli.check_only:
        metadata["status"] = "checkpoint_load_ok"
        return metadata

    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cli.eval_batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    gt_list = []
    gt_mask_list = []
    det_final = []
    seg_final = []
    with torch.no_grad():
        text_features = text_chooser()

    for image, label, mask in tqdm(test_loader, desc=f"madclip-{cli.method}-{cli.dataset}"):
        image = image.to(device, non_blocking=device.type == "cuda")
        mask = (mask > 0.5).float()
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=(not cli.no_amp) and device.type == "cuda"
        ):
            _, det_model, seg_model = model(image, text_features, use_sggp=False)
            if CLASS_INDEX[args.obj] > 0:
                anomaly_maps = []
                for seg_scores in seg_model:
                    seg_probs = loss_det.sync_as(seg_scores)
                    anomaly_map = 0.5 * (1 - seg_probs[:, 0]) + 0.5 * seg_probs[:, 1]
                    anomaly_maps.append(anomaly_map.detach().cpu().numpy())
                seg_final.extend(np.sum(np.stack(anomaly_maps), axis=0))

            anomaly_scores = 0
            for det_scores in det_model:
                anomaly_scores = anomaly_scores + loss_det.validation(det_scores)
            det_final.extend(anomaly_scores.detach().cpu().numpy())

        gt_mask_list.extend(mask.squeeze(1).cpu().numpy())
        gt_list.extend(label.cpu().numpy())

    gt_array = np.asarray(gt_list)
    gt_mask_array = (np.asarray(gt_mask_list) > 0).astype(np.int_)
    if CLASS_INDEX[args.obj] > 0:
        segment_scores = normalize_np(np.asarray(seg_final))
        pauc = finite_or_zero(
            safe_roc_auc(
                gt_mask_array.flatten(), segment_scores.flatten(), logger, f"{args.obj} pAUC"
            )
        )
        image_scores = segment_scores.reshape(segment_scores.shape[0], -1).max(axis=1)
        auc = finite_or_zero(safe_roc_auc(gt_array, image_scores, logger, f"{args.obj} AUC"))
    else:
        det_scores = normalize_np(np.asarray(det_final))
        auc = finite_or_zero(safe_roc_auc(gt_array, det_scores, logger, f"{args.obj} AUC"))
        pauc = 0.0

    metadata.update(
        {
            "auc": float(auc),
            "pauc": float(pauc) if CLASS_INDEX[args.obj] > 0 else None,
            "checkpoint_reload_match": (
                abs(float(checkpoint["auc"]) - float(auc)) <= 1e-10
                and abs(float(checkpoint["pauc"]) - float(pauc)) <= 1e-10
            ),
            "status": "ok",
        }
    )
    return metadata


def main():
    cli = parse_args()
    result_output = Path(cli.result_json).expanduser().resolve() if cli.result_json else None
    result = run(cli)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if result_output:
        result_output.parent.mkdir(parents=True, exist_ok=True)
        result_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
