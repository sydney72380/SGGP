#!/usr/bin/env python3
"""Inference-only evaluation for the released MVFA Table 1 checkpoints."""

import argparse
import json
import os
import random
import sys
from pathlib import Path


DATASETS = (
    "Retina_OCT2017",
    "Histopathology",
    "Chest",
    "Brain",
    "Liver",
    "Retina_RESC",
)
CLASS_INDEX = {
    "Brain": 3,
    "Liver": 2,
    "Retina_RESC": 1,
    "Retina_OCT2017": -1,
    "Chest": -2,
    "Histopathology": -3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a released MVFA Table 1 checkpoint")
    parser.add_argument("--method", choices=("baseline", "sggp"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing <dataset>_AD folders")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1, choices=(1,))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", default=None, help="Optional physical GPU id")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Build the model and load the checkpoint only")
    return parser.parse_args()


def setup_seed(torch, np, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    repository_root = source_root.parents[1]
    sys.path.insert(0, str(repository_root))
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
    from torch.nn import functional as F
    from tqdm import tqdm

    from CLIP.adapter import CLIP_Inplanted
    from CLIP.clip import create_model
    from dataset.medical_few import MedDataset
    from prompt import REAL_NAME
    from utils import augment, cos_sim, encode_text_with_prompt_ensemble

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    setup_seed(torch, np, 111)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=240,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = CLIP_Inplanted(clip_model=clip_model, features=[6, 12, 18, 24]).to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    result = {
        "backbone": "mvfa",
        "method": cli.method,
        "dataset": cli.dataset,
        "checkpoint": str(checkpoint_path),
    }
    if cli.check_only:
        result["status"] = "checkpoint_load_ok"
        return result

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "num_workers": cli.num_workers,
        "pin_memory": use_cuda,
    }
    test_dataset = MedDataset(str(data_root), cli.dataset, 240, 4, 0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cli.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    augment_normal_img, _ = augment(test_dataset.fewshot_norm_img)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset,
        batch_size=1,
        shuffle=True,
        **loader_kwargs,
    )

    amp_enabled = (not cli.no_amp) and use_cuda
    with torch.cuda.amp.autocast(enabled=amp_enabled), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(
            clip_model, REAL_NAME[cli.dataset], device
        )

    seg_features = []
    det_features = []
    for (image,) in support_loader:
        image = image.to(device)
        with torch.no_grad():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [tokens[0].contiguous() for tokens in seg_patch_tokens]
            det_patch_tokens = [tokens[0].contiguous() for tokens in det_patch_tokens]
            seg_features.append(seg_patch_tokens)
            det_features.append(det_patch_tokens)
    seg_mem_features = [
        torch.cat([features[layer] for features in seg_features], dim=0)
        for layer in range(len(seg_features[0]))
    ]
    det_mem_features = [
        torch.cat([features[layer] for features in det_features], dim=0)
        for layer in range(len(det_features[0]))
    ]

    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []

    for image, label, mask in tqdm(test_loader, desc=f"mvfa-{cli.method}-{cli.dataset}"):
        image = image.to(device)
        mask = (mask > 0.5).float()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [tokens[0, 1:, :] for tokens in seg_patch_tokens]
            det_patch_tokens = [tokens[0, 1:, :] for tokens in det_patch_tokens]

            if CLASS_INDEX[cli.dataset] > 0:
                anomaly_maps_few = []
                for index, tokens in enumerate(seg_patch_tokens):
                    similarity = cos_sim(seg_mem_features[index], tokens)
                    height = int(np.sqrt(similarity.shape[1]))
                    anomaly_map = torch.min(1 - similarity, 0)[0].reshape(
                        1, 1, height, height
                    )
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=240,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_maps_few.append(anomaly_map[0].cpu().numpy())
                seg_score_map_few.append(np.sum(anomaly_maps_few, axis=0))

                anomaly_maps_zero = []
                for tokens in seg_patch_tokens:
                    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * tokens @ text_features).unsqueeze(0)
                    batch, length, _ = anomaly_map.shape
                    height = int(np.sqrt(length))
                    anomaly_map = F.interpolate(
                        anomaly_map.permute(0, 2, 1).view(batch, 2, height, height),
                        size=240,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_maps_zero.append(
                        torch.softmax(anomaly_map, dim=1)[:, 1].cpu().numpy()
                    )
                seg_score_map_zero.append(np.sum(anomaly_maps_zero, axis=0))
            else:
                anomaly_maps_few = []
                for index, tokens in enumerate(det_patch_tokens):
                    similarity = cos_sim(det_mem_features[index], tokens)
                    height = int(np.sqrt(similarity.shape[1]))
                    anomaly_map = torch.min(1 - similarity, 0)[0].reshape(
                        1, 1, height, height
                    )
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=240,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_maps_few.append(anomaly_map[0].cpu().numpy())
                det_image_scores_few.append(np.sum(anomaly_maps_few, axis=0).mean())

                anomaly_score = 0
                for tokens in det_patch_tokens:
                    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * tokens @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

        gt_mask_list.append(mask.squeeze().cpu().numpy())
        gt_list.extend(label.cpu().numpy())

    gt_array = np.asarray(gt_list)
    gt_mask_array = (np.asarray(gt_mask_list) > 0).astype(np.int_)
    if CLASS_INDEX[cli.dataset] > 0:
        try:
            from cuml.metrics import roc_auc_score as segment_roc_auc_score
        except ImportError:
            segment_roc_auc_score = sklearn_roc_auc_score

        seg_zero = np.asarray(seg_score_map_zero)
        seg_few = np.asarray(seg_score_map_few)
        seg_zero = (seg_zero - seg_zero.min()) / (seg_zero.max() - seg_zero.min())
        seg_few = (seg_few - seg_few.min()) / (seg_few.max() - seg_few.min())
        segment_scores = 0.5 * seg_zero + 0.5 * seg_few

        if CLASS_INDEX[cli.dataset] == 3:
            split_point = len(gt_mask_array) // 2
            first = segment_roc_auc_score(
                gt_mask_array[:split_point].flatten(),
                segment_scores[:split_point].flatten(),
            )
            second = segment_roc_auc_score(
                gt_mask_array[split_point:].flatten(),
                segment_scores[split_point:].flatten(),
            )
            pauc = (first + second) / 2
        else:
            pauc = segment_roc_auc_score(
                gt_mask_array.flatten(), segment_scores.flatten()
            )
        image_scores = segment_scores.reshape(segment_scores.shape[0], -1).max(axis=1)
        auc = sklearn_roc_auc_score(gt_array, image_scores)
    else:
        det_zero = np.asarray(det_image_scores_zero)
        det_few = np.asarray(det_image_scores_few)
        det_zero = (det_zero - det_zero.min()) / (det_zero.max() - det_zero.min())
        det_few = (det_few - det_few.min()) / (det_few.max() - det_few.min())
        auc = sklearn_roc_auc_score(gt_array, 0.5 * det_zero + 0.5 * det_few)
        pauc = None

    result.update(
        {
            "auc": float(auc),
            "pauc": float(pauc) if pauc is not None else None,
            "status": "ok",
        }
    )
    return result


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
