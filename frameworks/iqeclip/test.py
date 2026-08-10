#!/usr/bin/env python3
"""Inference-only evaluation for the Table 1 IQE-CLIP checkpoints."""

import argparse
import json
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
CLASS_INDEX = {
    "Brain": 3,
    "Liver": 2,
    "Retina_RESC": 1,
    "Retina_OCT2017": -1,
    "Chest": -2,
    "Histopathology": -3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a released IQE-CLIP Table 1 checkpoint")
    parser.add_argument("--method", choices=("baseline", "sggp"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing <dataset>_AD folders")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", default=None, help="Optional physical GPU id; prefer CUDA_VISIBLE_DEVICES")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Build the model and load the checkpoint only")
    return parser.parse_args()


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
    import torchmetrics
    from scipy.ndimage import gaussian_filter
    from torch.nn import functional as F
    from tqdm import tqdm

    from CLIP.clip import create_model
    from CLIP.iqeclip import IQE_CLIP
    from dataset.medical_few import MedDataset
    from utils import _load_stages, normalize, setup_seed

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    setup_seed(111)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "ViT-L-14-336"
    image_size = 240
    features_list = [6, 12, 18, 24]
    with (source_root / "CLIP" / "model_configs" / f"{model_name}.json").open(
        "r", encoding="utf-8"
    ) as handle:
        model_configs = json.load(handle)

    model = create_model(
        model_name=model_name,
        img_size=image_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        deep_prompt_len=1,
        total_d_layer_len=11,
    ).to(device)
    model.eval()
    iqe_clip = IQE_CLIP(
        model,
        features_list=features_list,
        model_configs=model_configs,
        prompt_len=2,
        iqm_config=str(source_root / "config" / "config_iqm.json"),
        query_vison=True,
    ).to(device)
    iqe_clip.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    iqe_clip.trainable_layer.load_state_dict(checkpoint["trainable_linearlayer"], strict=False)
    iqe_clip.New_Lan_Embed.load_state_dict(checkpoint["New_Lan_Embed"])
    iqe_clip.iqm.load_state_dict(checkpoint["iqm"], strict=False)
    iqe_clip.query_tokens.data.copy_(checkpoint["query_tokens"])
    iqe_clip.query_linear.load_state_dict(checkpoint["query_linear"])
    _load_stages(model, checkpoint, "prompt")
    iqe_clip.eval()

    result = {
        "backbone": "iqeclip",
        "method": cli.method,
        "dataset": cli.dataset,
        "checkpoint": str(checkpoint_path),
        "evaluation_sggp": False,
    }
    if cli.check_only:
        result["status"] = "checkpoint_load_ok"
        return result

    test_dataset = MedDataset(
        str(data_root), cli.dataset, image_size, shot=4, iterate=0
    )
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
    anomaly_map_raw_list = []
    anomaly_map_new_list = []
    image_score_raw_list = []
    image_score_new_list = []

    for image, label, mask in tqdm(test_loader, desc=f"iqeclip-{cli.method}-{cli.dataset}"):
        image = image.to(device, non_blocking=device.type == "cuda")
        mask = (mask > 0.5).float()
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=(not cli.no_amp) and device.type == "cuda"
        ):
            image_features, patch_tokens, _ = iqe_clip.clip.encode_image(
                image, features_list, return_x=True
            )
            query_tokens = iqe_clip.get_query(image_features, use_global=True)
            class_token = iqe_clip.New_Lan_Embed.before_extract_feat(
                patch_tokens, image_features.clone(), use_global=True
            )
            text_embeddings = iqe_clip.prompt_pre.forward_ensemble(
                iqe_clip.clip, class_token, device
            ).permute(0, 2, 1)

            patch_tokens_linear = iqe_clip.trainable_layer(patch_tokens)
            anomaly_maps_new = []
            anomaly_maps_raw = []
            for dense_feature in patch_tokens_linear:
                query_feature = iqe_clip.iqm(
                    query_embeds=query_tokens,
                    encoder_hidden_states=dense_feature.clone(),
                    text_encoder_hidden_states=text_embeddings.permute(0, 2, 1),
                )[0]
                dense_norm = dense_feature / dense_feature.norm(dim=-1, keepdim=True)
                query_norm = query_feature / query_feature.norm(dim=-1, keepdim=True)
                anomaly_map_new = dense_norm @ query_norm.permute(0, 2, 1)
                batch, length, _ = anomaly_map_new.shape
                height = int(np.sqrt(length))
                anomaly_map_new = F.interpolate(
                    anomaly_map_new.permute(0, 2, 1).view(batch, 2, height, height),
                    size=image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_maps_new.append(torch.softmax(anomaly_map_new, dim=1)[:, 1].cpu().numpy())

                anomaly_map_raw = iqe_clip.New_Lan_Embed.prompt_temp.exp() * dense_norm @ text_embeddings
                anomaly_map_raw = F.interpolate(
                    anomaly_map_raw.permute(0, 2, 1).view(batch, 2, height, height),
                    size=image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_maps_raw.append(torch.softmax(anomaly_map_raw, dim=1)[:, 1].cpu().numpy())

        gt_list.extend(label.cpu().numpy())
        anomaly_map_raw_batch = np.mean(anomaly_maps_raw, axis=0)
        anomaly_map_new_batch = np.mean(anomaly_maps_new, axis=0)
        image_score_raw_list.append(
            np.mean(anomaly_map_raw_batch.reshape(anomaly_map_raw_batch.shape[0], -1), axis=1)
        )
        image_score_new_list.append(
            np.mean(anomaly_map_new_batch.reshape(anomaly_map_new_batch.shape[0], -1), axis=1)
        )
        if CLASS_INDEX[cli.dataset] > 0:
            gt_mask_list.append(mask.squeeze().cpu().numpy())
            anomaly_map_raw_list.append(anomaly_map_raw_batch)
            anomaly_map_new_list.append(anomaly_map_new_batch)

    gt_array = np.asarray(gt_list)
    image_scores = np.concatenate(image_score_raw_list, axis=0)
    image_scores += np.concatenate(image_score_new_list, axis=0)
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    gt_tensor = torch.from_numpy(gt_array).long().to(device)
    image_tensor = torch.from_numpy(image_scores).float().to(device).squeeze()
    auc = torchmetrics.functional.auroc(image_tensor, gt_tensor, task="binary").item()
    if CLASS_INDEX[cli.dataset] > 0:
        gt_mask_array = (np.concatenate(gt_mask_list, axis=0) > 0).astype(np.int_)
        anomaly_map_raw = np.concatenate(anomaly_map_raw_list, axis=0)
        anomaly_map_new = np.concatenate(anomaly_map_new_list, axis=0)
        segment_scores = normalize(
            gaussian_filter(
                0.2 * anomaly_map_raw + 0.8 * anomaly_map_new,
                sigma=8,
                axes=(1, 2),
            )
        )
        mask_tensor = torch.from_numpy(gt_mask_array).long().to(device)
        segment_tensor = torch.from_numpy(segment_scores).float().to(device).squeeze()
        pauc = torchmetrics.functional.auroc(segment_tensor, mask_tensor, task="binary").item()
    else:
        pauc = None

    result.update({"auc": float(auc), "pauc": float(pauc) if pauc is not None else None, "status": "ok"})
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
