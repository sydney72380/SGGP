import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _normalize_tokens(x, eps=1e-6):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _instance_norm_tokens(norm, x):
    # x: [seq_len, batch, channels]. InstanceNorm1d expects [batch, channels, seq_len].
    x = x.permute(1, 2, 0)
    x = norm(x)
    return x.permute(2, 0, 1)


class MCNNFC(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=1, stride=1)
        self.norm1 = nn.InstanceNorm1d(c_in, affine=False)
        self.relu = nn.LeakyReLU(inplace=False)
        self.fc1 = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=False))
        self.fc2 = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=False))

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.cnn(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = x.permute(2, 0, 1)
        return self.fc1(x), self.fc2(x)


class MFCFC(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(c_in, c_in, bias=False), nn.LeakyReLU(inplace=False))
        self.norm1 = nn.InstanceNorm1d(c_in, affine=False)
        self.fc2 = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=False))
        self.fc3 = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=False))

    def forward(self, x):
        x = self.fc1(x)
        x = _instance_norm_tokens(self.norm1, x)
        return self.fc2(x), self.fc3(x)


class MViTFC(nn.Module):
    def __init__(self, c_in, bottleneck=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=c_in,
            nhead=num_heads,
            dim_feedforward=c_in,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.fc = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=True))
        self.fc1 = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return self.fc(x), self.fc1(x)


def _make_adapter(kind, in_channels=1024, bottleneck=768):
    if kind == "MCNNFC":
        return MCNNFC(in_channels, bottleneck=bottleneck)
    if kind == "MFCFC":
        return MFCFC(in_channels, bottleneck=bottleneck)
    if kind == "MViTFC":
        return MViTFC(in_channels, bottleneck=bottleneck)
    raise ValueError(f"Unknown MadCLIP adapter type: {kind}")


class MadCLIPInplanted(nn.Module):
    """MadCLIP dual-branch adapter head on a frozen CLIP visual encoder."""

    def __init__(self, args, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = list(args.features_list)
        self.img_size = args.img_size
        self.contrast_mood = args.contrast_mood

        for param in self.clipmodel.parameters():
            param.requires_grad = False

        adapter_type = args.visionA
        self.normal_det_adapters = nn.ModuleList(
            [_make_adapter(adapter_type, 1024, 768) for _ in self.features]
        )
        self.abnormal_det_adapters = nn.ModuleList(
            [_make_adapter(adapter_type, 1024, 768) for _ in self.features]
        )

        if self.contrast_mood == "no":
            self.contrast = lambda same, opposite: same
        elif self.contrast_mood == "yes":
            self.contrast = lambda same, opposite: same - opposite
        else:
            raise ValueError(f"Unknown contrast_mood: {self.contrast_mood}")

    def adapter_parameters(self):
        return list(self.normal_det_adapters.parameters()) + list(self.abnormal_det_adapters.parameters())

    def _prepare_tokens(self, image, use_sggp=False, sggp_context=None):
        with torch.no_grad():
            x = self.image_encoder.conv1(image)
            rand_index = None
            if use_sggp:
                x, rand_index = self.sggp_mix_conv(x, sggp_context=sggp_context)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            cls_token = self.image_encoder.class_embedding.to(x.dtype)
            cls_token = cls_token + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.image_encoder.positional_embedding.to(x.dtype)
            x = self.image_encoder.patch_dropout(x)
            x = self.image_encoder.ln_pre(x)
            x = x.permute(1, 0, 2)
        return x, rand_index

    def _block_forward_no_grad(self, block, x):
        with torch.no_grad():
            out = block(x, attn_mask=None)
            if isinstance(out, tuple):
                return out[0]
            return out

    def _dual_contrast(self, features, same_text, opposite_text):
        same_view = features @ same_text.unsqueeze(-1)
        cross_view = features @ opposite_text.unsqueeze(-1)
        return self.contrast(same_view, cross_view)

    def _score_layer(self, tokens, layer_index, text_features, return_features=False):
        normal_det_i, normal_seg_i = self.normal_det_adapters[layer_index](tokens)
        abnormal_det_i, abnormal_seg_i = self.abnormal_det_adapters[layer_index](tokens)

        normal_det_i = _normalize_tokens(normal_det_i.permute(1, 0, 2)[:, 1:, :])
        normal_seg_i = _normalize_tokens(normal_seg_i.permute(1, 0, 2)[:, 1:, :])
        abnormal_det_i = _normalize_tokens(abnormal_det_i.permute(1, 0, 2)[:, 1:, :])
        abnormal_seg_i = _normalize_tokens(abnormal_seg_i.permute(1, 0, 2)[:, 1:, :])

        sim_det_normal = self._dual_contrast(normal_det_i, text_features[:, 0], text_features[:, 1])
        sim_det_abnormal = self._dual_contrast(abnormal_det_i, text_features[:, 1], text_features[:, 0])
        det_scores = torch.cat([sim_det_normal, sim_det_abnormal], dim=-1)

        sim_seg_normal = self._dual_contrast(normal_seg_i, text_features[:, 0], text_features[:, 1])
        sim_seg_abnormal = self._dual_contrast(abnormal_seg_i, text_features[:, 1], text_features[:, 0])
        seg_scores = torch.cat([sim_seg_normal, sim_seg_abnormal], dim=-1)
        if return_features:
            return det_scores, seg_scores, normal_det_i, normal_seg_i
        return det_scores, seg_scores

    def forward(self, image, text_features, use_sggp=False, return_features=False, sggp_context=None):
        x, rand_index = self._prepare_tokens(image, use_sggp=use_sggp, sggp_context=sggp_context)
        det_scores = []
        seg_scores = []
        det_features = []
        seg_features = []

        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            x = self._block_forward_no_grad(block, x)
            layer_number = i + 1
            if layer_number in self.features:
                scored = self._score_layer(
                    x,
                    self.features.index(layer_number),
                    text_features,
                    return_features=return_features,
                )
                if return_features:
                    det_i, seg_i, det_feat_i, seg_feat_i = scored
                    det_features.append(det_feat_i)
                    seg_features.append(seg_feat_i)
                else:
                    det_i, seg_i = scored
                det_scores.append(det_i)
                seg_scores.append(seg_i)

        if use_sggp:
            if return_features:
                return None, det_scores, seg_scores, rand_index, det_features, seg_features
            return None, det_scores, seg_scores, rand_index
        if return_features:
            return None, det_scores, seg_scores, det_features, seg_features
        return None, det_scores, seg_scores

    def sggp_mix_conv(self, x, sggp_context=None):
        raise RuntimeError("This MadCLIP model was created without SGGP mixing support.")


class MadCLIPInplantedSGGP(MadCLIPInplanted):
    """MadCLIP plus SGGP group PnMix before CLIP patch tokenization."""

    def __init__(self, args, clip_model):
        super().__init__(args, clip_model)
        self.nGroups = args.nGroups
        self.nMaxN = args.nMaxN
        self.bBySum = args.bBySum
        self.pono_eps = 1e-5
        self.sggp_salience_mode = getattr(args, "sggp_salience_mode", "global_energy")
        self.sggp_anomaly_eps = float(getattr(args, "sggp_anomaly_eps", 1e-6))
        self.sggp_topk_ratio = float(getattr(args, "sggp_topk_ratio", 0.15))
        self.sggp_fg_ratio = float(getattr(args, "sggp_fg_ratio", 0.8))
        self.sggp_local_mask_source = getattr(args, "sggp_local_mask_source", "gt_or_topk")
        self.sggp_local_residual_alpha = float(getattr(args, "sggp_local_residual_alpha", 0.7))
        self.sggp_cc_topk = int(getattr(args, "sggp_cc_topk", 0))
        legacy_local = self.sggp_salience_mode in ("local_sggp", "local_residual")
        self.sggp_score_scope = getattr(args, "sggp_score_scope", None) or (
            "local" if legacy_local else "legacy"
        )
        self.sggp_write_scope = getattr(args, "sggp_write_scope", None) or (
            "local" if legacy_local else "global"
        )
        if self.sggp_score_scope not in ("global", "local", "legacy"):
            raise ValueError(f"Unsupported sggp_score_scope={self.sggp_score_scope}")
        if self.sggp_write_scope not in ("global", "local"):
            raise ValueError(f"Unsupported sggp_write_scope={self.sggp_write_scope}")
        self._normal_memory = None

    def set_normal_memory(self, memory):
        self._normal_memory = memory

    def _pono_group(self, x, num_groups):
        batch_size, channels, height, width = x.shape
        if channels % num_groups != 0:
            raise ValueError(f"channels={channels} is not divisible by num_groups={num_groups}")
        group_channels = channels // num_groups
        grouped = x.view(batch_size, num_groups, group_channels, height, width)
        mean = grouped.mean(dim=2, keepdim=True)
        if num_groups == channels:
            normalized = torch.zeros_like(grouped)
            std = torch.zeros_like(grouped)
        else:
            std = (grouped.var(dim=2, keepdim=True) + self.pono_eps).sqrt()
            normalized = (grouped - mean) / std
        return normalized, mean, std

    @staticmethod
    def _moment_shortcut_group(x, mean, std):
        return (x * std + mean).view(x.size(0), -1, x.size(3), x.size(4))

    def _resize_context_mask(self, mask, x, fallback=None):
        if mask is None:
            return fallback
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=x.device, dtype=x.dtype)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        return mask.clamp(0.0, 1.0)

    def _activation_topk_mask(self, x):
        energy = x.detach().abs().mean(dim=1, keepdim=True)
        return self._topk_mask_from_energy(energy, self.sggp_topk_ratio)

    @staticmethod
    def _topk_mask_from_energy(energy, ratio):
        flat = energy.flatten(1)
        k = max(1, int(round(flat.shape[1] * max(0.0, min(1.0, ratio)))))
        threshold = torch.topk(flat, k=k, dim=1).values[:, -1].view(-1, 1, 1, 1)
        return (energy >= threshold).to(dtype=energy.dtype)

    def _adaptive_topk_mask(self, x):
        energy = x.detach().abs().mean(dim=1, keepdim=True)
        flat = energy.flatten(1)
        prob = flat / flat.sum(dim=1, keepdim=True).clamp_min(self.sggp_anomaly_eps)
        entropy = -(prob * prob.clamp_min(self.sggp_anomaly_eps).log()).sum(dim=1)
        entropy = entropy / np.log(max(2, flat.shape[1]))
        ratio = 0.01 + (0.08 * entropy).clamp(0.0, 1.0)
        masks = []
        for batch_idx in range(x.shape[0]):
            masks.append(self._topk_mask_from_energy(energy[batch_idx:batch_idx + 1], float(ratio[batch_idx])))
        return torch.cat(masks, dim=0).to(dtype=x.dtype)

    @staticmethod
    def _component_filter_2d(mask_2d, top_components):
        mask_np = mask_2d.detach().cpu().numpy().astype(bool)
        height, width = mask_np.shape
        visited = np.zeros_like(mask_np, dtype=bool)
        components = []
        for row in range(height):
            for col in range(width):
                if not mask_np[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                pixels = []
                while stack:
                    cur_r, cur_c = stack.pop()
                    pixels.append((cur_r, cur_c))
                    for nxt_r, nxt_c in (
                        (cur_r - 1, cur_c), (cur_r + 1, cur_c),
                        (cur_r, cur_c - 1), (cur_r, cur_c + 1),
                    ):
                        if (
                            0 <= nxt_r < height and 0 <= nxt_c < width
                            and mask_np[nxt_r, nxt_c] and not visited[nxt_r, nxt_c]
                        ):
                            visited[nxt_r, nxt_c] = True
                            stack.append((nxt_r, nxt_c))
                components.append(pixels)
        components.sort(key=len, reverse=True)
        keep = np.zeros_like(mask_np, dtype=np.float32)
        for pixels in components[:max(1, int(top_components))]:
            for row, col in pixels:
                keep[row, col] = 1.0
        return torch.from_numpy(keep).to(device=mask_2d.device, dtype=mask_2d.dtype)

    def _connected_component_mask(self, mask, top_components):
        if top_components <= 0:
            return mask
        filtered = []
        for batch_idx in range(mask.shape[0]):
            filtered.append(self._component_filter_2d(mask[batch_idx, 0], top_components).view(1, 1, *mask.shape[-2:]))
        return torch.cat(filtered, dim=0)

    def _center_foreground_mask(self, x):
        batch, _, height, width = x.shape
        ratio = max(0.05, min(1.0, self.sggp_fg_ratio))
        fg_h = max(1, int(round(height * ratio)))
        fg_w = max(1, int(round(width * ratio)))
        top = max(0, (height - fg_h) // 2)
        left = max(0, (width - fg_w) // 2)
        mask = torch.zeros(batch, 1, height, width, device=x.device, dtype=x.dtype)
        mask[:, :, top:top + fg_h, left:left + fg_w] = 1.0
        return mask

    def _score_mix(self, mix_y, context=None):
        context = context or {}
        mode = self.sggp_salience_mode
        if mode == "anomaly_aware":
            anomaly_mask = self._resize_context_mask(
                context.get("anomaly_mask"), mix_y, fallback=self._activation_topk_mask(mix_y)
            )
            fg_mask = self._resize_context_mask(context.get("foreground_mask"), mix_y)
            if fg_mask is not None:
                anomaly_mask = anomaly_mask * fg_mask
                normal_mask = (1.0 - anomaly_mask) * fg_mask
            else:
                normal_mask = 1.0 - anomaly_mask
            energy = mix_y.abs().mean(dim=1, keepdim=True)
            anomaly_energy = (energy * anomaly_mask).sum()
            normal_energy = (energy * normal_mask).sum()
            return anomaly_energy / normal_energy.clamp_min(self.sggp_anomaly_eps)

        if mode == "residual_memory":
            return self._score_mix_residual(mix_y)

        if mode == "local_sggp":
            return self._score_mix_local(mix_y, context)

        if mode == "foreground_energy":
            fg_mask = self._resize_context_mask(
                context.get("foreground_mask"), mix_y, fallback=self._center_foreground_mask(mix_y)
            )
            energy = mix_y.abs().mean(dim=1, keepdim=True)
            return (energy * fg_mask).sum()

        return self._score_mix_global(mix_y)

    def _local_mask(self, x, context=None):
        context = context or {}
        source = self.sggp_local_mask_source
        if source == "gt":
            mask = self._resize_context_mask(context.get("anomaly_mask"), x, fallback=None)
            if mask is not None:
                return mask
            return self._activation_topk_mask(x)
        if source == "gt_or_topk":
            mask = self._resize_context_mask(context.get("anomaly_mask"), x, fallback=None)
            if mask is not None:
                return mask
            return self._activation_topk_mask(x)
        if source == "adaptive_topk":
            mask = self._adaptive_topk_mask(x)
        elif source in ("topk", "cc_topk"):
            mask = self._activation_topk_mask(x)
        else:
            raise ValueError(f"Unsupported sggp_local_mask_source={source}")
        if source == "cc_topk":
            mask = self._connected_component_mask(mask, self.sggp_cc_topk)
        return mask

    def _score_mix_local(self, mix_y, context=None):
        local_mask = self._local_mask(mix_y, context)
        energy = mix_y.abs().mean(dim=1, keepdim=True)
        return (energy * local_mask).sum()

    def _score_mix_residual(self, mix_y):
        if self._normal_memory is None or self._normal_memory.numel() == 0:
            return self._score_mix_global(mix_y)
        query = mix_y.permute(0, 2, 3, 1).reshape(-1, mix_y.shape[1]).float()
        memory = self._normal_memory.to(query.device).float()
        distances = torch.cdist(query, memory, p=2).min(dim=1).values
        k = max(1, int(round(distances.numel() * max(0.0, min(1.0, self.sggp_topk_ratio)))))
        return torch.topk(distances, k=k).values.mean()

    @staticmethod
    def _normalize_score_stack(scores):
        stack = torch.stack(scores)
        score_min = stack.min()
        score_max = stack.max()
        return (stack - score_min) / (score_max - score_min).clamp_min(1e-6)

    def _score_mix_global(self, mix_y):
        if self.bBySum == 1:
            score = torch.sum(mix_y)
        elif self.bBySum == 0:
            score = torch.var(mix_y)
        elif self.bBySum == 2:
            score = torch.max(mix_y)
        elif self.bBySum == 3:
            score = torch.norm(mix_y, p=2, dim=(1, 2, 3)).sum()
        else:
            raise ValueError(f"Unsupported bBySum={self.bBySum}")
        return score

    def _local_mix_output(self, original_x, mixed_x, context):
        if self.sggp_write_scope != "local":
            return mixed_x
        anomaly_mask = self._local_mask(original_x, context)
        fg_mask = self._resize_context_mask((context or {}).get("foreground_mask"), original_x)
        if fg_mask is not None:
            anomaly_mask = anomaly_mask * fg_mask
        return anomaly_mask * mixed_x + (1.0 - anomaly_mask) * original_x

    def sggp_mix_conv(self, x, sggp_context=None):
        batch_size = x.size(0)
        rand_index = torch.randperm(batch_size, device=x.device)
        lam = 0.5
        context = dict(sggp_context or {})
        if "anomaly_mask" in context and context["anomaly_mask"] is not None:
            mask = context["anomaly_mask"]
            context["anomaly_mask"] = torch.maximum(mask, mask[rand_index])

        pure_local = (
            self.sggp_score_scope == "local"
            and self.sggp_salience_mode == "local_residual"
            and self.sggp_local_residual_alpha >= 1.0
        )
        mix_list = []
        score_list = []
        max_group_power = min(self.nGroups, int(np.log2(x.size(1))) + 1)
        for i in range(max_group_power):
            num_groups = 2 ** i
            x_input, mean_input, std_input = self._pono_group(x, num_groups)
            x2_input, mean2_input, std2_input = self._pono_group(x[rand_index, :], num_groups)
            x1 = self._moment_shortcut_group(x_input, mean2_input, std2_input)
            x2 = self._moment_shortcut_group(x2_input, mean_input, std_input)
            mix_y = lam * x1 + (1.0 - lam) * x2
            mix_list.append(mix_y)
            if self.sggp_score_scope == "local":
                local_score = self._score_mix_local(mix_y, context=context)
                if pure_local or self.sggp_salience_mode != "local_residual":
                    score_list.append(local_score)
                else:
                    score_list.append((local_score, self._score_mix_residual(mix_y)))
            elif self.sggp_score_scope == "global":
                score_list.append(self._score_mix_global(mix_y))
            else:
                score_list.append(self._score_mix(mix_y, context=context))

        top_k = min(self.nMaxN, len(mix_list))
        if self.sggp_score_scope == "local":
            if pure_local:
                final_scores = self._normalize_score_stack(score_list)
            elif self.sggp_salience_mode != "local_residual":
                final_scores = torch.stack(score_list)
            else:
                local_scores = self._normalize_score_stack([score[0] for score in score_list])
                residual_scores = self._normalize_score_stack([score[1] for score in score_list])
                alpha = max(0.0, min(1.0, self.sggp_local_residual_alpha))
                final_scores = alpha * local_scores + (1.0 - alpha) * residual_scores
        else:
            final_scores = torch.stack(score_list)
        indices = torch.topk(final_scores, k=top_k).indices.detach().cpu().tolist()
        selected = [mix_list[idx] for idx in indices]
        mixed = torch.stack(selected, dim=0).mean(dim=0)
        mixed = self._local_mix_output(x, mixed, context)
        return mixed, rand_index
