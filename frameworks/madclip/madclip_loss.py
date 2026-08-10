import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossSigmoid(nn.Module):
    def __init__(self, decision):
        super().__init__()
        self.log_sigmoid = nn.LogSigmoid()
        self.decision = decision

    def forward(self, logits, labels):
        labels = (2.0 * labels.float() - 1.0).unsqueeze(-1)
        label_all = torch.cat([-labels, labels], dim=1)
        loss_each = self.log_sigmoid(label_all.unsqueeze(1) * logits)
        loss_each = self.decision(loss_each)
        return -torch.mean(torch.sum(loss_each, dim=-1))

    def validation(self, logits):
        anomaly_score = F.softmax(logits, dim=-1)
        return self.decision(anomaly_score[:, :, 1])


class LossSoftmaxBased(nn.Module):
    def __init__(self, decision):
        super().__init__()
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.decision = decision

    def forward(self, logits, labels):
        logits = F.softmax(logits, dim=-1)
        normality_score = self.decision(logits[:, :, 0])
        anomaly_score = self.decision(logits[:, :, 1])
        return self.loss_bce(1 - normality_score, labels.float()) + self.loss_bce(
            anomaly_score, labels.float()
        )

    def validation(self, logits):
        logits = F.softmax(logits, dim=-1)
        return self.decision(logits[:, :, 1])


class MadCLIPDetectionLoss(nn.Module):
    """Detection and segmentation score utilities used by MadCLIP."""

    def __init__(self, img_size, device, loss_type="sigmoid", dec_type="mean"):
        super().__init__()
        self.img_size = img_size
        self.loss_type = loss_type
        self.dec_type = dec_type

        if dec_type == "mean":
            self.decision = lambda x: torch.mean(x, dim=1)
        elif dec_type == "max":
            self.decision = lambda x: torch.max(x, dim=1)[0]
        elif dec_type == "both":
            self.alphadec = nn.Parameter(torch.zeros(1, device=device))
            self.decision = lambda x: torch.sigmoid(self.alphadec) * torch.mean(x, dim=1) + (
                1 - torch.sigmoid(self.alphadec)
            ) * torch.max(x, dim=1)[0]
        else:
            raise ValueError(f"Unknown dec_type: {dec_type}")

        if loss_type == "sigmoid":
            self.loss_impl = LossSigmoid(self.decision)
        elif loss_type == "softmax":
            self.loss_impl = LossSoftmaxBased(self.decision)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.to(device)

    def forward(self, logits, labels):
        return self.loss_impl(logits, labels)

    def validation(self, logits):
        return self.loss_impl.validation(logits)

    def sync_as(self, logits):
        batch_size, num_tokens, channels = logits.shape
        height = int(np.sqrt(num_tokens))
        if height * height != num_tokens:
            raise ValueError(f"Cannot reshape {num_tokens} tokens into a square feature map")
        logits = logits.permute(0, 2, 1).view(batch_size, channels, height, height)
        if torch.is_grad_enabled() and logits.requires_grad:
            logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode="nearest")
        else:
            logits = F.interpolate(
                logits,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=True,
            )
        return torch.softmax(logits, dim=1)
