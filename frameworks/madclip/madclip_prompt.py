import re

import torch
import torch.nn as nn

from CLIP.tokenizer import SimpleTokenizer, tokenize


REAL_NAME = {
    "all": "medical image",
    "Brain": "Brain MRI",
    "Liver": "liver CT",
    "Retina_RESC": "retinal OCT",
    "Retina_OCT2017": "retinal OCT",
    "Chest": "Chest X-ray film",
    "Histopathology": "histopathological image",
}


def display_name(obj):
    return REAL_NAME.get(obj, obj.replace("_", " "))


def encode_fixed_madclip_text(model, obj, device):
    prompt_normal = [
        "{}",
        "flawless {}",
        "perfect {}",
        "unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without damage",
    ]
    prompt_abnormal = [
        "damaged {}",
        "broken {}",
        "{} with flaw",
        "{} with defect",
        "{} with damage",
        "disease {}",
        "abnormal {}",
    ]
    prompt_templates = [
        "a bad photo of a {}.",
        "a low resolution photo of the {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a bright photo of a {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a photo of one {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "a low resolution photo of a {}.",
        "a photo of a large {}.",
        "a blurry photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "a dark photo of a {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "there is a {} in the scene.",
        "there is the {} in the scene.",
        "this is a {} in the scene.",
        "this is the {} in the scene.",
        "this is one {} in the scene.",
    ]

    text_features = []
    for prompt_state in (prompt_normal, prompt_abnormal):
        prompted_state = [state.format(obj) for state in prompt_state]
        prompted_sentences = [
            template.format(state) for state in prompted_state for template in prompt_templates
        ]
        prompted_sentences = tokenize(prompted_sentences).to(device)
        class_embeddings = model.encode_text(prompted_sentences)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm().clamp_min(1e-6)
        text_features.append(class_embedding)
    return torch.stack(text_features, dim=1).to(device)


class DummyOptimizer:
    def step(self):
        pass

    def zero_grad(self, *args, **kwargs):
        pass


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding[: prompts.shape[1]].to(prompts.dtype)
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        x = out[0] if isinstance(out, tuple) else out
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]
        return x @ self.text_projection


class PromptLearner(nn.Module):
    def __init__(self, prompts, n_ctx, csc, class_token_position, clip_model, device):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.ln_final.weight.dtype
        self.ctx = nn.ParameterDict()
        self.class_token_position = class_token_position
        self.n_ctx = n_ctx

        prompt_prefix = " ".join(["X"] * n_ctx)
        tokenizer = SimpleTokenizer()

        self.tokenized_prompts = {}
        self.prompts_lens = {}
        self.register_embeddings = {}

        for cls, prompt_list in prompts.items():
            clean_prompts = [prompt.replace("_", " ") for prompt in prompt_list]
            self.prompts_lens[cls] = [len(tokenizer.encode(prompt)) for prompt in clean_prompts]
            learnable_prompts = [prompt_prefix + " " + prompt + "." for prompt in clean_prompts]
            tokenized = torch.cat([tokenize(prompt) for prompt in learnable_prompts]).to(device)
            self.tokenized_prompts[cls] = tokenized

            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized).to(dtype)
            self.register_embeddings[f"{cls}_token_prefix"] = embedding[:, :1, :]
            self.register_embeddings[f"{cls}_token_suffix"] = embedding[:, 1 + n_ctx :, :]

            for position in class_token_position:
                if csc:
                    ctx_vectors = torch.empty(len(prompt_list), n_ctx, ctx_dim, dtype=dtype, device=device)
                else:
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx[f"{cls}_{position}"] = nn.Parameter(ctx_vectors)

    def forward(self):
        cls_prompts = {}
        for cls in self.tokenized_prompts:
            prefix = self.register_embeddings[f"{cls}_token_prefix"]
            suffix = self.register_embeddings[f"{cls}_token_suffix"]
            prompts_for_cls = []
            for position in self.class_token_position:
                ctx = self.ctx[f"{cls}_{position}"]
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(len(self.prompts_lens[cls]), -1, -1)

                if position == "end":
                    prompts = torch.cat([prefix, ctx, suffix], dim=1)
                elif position == "middle":
                    half = self.n_ctx // 2
                    prompt_parts = []
                    for i, prompt_len in enumerate(self.prompts_lens[cls]):
                        prefix_i = prefix[i : i + 1]
                        class_i = suffix[i : i + 1, :prompt_len]
                        suffix_i = suffix[i : i + 1, prompt_len:]
                        prompt_parts.append(
                            torch.cat(
                                [prefix_i, ctx[i : i + 1, :half], class_i, ctx[i : i + 1, half:], suffix_i],
                                dim=1,
                            )
                        )
                    prompts = torch.cat(prompt_parts, dim=0)
                else:
                    if position != "front":
                        raise ValueError(f"Unknown class_token_position: {position}")
                    prompt_parts = []
                    for i, prompt_len in enumerate(self.prompts_lens[cls]):
                        prefix_i = prefix[i : i + 1]
                        class_i = suffix[i : i + 1, :prompt_len]
                        suffix_i = suffix[i : i + 1, prompt_len:]
                        prompt_parts.append(
                            torch.cat([prefix_i, class_i, ctx[i : i + 1], suffix_i], dim=1)
                        )
                    prompts = torch.cat(prompt_parts, dim=0)
                prompts_for_cls.append(prompts)
            cls_prompts[cls] = torch.cat(prompts_for_cls, dim=0)
        return cls_prompts


class PromptMaker(nn.Module):
    def __init__(self, prompts, clip_model, device, n_ctx=8, csc=True, class_token_position=None):
        super().__init__()
        if class_token_position is None:
            class_token_position = ["end", "front", "middle"]
        self.prompt_learner = PromptLearner(
            prompts=prompts,
            n_ctx=n_ctx,
            csc=csc,
            class_token_position=class_token_position,
            clip_model=clip_model,
            device=device,
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.class_token_position = class_token_position
        self.text_encoder = TextEncoder(clip_model)

    def forward(self):
        prompts = self.prompt_learner()
        text_features = []
        for cls, prompt_tensor in prompts.items():
            tokenized = self.tokenized_prompts[cls].repeat(len(self.class_token_position), 1)
            class_embedding = self.text_encoder(prompt_tensor, tokenized)
            class_embedding = class_embedding.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm().clamp_min(1e-6)
            text_features.append(class_embedding)
        return torch.stack(text_features, dim=1)


class PromptChooser(nn.Module):
    def __init__(self, clip_model, args, device, obj_name=None):
        super().__init__()
        self.text_mood = args.text_mood
        self.lr = args.learning_rate
        self.obj_name = obj_name if obj_name is not None else args.obj
        real_name = display_name(self.obj_name)

        if self.text_mood == "fix":
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()), torch.no_grad():
                encoded = encode_fixed_madclip_text(clip_model, real_name, device)
            self.register_buffer("text_features_fix", encoded)
            self.text_optimizer = DummyOptimizer()
            return

        prompt_abnormal = [
            "damaged {}",
            "broken {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
            "disease {}",
            "abnormal {}",
        ]
        abnormal_prompts = [state.format(real_name) for state in prompt_abnormal]
        self.prompt_maker_abnormal = PromptMaker(
            prompts={"abnormal": abnormal_prompts},
            clip_model=clip_model,
            device=device,
            n_ctx=args.n_ctx,
            class_token_position=args.class_token_position,
        ).to(device)
        self.prompt_maker_abnormal.train()

        if self.text_mood == "learnable_all":
            prompt_normal = [
                "{}",
                "flawless {}",
                "perfect {}",
                "unblemished {}",
                "{} without flaw",
                "{} without defect",
                "{} without damage",
            ]
            normal_prompts = [state.format(real_name) for state in prompt_normal]
            self.prompt_maker_normal = PromptMaker(
                prompts={"normal": normal_prompts},
                clip_model=clip_model,
                device=device,
                n_ctx=args.n_ctx,
                class_token_position=args.class_token_position,
            ).to(device)
            self.prompt_maker_normal.train()
            self.text_optimizer = torch.optim.Adam(
                [
                    {"params": self.prompt_maker_normal.prompt_learner.parameters(), "lr": self.lr},
                    {"params": self.prompt_maker_abnormal.prompt_learner.parameters(), "lr": self.lr},
                ],
                betas=(0.5, 0.999),
            )
        elif self.text_mood == "learnable_abnormal":
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()), torch.no_grad():
                normal = encode_fixed_madclip_text(clip_model, real_name, device)[:, 0].unsqueeze(1)
            self.register_buffer("text_features_normal", normal)
            self.text_optimizer = torch.optim.Adam(
                [{"params": self.prompt_maker_abnormal.prompt_learner.parameters(), "lr": self.lr}],
                betas=(0.5, 0.999),
            )
        else:
            raise ValueError(f"Unknown text_mood: {self.text_mood}")

    def forward(self):
        if self.text_mood == "fix":
            return self.text_features_fix
        text_features_abnormal = self.prompt_maker_abnormal()
        if self.text_mood == "learnable_all":
            text_features_normal = self.prompt_maker_normal()
            return torch.cat([text_features_normal, text_features_abnormal], dim=1)
        return torch.cat([self.text_features_normal, text_features_abnormal], dim=1)

    def save_prompt(self, save_dict):
        if self.text_mood == "fix":
            save_dict["text_features_fix"] = self.text_features_fix
        elif self.text_mood == "learnable_all":
            save_dict["prompt_maker_normal"] = self.prompt_maker_normal.state_dict()
            save_dict["prompt_maker_abnormal"] = self.prompt_maker_abnormal.state_dict()
        else:
            save_dict["prompt_maker_abnormal"] = self.prompt_maker_abnormal.state_dict()
        return save_dict


def parse_token_positions(value):
    if isinstance(value, list):
        return value
    pieces = [piece.strip() for piece in re.split(r"[, ]+", value) if piece.strip()]
    if not pieces:
        raise ValueError("class_token_position cannot be empty")
    for piece in pieces:
        if piece not in {"end", "front", "middle"}:
            raise ValueError(f"Unknown prompt token position: {piece}")
    return pieces
