import logging
import os
import random
import subprocess

import numpy as np
import torch


def setup_seed(seed, deterministic=True):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dataloader_kwargs(batch_size, shuffle, num_workers=0, pin_memory=False,
                      persistent_workers=False, prefetch_factor=2, seed=None):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator

        def seed_worker(worker_id):
            worker_seed = int(seed) + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed % (2 ** 32))
            torch.manual_seed(worker_seed)

        kwargs["worker_init_fn"] = seed_worker
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = max(1, prefetch_factor)
    return kwargs


def make_logger(save_path, name="madclip"):
    ensure_dir(save_path)
    log_path = os.path.join(save_path, "log.txt")
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def log_environment(logger, args):
    logger.info("========== Environment Info ==========")
    for arg in vars(args):
        logger.info(f"args.{arg}: {getattr(args, arg)}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.STDOUT,
        ).decode().strip()
        logger.info(f"git commit: {commit}")
    except Exception:
        logger.info("git commit: unavailable")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"cuda device count: {torch.cuda.device_count()}")
        logger.info(f"current cuda device: {torch.cuda.current_device()}")
        logger.info(f"current cuda name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.info("CUDA unavailable, using CPU")
    logger.info("======================================")


def normalize_np(scores):
    scores = np.asarray(scores)
    score_min = scores.min()
    score_max = scores.max()
    denom = score_max - score_min
    if denom == 0:
        return np.zeros_like(scores)
    return (scores - score_min) / denom


def safe_roc_auc(labels, scores, logger, name):
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(labels, scores))
    except ValueError as exc:
        logger.warning(f"{name} roc_auc_score failed: {exc}")
        return float("nan")


def finite_or_zero(value):
    return value if np.isfinite(value) else 0.0


def autocast_enabled(args, device):
    return bool(getattr(args, "amp", 1)) and device.type == "cuda"


def zero_optimizers(*optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def step_optimizers(*optimizers):
    for optimizer in optimizers:
        optimizer.step()
