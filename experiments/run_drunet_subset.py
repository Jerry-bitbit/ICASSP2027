"""DRUNet inference and image metrics retained from the original experiments."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.paths import ROOT, DATA
from methods.ectv import tv_energy

KAIR_MINIMAL = ROOT / "third_party" / "kair"
DRUNET_WEIGHTS = DATA / "models" / "drunet_color.pth"

def read_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def add_gaussian(clean: np.ndarray, sigma255: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, sigma255 / 255.0, clean.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def gradient_correlation(clean: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    def grad_mag(x: np.ndarray) -> np.ndarray:
        gray = rgb_to_gray(x)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        gy[:-1, :] = gray[1:, :] - gray[:-1, :]
        return np.sqrt(gx * gx + gy * gy)

    gc = grad_mag(clean)
    go = grad_mag(out)
    denom = float(np.linalg.norm(gc.ravel()) * np.linalg.norm(go.ravel()) + 1e-12)
    corr = float(np.dot(gc.ravel(), go.ravel()) / denom)
    mae = float(np.mean(np.abs(gc - go)))
    return corr, mae


def ssim_channel(x: np.ndarray, y: np.ndarray, win: int = 7) -> float:
    from scipy.ndimage import uniform_filter

    c1 = 0.01**2
    c2 = 0.03**2
    ux = uniform_filter(x, size=win, mode="reflect")
    uy = uniform_filter(y, size=win, mode="reflect")
    uxx = uniform_filter(x * x, size=win, mode="reflect")
    uyy = uniform_filter(y * y, size=win, mode="reflect")
    uxy = uniform_filter(x * y, size=win, mode="reflect")
    vx = np.maximum(0.0, uxx - ux * ux)
    vy = np.maximum(0.0, uyy - uy * uy)
    cov = uxy - ux * uy
    num = (2.0 * ux * uy + c1) * (2.0 * cov + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def metric_row(clean: np.ndarray, noisy: np.ndarray, out: np.ndarray) -> dict[str, float]:
    try:
        from skimage import color
        from skimage.filters import sobel
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        ch_axis = -1 if clean.ndim == 3 else None
        min_side = min(clean.shape[:2])
        win_size = 7 if min_side >= 7 else max(3, min_side // 2 * 2 - 1)
        psnr = float(peak_signal_noise_ratio(clean, out, data_range=1.0))
        ssim = float(structural_similarity(clean, out, channel_axis=ch_axis, data_range=1.0, win_size=win_size))
        clean_gray = color.rgb2gray(clean) if clean.ndim == 3 else clean
        out_gray = color.rgb2gray(out) if out.ndim == 3 else out
        gc = sobel(clean_gray)
        go = sobel(out_gray)
        denom = float(np.linalg.norm(gc.ravel()) * np.linalg.norm(go.ravel()) + 1e-12)
        grad_corr = float(np.dot(gc.ravel(), go.ravel()) / denom)
        grad_mae = float(np.mean(np.abs(gc - go)))
    except Exception:
        mse = float(np.mean((clean - out) ** 2))
        psnr = 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))
        ssim = float(np.mean([ssim_channel(clean[..., c], out[..., c]) for c in range(3)]))
        grad_corr, grad_mae = gradient_correlation(clean, out)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "grad_corr": grad_corr,
        "grad_mae": grad_mae,
        "tv_input": tv_energy(noisy),
        "tv_output": tv_energy(out),
    }


def load_drunet(torch_deps: str | None = None):
    if str(KAIR_MINIMAL) not in sys.path:
        sys.path.insert(0, str(KAIR_MINIMAL))
    import torch
    import torch.nn.functional as F
    from network_unet import UNetRes

    if not DRUNET_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing DRUNet weights: {DRUNET_WEIGHTS}")

    model = UNetRes(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        bias=False,
    )
    state = torch.load(DRUNET_WEIGHTS, map_location="cpu")
    if isinstance(state, dict):
        for key in ["params_ema", "params", "state_dict", "model_state_dict"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict):
        state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    return model, torch, F


def denoise_drunet(model, torch, F, noisy: np.ndarray, sigma_est: float) -> np.ndarray:
    h, w = noisy.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    sigma_map = np.full((h, w, 1), float(max(sigma_est, 0.0)), dtype=np.float32)
    x = np.concatenate([noisy.astype(np.float32), sigma_map], axis=2)
    tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        out = model(tensor).clamp_(0.0, 1.0)
    out = out[..., :h, :w].squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out.astype(np.float32), 0.0, 1.0)
