"""
inference_rsna.py —— MedTurbo RSNA 肺炎推理脚本

输出结构：
  output_dir/
    composited_fakes/
    raw_fakes/
    aligned_masks/
    sampled_masks/
    grids/
    manifest.csv
"""

import os
import sys
import csv
import glob
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

from config_rsna import (
    DATA_ROOT, OUTPUT_DIR,
    SD_TURBO_PATH,
    NORMAL_DIR_NAME, LUNG_MASK_DIR_NAME,
    VALID_EXTENSIONS,
    IMG_SIZE, PROMPT, INFER_TIMESTEP, DEVICE,
    LORA_RANK_UNET, LORA_RANK_VAE,
    TRAIN_MASK_DILATE_KERNEL, TRAIN_MASK_BLUR_KERNEL,
    USE_MASK_COMPOSITING,
)
from losses_rsna import make_composite_mask
from med_turbo_rsna import MedTurboRSNAModel, load_lora_checkpoint


# ════════════════════════════════════════════════════════════
# Pillow 兼容性
# ════════════════════════════════════════════════════════════
try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST

VALID_EXT = set(VALID_EXTENSIONS)


# ════════════════════════════════════════════════════════════
# Transform
# ════════════════════════════════════════════════════════════
def _img_transform(size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _mask_transform(size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


IMG_TF = _img_transform()
MASK_TF = _mask_transform()


# ════════════════════════════════════════════════════════════
# Mask
# ════════════════════════════════════════════════════════════
def make_infer_mask(
    mask: torch.Tensor,
    lung_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    m = mask.float().clamp(0.0, 1.0)
    if lung_mask is not None:
        m = m * lung_mask.float().clamp(0.0, 1.0)
    return make_composite_mask(m, TRAIN_MASK_DILATE_KERNEL, TRAIN_MASK_BLUR_KERNEL)


def make_infer_composite_mask(
    m_core: torch.Tensor,
    lung_t: torch.Tensor | None,
    dilate_kernel: int,
    blur_kernel: int,
) -> torch.Tensor:
    m = m_core.float().clamp(0.0, 1.0)

    if lung_t is not None:
        m = m * lung_t.float().clamp(0.0, 1.0)

    m = (m > 0.05).float()

    if m.sum() == 0:
        return m

    dk = max(1, dilate_kernel)
    bk = max(1, blur_kernel)

    if dk % 2 == 0:
        dk += 1
    if bk % 2 == 0:
        bk += 1

    m_comp = make_composite_mask(m, dk, bk)
    return m_comp.clamp(0.0, 1.0)


# ════════════════════════════════════════════════════════════
# Lung-aware mask alignment
# ════════════════════════════════════════════════════════════
def _fg_bbox(binary: np.ndarray, thr: int = 127):
    ys, xs = np.where(binary > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _connected_components_topk(binary: np.ndarray, k: int = 2):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num <= 1:
        bbox = _fg_bbox(binary)
        return [(binary, bbox)]

    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num)]
    areas.sort(reverse=True)

    result = []
    for _, lbl in areas[:k]:
        comp = ((labels == lbl) * 255).astype(np.uint8)
        result.append((comp, _fg_bbox(comp)))

    return result


def align_mask_to_lung(
    sampled_mask: np.ndarray,
    lung_mask: np.ndarray,
    output_size: int = 512,
    scale_range: tuple = (0.3, 0.6),
    rng: random.Random = None,
) -> np.ndarray:
    _rng = rng or random

    sm = cv2.resize(sampled_mask, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    lm = cv2.resize(lung_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    try:
        binary_lung = (lm > 127).astype(np.uint8)
        components = _connected_components_topk(binary_lung, k=2)

        if not components:
            raise ValueError("没有找到 lung component")

        comp_mask, bbox = _rng.choice(components)

        if bbox is None:
            raise ValueError("bbox 为 None")

        cbx1, cby1, cbx2, cby2 = bbox
        cb_w = max(cbx2 - cbx1, 1)
        cb_h = max(cby2 - cby1, 1)

        sm_bbox = _fg_bbox(sm)
        if sm_bbox is None:
            raise ValueError("sampled mask 全零")

        sx1, sy1, sx2, sy2 = sm_bbox
        sm_fg = sm[sy1:sy2 + 1, sx1:sx2 + 1]

        scale = _rng.uniform(*scale_range)
        tgt_w = max(int(cb_w * scale), 8)
        tgt_h = max(int(cb_h * scale), 8)

        sm_r = cv2.resize(sm_fg, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

        max_dx = max(cb_w - tgt_w, 0)
        max_dy = max(cb_h - tgt_h, 0)

        dx = _rng.randint(0, max_dx) if max_dx > 0 else 0
        dy = _rng.randint(0, max_dy) if max_dy > 0 else 0

        canvas = np.zeros((output_size, output_size), dtype=np.float32)

        px, py = cbx1 + dx, cby1 + dy
        ex, ey = min(px + tgt_w, output_size), min(py + tgt_h, output_size)
        sw, sh = ex - px, ey - py

        if sw > 0 and sh > 0:
            canvas[py:ey, px:ex] = sm_r[:sh, :sw].astype(np.float32) / 255.0

        comp_f = (comp_mask > 0).astype(np.float32)
        aligned = canvas * comp_f
        aligned = cv2.GaussianBlur(aligned, (31, 31), 0)

        amax = aligned.max()
        if amax > 1e-5:
            aligned = aligned / amax

        return aligned.astype(np.float32)

    except Exception as e:
        warnings.warn(f"[align_mask_to_lung] 对齐失败 ({e})，使用 fallback")

        sm_f = sm.astype(np.float32) / 255.0
        lm_f = (lm > 127).astype(np.float32)

        fb = sm_f * lm_f
        fb = cv2.GaussianBlur(fb, (31, 31), 0)

        fbmax = fb.max()
        if fbmax > 1e-5:
            fb = fb / fbmax

        return fb.astype(np.float32)


# ════════════════════════════════════════════════════════════
# Save QC grid
# ════════════════════════════════════════════════════════════
def save_qc_7(
    normal_bgr: np.ndarray,
    lung_bgr: np.ndarray,
    sampled_bgr: np.ndarray,
    aligned_f: np.ndarray,
    raw_bgr: np.ndarray,
    comp_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    save_path: str,
    panel_size: int = 384,
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    labels = [
        "Input_Normal",
        "LungMask",
        "SampledMask",
        "AlignedMask",
        "RawFake",
        "CompositeFake",
        "DiffMap",
    ]

    arrs = [
        normal_bgr,
        lung_bgr,
        sampled_bgr,
        cv2.cvtColor((aligned_f * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        raw_bgr,
        comp_bgr,
        diff_bgr,
    ]

    panels = []

    for arr, lbl in zip(arrs, labels):
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        arr = cv2.resize(arr.astype(np.uint8), (panel_size, panel_size))
        cv2.putText(arr, lbl, (5, 22), font, 0.6, (0, 255, 0), 1)
        panels.append(arr)

    cv2.imwrite(save_path, cv2.hconcat(panels))


# ════════════════════════════════════════════════════════════
# Tensor to BGR
# ════════════════════════════════════════════════════════════
def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = ((t[0].float() + 1) / 2).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ════════════════════════════════════════════════════════════
# Checkpoint
# ════════════════════════════════════════════════════════════
def find_checkpoint(output_dir: str, ckpt_path: str | None) -> str | None:
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path

    candidates = sorted(glob.glob(os.path.join(output_dir, "ckpt_ep*.pth")))
    if candidates:
        return candidates[-1]

    final = os.path.join(output_dir, "ckpt_final.pth")
    return final if os.path.exists(final) else None


# ════════════════════════════════════════════════════════════
# Main inference
# ════════════════════════════════════════════════════════════
def run_inference(args):
    ckpt = find_checkpoint(OUTPUT_DIR, args.ckpt_path)

    if ckpt is None:
        print("❌ 找不到 checkpoint，请先训练或指定 --ckpt_path")
        sys.exit(1)

    print(f"使用 checkpoint: {ckpt}")

    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)

    res_ep = out_root

    dir_comp = os.path.join(res_ep, "composited_fakes")
    dir_raw = os.path.join(res_ep, "raw_fakes")
    dir_align = os.path.join(res_ep, "aligned_masks")
    dir_samp = os.path.join(res_ep, "sampled_masks")
    dir_grid = os.path.join(res_ep, "grids")

    for d in [dir_comp, dir_raw, dir_align, dir_samp, dir_grid]:
        os.makedirs(d, exist_ok=True)

    fname_prefix = "" if args.no_phase_prefix else f"{args.phase}_"

    model_G = MedTurboRSNAModel(
        SD_TURBO_PATH,
        lora_rank_unet=LORA_RANK_UNET,
        lora_rank_vae=LORA_RANK_VAE,
    ).to(DEVICE)

    load_lora_checkpoint(model_G, ckpt, DEVICE)
    model_G.eval()

    with torch.no_grad():
        base_embeds = model_G.encode_text([PROMPT])

    root_p = Path(args.data_root)
    normal_dir = root_p / args.phase / NORMAL_DIR_NAME
    lung_dir = root_p / args.phase / LUNG_MASK_DIR_NAME
    pool_dir = Path(args.mask_pool_dir)

    if not normal_dir.exists():
        print(f"❌ {normal_dir} 不存在")
        sys.exit(1)

    if not pool_dir.exists():
        print(f"❌ Mask Pool {pool_dir} 不存在")
        sys.exit(1)

    pool_files = [p for p in pool_dir.iterdir() if p.suffix.lower() in VALID_EXT]

    if not pool_files:
        print("❌ Mask Pool 为空")
        sys.exit(1)

    print(f"Mask Pool: {len(pool_files)} 个 opacity mask")

    normal_files = sorted([p for p in normal_dir.iterdir() if p.suffix.lower() in VALID_EXT])

    if args.num_samples > 0:
        normal_files = normal_files[:args.num_samples]

    print(f"Normal CXR: {len(normal_files)} 张 → {res_ep}")
    print(f"composited_fakes: {dir_comp}")

    manifest_rows = []

    for img_path in tqdm(normal_files):
        stem = img_path.stem

        img_pil = Image.open(str(img_path)).convert("RGB")
        img_t = IMG_TF(img_pil).unsqueeze(0).to(DEVICE)

        img_bgr = cv2.cvtColor(
            np.array(img_pil.resize((512, 512), RESAMPLE_BILINEAR)),
            cv2.COLOR_RGB2BGR,
        )

        lung_np = None
        lp_found = None

        if lung_dir.exists():
            for ext in VALID_EXT:
                lp = lung_dir / (stem + ext)
                if lp.exists():
                    lp_found = lp
                    break

        if lp_found:
            lung_np = np.array(
                Image.open(str(lp_found))
                .convert("L")
                .resize((512, 512), RESAMPLE_NEAREST)
            )
        else:
            warnings.warn(f"{stem}: 缺少 lung mask，使用全1")
            lung_np = np.ones((512, 512), dtype=np.uint8) * 255

        lung_bgr = cv2.cvtColor(lung_np, cv2.COLOR_GRAY2BGR)

        sampled_path = random.choice(pool_files)
        sampled_np = np.array(
            Image.open(str(sampled_path))
            .convert("L")
            .resize((512, 512), RESAMPLE_BILINEAR)
        )
        sampled_bgr = cv2.cvtColor(sampled_np, cv2.COLOR_GRAY2BGR)

        aligned_f = align_mask_to_lung(sampled_np, lung_np, output_size=512)

        m_core = torch.from_numpy(aligned_f).unsqueeze(0).unsqueeze(0).to(DEVICE).float()

        lung_t = torch.from_numpy((lung_np > 127).astype(np.float32))
        lung_t = lung_t.unsqueeze(0).unsqueeze(0).to(DEVICE)

        _ = make_infer_mask(m_core, lung_t)

        m_comp = make_infer_composite_mask(
            m_core,
            lung_t,
            dilate_kernel=args.composite_mask_dilate,
            blur_kernel=args.composite_mask_blur,
        )

        with torch.no_grad():
            prompt_embeds = base_embeds.expand(1, -1, -1)

            with torch.amp.autocast("cuda"):
                fake_raw_t = model_G(
                    img_t,
                    m_core,
                    prompt_embeds,
                    timestep=INFER_TIMESTEP,
                )

            if USE_MASK_COMPOSITING:
                if args.use_delta_composite:
                    delta = fake_raw_t - img_t
                    fake_t = img_t + args.composite_alpha * m_comp * delta
                    fake_t = fake_t.clamp(-1.0, 1.0)
                else:
                    fake_t = fake_raw_t * m_comp + img_t * (1.0 - m_comp)
            else:
                fake_t = fake_raw_t

        raw_bgr = tensor_to_bgr(fake_raw_t)
        comp_bgr = tensor_to_bgr(fake_t)

        diff_np = np.abs(
            comp_bgr.astype(np.float32) - img_bgr.astype(np.float32)
        ).mean(axis=2)

        d_max = diff_np.max()

        diff_vis = cv2.applyColorMap(
            (diff_np / (d_max + 1e-5) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        out_name = f"{fname_prefix}{stem}.png"

        cv2.imwrite(os.path.join(dir_comp, out_name), comp_bgr)
        cv2.imwrite(os.path.join(dir_raw, out_name), raw_bgr)
        cv2.imwrite(os.path.join(dir_align, out_name), (aligned_f * 255).astype(np.uint8))

        cv2.imwrite(
            os.path.join(dir_samp, f"{fname_prefix}{sampled_path.stem}_{stem}.png"),
            sampled_np,
        )

        save_qc_7(
            img_bgr,
            lung_bgr,
            sampled_bgr,
            aligned_f,
            raw_bgr,
            comp_bgr,
            diff_vis,
            os.path.join(dir_grid, f"qc_{fname_prefix}{stem}.png"),
        )

        manifest_rows.append({
            "normal_filename": img_path.name,
            "output_filename": out_name,
            "sampled_mask_filename": sampled_path.name,
            "aligned_mask_area": f"{float(aligned_f.sum()) / (512 * 512):.5f}",
            "checkpoint": os.path.basename(ckpt),
            "phase": args.phase,
            "composite_alpha": f"{args.composite_alpha:.3f}",
            "composite_mask_dilate": args.composite_mask_dilate,
            "composite_mask_blur": args.composite_mask_blur,
            "use_delta_composite": args.use_delta_composite,
        })

    manifest_path = os.path.join(res_ep, "manifest.csv")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "normal_filename",
                "output_filename",
                "sampled_mask_filename",
                "aligned_mask_area",
                "checkpoint",
                "phase",
                "composite_alpha",
                "composite_mask_dilate",
                "composite_mask_blur",
                "use_delta_composite",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n✅ 推理完成！")
    print(f"composited_fakes → {dir_comp}")
    print(f"raw_fakes        → {dir_raw}")
    print(f"aligned_masks    → {dir_align}")
    print(f"sampled_masks    → {dir_samp}")
    print(f"grids            → {dir_grid}")
    print(f"manifest         → {manifest_path}")


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTurbo RSNA 肺炎推理")

    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--mask_pool_dir", default=None)
    parser.add_argument("--phase", default="test")
    parser.add_argument("--num_samples", type=int, default=0, help="0=全部")

    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出根目录；未指定时默认 /root/autodl-tmp/MedTurbo_RSNA/generated_fake_normal/{phase}",
    )

    parser.add_argument(
        "--no_phase_prefix",
        action="store_true",
        default=False,
        help="不在输出文件名前加 phase 前缀",
    )

    parser.add_argument("--composite_alpha", type=float, default=1.5)
    parser.add_argument("--composite_mask_dilate", type=int, default=15)
    parser.add_argument("--composite_mask_blur", type=int, default=31)

    parser.add_argument(
        "--use_delta_composite",
        dest="use_delta_composite",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no_delta_composite",
        dest="use_delta_composite",
        action="store_false",
    )

    args = parser.parse_args()

    if args.mask_pool_dir is None:
        args.mask_pool_dir = str(Path(args.data_root) / "train" / "Mask_Pool")

    if args.output_dir is None:
        args.output_dir = f"/root/autodl-tmp/MedTurbo_RSNA/generated_fake_normal/{args.phase}"

    run_inference(args)
