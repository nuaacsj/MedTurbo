"""
med_turbo_rsna.py  ——  MedTurbo RSNA 肺炎模型
"""

import os
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


# ════════════════════════════════════════════════════════════
#  LoRA Linear
# ════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """为现有 Linear 层追加低秩旁路。"""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.alpha  = alpha
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Linear(in_f, rank,  bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)


def inject_lora(
    module: nn.Module,
    rank:   int,
    target_names: tuple = ("to_q", "to_k", "to_v", "to_out.0"),
):
    """精确注入 LoRA：仅替换 target_names 对应的 Linear 层。"""
    for _name, submodule in module.named_modules():
        for target in target_names:
            parts = target.split(".")
            m = submodule
            try:
                for p in parts[:-1]:
                    m = getattr(m, p)
                leaf_name = parts[-1]
                leaf = getattr(m, leaf_name, None)
                if isinstance(leaf, nn.Linear) and not isinstance(leaf, LoRALinear):
                    setattr(m, leaf_name, LoRALinear(leaf, rank=rank, alpha=1.0))
            except AttributeError:
                continue


# ════════════════════════════════════════════════════════════
#  SpatialMaskAdapter（可训练，zero-init 最后一层）
# ════════════════════════════════════════════════════════════

class SpatialMaskAdapter(nn.Module):
    """
    opacity mask [B,1,H,W] → latent delta [B,4,h,w]

    设计要点：
      1. 输入 mask resize 到 latent size (h,w)
      2. 三层小 CNN，保持空间分辨率
      3. 最后一层 zero-init：训练初期 delta≈0，不破坏 SD-Turbo latent
      4. 全部参数 requires_grad=True
    """

    def __init__(self, out_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
        )
        # zero-init：保证初始 mask_delta ≈ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mask: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        """
        Args:
            mask:      [B, 1, H, W]，[0, 1]
            target_hw: (h, w) latent 空间尺寸
        Returns:
            delta: [B, 4, h, w]
        """
        m = F.interpolate(mask.float(), size=target_hw,
                          mode="bilinear", align_corners=False)
        return self.net(m)


# ════════════════════════════════════════════════════════════
#  MedTurboRSNAModel
# ════════════════════════════════════════════════════════════

class MedTurboRSNAModel(nn.Module):
    """
    RSNA 肺炎 opacity 合成模型。

    训练：
      pixel_values = source_img（degraded Lung_Opacity）
      masks        = opacity mask
      目标          = Lung_Opacity target（损失在训练脚本中计算）

    forward 流程：
      1. VAE Encoder：source → z_input
      2. 添加噪声：z_noisy（t = TRAIN_TIMESTEP）
      3. SpatialMaskAdapter：mask → latent delta
      4. z_noisy_cond = z_noisy + delta
      5. U-Net：z_noisy_cond → noise_pred
      6. 单步 DDPM x0 预测
      7. VAE Decoder（含 LoRA）→ fake_raw [-1, 1]
    """

    def __init__(
        self,
        sd_turbo_path:  str,
        lora_rank_unet: int = 8,
        lora_rank_vae:  int = 4,
    ):
        super().__init__()
        print(f"[MedTurboRSNAModel] 加载 SD-Turbo: {sd_turbo_path}")

        self.tokenizer    = CLIPTokenizer.from_pretrained(
            sd_turbo_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_turbo_path, subfolder="text_encoder")
        self.vae          = AutoencoderKL.from_pretrained(
            sd_turbo_path, subfolder="vae")
        self.unet         = UNet2DConditionModel.from_pretrained(
            sd_turbo_path, subfolder="unet")
        self.scheduler    = DDPMScheduler.from_pretrained(
            sd_turbo_path, subfolder="scheduler")

        # 冻结所有主干参数
        for component in [self.text_encoder, self.vae, self.unet]:
            for p in component.parameters():
                p.requires_grad = False

        # LoRA 注入（可训练）
        print(f"[MedTurboRSNAModel] 注入 LoRA: "
              f"unet_rank={lora_rank_unet}, vae_rank={lora_rank_vae}")
        inject_lora(self.unet,         rank=lora_rank_unet)
        inject_lora(self.vae.decoder,  rank=lora_rank_vae)

        # SpatialMaskAdapter（可训练）
        self.mask_adapter = SpatialMaskAdapter(out_channels=4)

        self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        print(f"[MedTurboRSNAModel] 就绪 | vae_scale={self.vae_scale}")

    # ── 文本编码（只调用一次，缓存 embeds）───────────────────
    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device
        tokens = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        return self.text_encoder(tokens)[0]

    # ── VAE 编码 ──────────────────────────────────────────────
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """RGB [-1,1] → scaled latent。"""
        return (self.vae.encode(x).latent_dist.sample()
                * self.vae.config.scaling_factor)

    # ── VAE 解码 ──────────────────────────────────────────────
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """scaled latent → RGB [-1,1]（VAE Decoder 含 LoRA）。"""
        return self.vae.decode(z / self.vae.config.scaling_factor).sample

    # ── Forward ───────────────────────────────────────────────
    def forward(
        self,
        pixel_values:  torch.Tensor,
        masks:         torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep:      Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values:  [B, 3, H, W]，[-1, 1]，训练时为 source_img
            masks:         [B, 1, H, W]，[0, 1]，opacity mask (m_core)
            prompt_embeds: [B, seq_len, D]
            timestep:      整数时间步（None 时默认 499）
        Returns:
            fake_raw: [B, 3, H, W]，[-1, 1]
        """
        B      = pixel_values.shape[0]
        device = pixel_values.device

        t = timestep if timestep is not None else 499
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

        # 1. 编码输入
        with torch.no_grad():
            z_input = self.encode_image(pixel_values)   # [B, 4, h, w]

        latent_hw = z_input.shape[-2:]

        # 2. 添加噪声
        noise   = torch.randn_like(z_input)
        z_noisy = self.scheduler.add_noise(z_input, noise, t_tensor)

        # 3. SpatialMaskAdapter → latent delta
        mask_delta   = self.mask_adapter(masks, latent_hw)   # [B, 4, h, w]
        z_noisy_cond = z_noisy + mask_delta

        # 4. U-Net 去噪
        noise_pred = self.unet(
            z_noisy_cond,
            t_tensor,
            encoder_hidden_states=prompt_embeds,
        ).sample   # [B, 4, h, w]

        # 5. 单步 DDPM x0 预测
        alphas  = self.scheduler.alphas_cumprod.to(device)
        alpha_t = alphas[t_tensor].view(B, 1, 1, 1).float()
        z_pred  = (
            z_noisy.float() - (1.0 - alpha_t).sqrt() * noise_pred.float()
        ) / alpha_t.sqrt()

        # 6. 解码
        fake_raw = self.decode_latent(z_pred.to(self.vae.dtype))

        # 7. Resize 回输入分辨率（VAE 输出尺寸与输入不一致时）
        if fake_raw.shape[-2:] != pixel_values.shape[-2:]:
            fake_raw = F.interpolate(
                fake_raw, size=pixel_values.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        return fake_raw


# ════════════════════════════════════════════════════════════
#  工具函数（与 skin 分支对齐）
# ════════════════════════════════════════════════════════════

def get_trainable_params(model: nn.Module) -> list:
    """返回所有 requires_grad=True 的参数。"""
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} K"
    return str(n)


def save_lora_checkpoint(model: nn.Module, epoch: int, path: str):
    """仅保存可训练参数（LoRA + SpatialMaskAdapter）。"""
    path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.detach().cpu()
             for k, v in model.named_parameters() if v.requires_grad}
    torch.save({"epoch": epoch, "model": state}, path)
    mb = sum(v.numel() * 4 for v in state.values()) / 1e6
    print(f"[save_lora_checkpoint] {path}  ({mb:.1f} MB)")


def load_lora_checkpoint(
    model:  nn.Module,
    path:   str,
    device: torch.device,
) -> int:
    """从 checkpoint 恢复可训练参数，返回 epoch。"""
    print(f"[load_lora_checkpoint] {path}")
    ckpt   = torch.load(path, map_location=device)
    params = dict(model.named_parameters())
    loaded, skipped = 0, 0
    for k, v in ckpt["model"].items():
        if k in params and params[k].requires_grad:
            params[k].data.copy_(v.to(device))
            loaded += 1
        else:
            skipped += 1
    epoch = ckpt.get("epoch", 0)
    print(f"  恢复：epoch={epoch}，加载 {loaded} 个参数，跳过 {skipped} 个")
    return epoch
