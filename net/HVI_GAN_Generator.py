import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *


class HVIGANGenerator(nn.Module):
    """
    GAN Generator with the same dual-channel (HV + I) architecture as CIDNet.

    The two processing streams are:
      - HV stream  : encodes/decodes the colour (Hue/Value) channels of the
                     HVI colour space.
      - I  stream  : encodes/decodes the Intensity (I) channel.

    Cross-channel communication between the two streams is provided by the
    same Lightweight Cross-Attention (LCA) modules used in CIDNet.

    Unlike the original CIDNet, `forward()` returns BOTH the reconstructed
    RGB image AND the residual HVI tensor so that two separate discriminators
    (one per colour space) can be trained simultaneously.
    """

    def __init__(
        self,
        channels: list[int] = [36, 36, 72, 144],
        heads:    list[int] = [1, 2, 4, 8],
        norm: bool = False,
    ):
        super().__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # ------------------------------------------------------------------ #
        # HV stream – encoder
        # ------------------------------------------------------------------ #
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # HV stream – decoder
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False),
        )

        # ------------------------------------------------------------------ #
        # I stream – encoder
        # ------------------------------------------------------------------ #
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # I stream – decoder
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        # ------------------------------------------------------------------ #
        # LCA cross-attention modules (6 per stream, mirroring CIDNet)
        # ------------------------------------------------------------------ #
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        # ------------------------------------------------------------------ #
        # HVI colour space transform (learnable density_k parameter)
        # ------------------------------------------------------------------ #
        self.trans = RGB_HVI()

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Low-light RGB image  (B, 3, H, W), values in [0, 1].

        Returns:
            output_rgb : Enhanced RGB image   (B, 3, H, W)
            output_hvi : Enhanced HVI tensor  (B, 3, H, W)
                         → fed to the HVI discriminator during training.
        """
        dtypes = x.dtype

        # --- RGB → HVI projection ---
        hvi = self.trans.HVIT(x)                           # (B,3,H,W)
        i   = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)     # (B,1,H,W)

        # ---- Encoder --------------------------------------------------------
        i_enc0  = self.IE_block0(i)
        i_enc1  = self.IE_block1(i_enc0)
        hv_0    = self.HVE_block0(hvi)
        hv_1    = self.HVE_block1(hv_0)

        i_jump0  = i_enc0
        hv_jump0 = hv_0

        # — LCA level 1 (ch2) —
        i_enc2   = self.I_LCA1(i_enc1, hv_1)
        hv_2     = self.HV_LCA1(hv_1,  i_enc1)
        v_jump1  = i_enc2
        hv_jump1 = hv_2
        i_enc2   = self.IE_block2(i_enc2)
        hv_2     = self.HVE_block2(hv_2)

        # — LCA level 2 (ch3) —
        i_enc3   = self.I_LCA2(i_enc2, hv_2)
        hv_3     = self.HV_LCA2(hv_2,  i_enc2)
        v_jump2  = i_enc3
        hv_jump2 = hv_3
        i_enc3   = self.IE_block3(i_enc2)   # note: uses i_enc2 as in original
        hv_3     = self.HVE_block3(hv_2)    # note: uses hv_2 as in original

        # — Bottleneck LCA (ch4) —
        i_enc4   = self.I_LCA3(i_enc3, hv_3)
        hv_4     = self.HV_LCA3(hv_3,  i_enc3)

        i_dec4   = self.I_LCA4(i_enc4, hv_4)
        hv_4     = self.HV_LCA4(hv_4,  i_enc4)

        # ---- Decoder --------------------------------------------------------
        hv_3    = self.HVD_block3(hv_4,   hv_jump2)
        i_dec3  = self.ID_block3(i_dec4,  v_jump2)

        # — LCA level 5 (ch3) —
        i_dec2  = self.I_LCA5(i_dec3, hv_3)
        hv_2    = self.HV_LCA5(hv_3,  i_dec3)

        hv_2    = self.HVD_block2(hv_2,   hv_jump1)
        i_dec2  = self.ID_block2(i_dec3,  v_jump1)   # matches original

        # — LCA level 6 (ch2) —
        i_dec1  = self.I_LCA6(i_dec2, hv_2)
        hv_1    = self.HV_LCA6(hv_2,  i_dec2)

        i_dec1  = self.ID_block1(i_dec1, i_jump0)
        i_dec0  = self.ID_block0(i_dec1)
        hv_1    = self.HVD_block1(hv_1,  hv_jump0)
        hv_0    = self.HVD_block0(hv_1)

        # ---- Output ---------------------------------------------------------
        # Residual addition in HVI space (same as CIDNet)
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi  # (B,3,H,W)
        output_rgb = self.trans.PHVIT(output_hvi)             # (B,3,H,W)

        return output_rgb, output_hvi

    # Convenience: project any RGB tensor to HVI (used to get gt_hvi)
    def HVIT(self, x: torch.Tensor) -> torch.Tensor:
        return self.trans.HVIT(x)
