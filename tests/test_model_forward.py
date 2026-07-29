"""Forward-pass shape + param-count smoke test for LVTEGModel.

Verifies paper Sec 3.4 / Sec impl numbers:
    * 4-channel input accepted
    * Output sequence length = W / 4 (paper Sec 3.5: T = W/4)
    * Total ~23.58M params (paper Implementation Details)
"""

import pytest
import torch

from lvt_eg.models.lvt_eg import LVTEGModel


@pytest.mark.parametrize("img_h,img_w", [(48, 192), (32, 128)])
def test_forward_shape(img_h, img_w):
    model = LVTEGModel(num_classes=37, img_h=img_h, img_w=img_w, use_stn=True).eval()
    B = 2
    lr = torch.randn(B, 3, img_h, img_w)
    with torch.no_grad():
        log_probs = model(lr)
    # RCTC reader collapses H, returns [B, W/4, num_classes]
    assert log_probs.shape == (B, img_w // 4, 37)


def test_forward_with_hr_returns_edges():
    model = LVTEGModel(num_classes=37, img_h=48, img_w=192, use_stn=True).eval()
    lr = torch.randn(1, 3, 48, 192)
    hr = torch.randn(1, 3, 48, 192)
    with torch.no_grad():
        log_probs, pred_edge, gt_edge = model(lr, hr_images=hr, return_all=True)
    assert pred_edge.shape == (1, 1, 48, 192)
    assert gt_edge.shape == (1, 1, 48, 192)


def test_param_count_within_paper_range():
    """Paper Implementation Details: 23.58M total = 23.20M SVTRv2-Base (enc+RCTC)
    + 284K STN + 94.4K Visual-Tactile Branch. The notebook prints the full model
    as 'SVTRv2EdgeOCR: 23.58M params'. Allow +/- 5% slack."""
    model = LVTEGModel(num_classes=37, img_h=48, img_w=192, use_stn=True)
    total_m = sum(p.numel() for p in model.parameters()) / 1e6
    assert 22.0 <= total_m <= 26.0, f"Got {total_m:.2f}M (paper: 23.58M)"


def test_ccpd_legacy_conv_mixer_param_count():
    """CCPD (notebook v60a) used the grouped-conv ConvMixer (no BatchNorm), ~3.4M
    lighter than the v61a full-conv+BN variant. The trained CCPD model is 20.12M
    (notebook print: 'Total: 20.12M | SVTRv2 enc: 17.65M | RCTC dec: 2.10M').
    legacy_conv_mixer=True must reproduce it; the default (False) is the heavier
    23.52M variant. Guards against the ConvMixer re-implementation drift."""
    legacy = LVTEGModel(num_classes=68, img_h=32, img_w=128, use_stn=True,
                        use_pos_embed=False, legacy_conv_mixer=True)
    full = LVTEGModel(num_classes=68, img_h=32, img_w=128, use_stn=True,
                      use_pos_embed=False, legacy_conv_mixer=False)
    legacy_m = sum(p.numel() for p in legacy.parameters()) / 1e6
    full_m = sum(p.numel() for p in full.parameters()) / 1e6
    assert 19.8 <= legacy_m <= 20.4, f"CCPD legacy got {legacy_m:.2f}M (paper: 20.12M)"
    assert full_m - legacy_m > 3.0, f"flag had no effect: {full_m:.2f} vs {legacy_m:.2f}"


def test_visual_tactile_branch_param_count():
    """Paper Sec 3.3: Visual-Tactile Branch = 94.4K params."""
    model = LVTEGModel(num_classes=37, img_h=48, img_w=192, use_stn=True)
    # `imaginator.net` excludes the frozen Sobel buffer (not a Parameter).
    branch_k = sum(p.numel() for p in model.imaginator.net.parameters()) / 1e3
    assert 90.0 <= branch_k <= 100.0, f"Got {branch_k:.1f}K (paper: 94.4K)"


def test_no_edge_ablation_model():
    """use_edge=False (paper Table VI row 7): RGB-only 3-channel model with no
    imaginator and no edge outputs - lighter than the full 4-channel model.
    Guards against the edge branch being un-removable (edge_l1_weight=0 alone
    would NOT reproduce the no-edge ablation)."""
    full = LVTEGModel(num_classes=37, img_h=48, img_w=192, use_stn=True)
    no_edge = LVTEGModel(num_classes=37, img_h=48, img_w=192, use_stn=True,
                         use_edge=False).eval()
    assert not hasattr(no_edge, "imaginator")
    assert no_edge.stn.localization[0].in_channels == 3
    lr = torch.randn(2, 3, 48, 192)
    with torch.no_grad():
        log_probs, pred_edge, gt_edge = no_edge(
            lr, hr_images=torch.randn(2, 3, 48, 192), return_all=True)
    assert log_probs.shape == (2, 48, 37)
    assert pred_edge is None and gt_edge is None
    full_m = sum(p.numel() for p in full.parameters())
    ne_m = sum(p.numel() for p in no_edge.parameters())
    assert ne_m < full_m, f"no-edge ({ne_m}) should be lighter than full ({full_m})"


def test_rect_after_stn_order():
    """stn_order='rect_after' (paper Table VI Rect.=after rows): STN rectifies the
    fused 4-channel [RGB, edge] tensor; forward still returns edges + shape."""
    model = LVTEGModel(num_classes=37, img_h=48, img_w=192, use_stn=True,
                       stn_order="rect_after").eval()
    assert model.stn.localization[0].in_channels == 4  # STN sees RGB+edge
    lr = torch.randn(1, 3, 48, 192)
    hr = torch.randn(1, 3, 48, 192)
    with torch.no_grad():
        log_probs, pred_edge, gt_edge = model(lr, hr_images=hr, return_all=True)
    assert log_probs.shape == (1, 48, 37)
    assert pred_edge.shape == (1, 1, 48, 192)
    assert gt_edge.shape == (1, 1, 48, 192)


def test_stn_first_default_channels():
    """Default (stn_order='stn_first'): STN rectifies the 3-channel RGB crop."""
    model = LVTEGModel(num_classes=37, img_h=48, img_w=192, use_stn=True)
    assert model.stn.localization[0].in_channels == 3


def test_stn_order_invalid_raises():
    with pytest.raises(ValueError):
        LVTEGModel(num_classes=37, stn_order="bogus")
