from __future__ import annotations

import torch
import torch.nn as nn

from deepinv.models.unet import UNet


def contrast_background(layer: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Render a text layer on a background that contrasts with the text itself.

    A separated text layer is mostly zeros, so displaying it directly gives near-black glyphs on
    black. This composites the layer over a flat background chosen to be far from the mean colour
    of the glyphs: the background is the per-channel complement of that mean, pushed to full
    contrast in luminance, so the text stays readable whatever colour it happens to be.

    :param torch.Tensor layer: text layer of shape ``(B, C, H, W)``, zero away from the glyphs.
    :param float eps: guards the division when a layer is empty.
    :return: image of shape ``(B, C, H, W)`` in ``[0, 1]``, glyphs over a contrasting flat colour.
    :rtype: torch.Tensor
    """
    batch, channels = layer.shape[0], layer.shape[1]

    # Where the layer is active, and the mean colour of the glyphs there.
    alpha = layer.abs().amax(dim=1, keepdim=True)
    alpha = alpha / alpha.amax(dim=(1, 2, 3), keepdim=True).clamp_min(eps)

    weight = alpha.sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    ink = (layer * alpha).sum(dim=(1, 2, 3), keepdim=True) / weight

    # Complement of the ink colour, forced to the opposite end of the luminance range.
    background = 1.0 - ink.clamp(0.0, 1.0)
    luminance = ink.clamp(0.0, 1.0).mean(dim=1, keepdim=True)
    background = torch.where(luminance < 0.5, background.clamp_min(0.75), background.clamp_max(0.25))
    background = background.expand(batch, channels, 1, 1)

    ink_image = layer.clamp(0.0, 1.0)
    return background * (1.0 - alpha) + ink_image * alpha


class TextLayerSeparator(nn.Module):
    r"""Separate an image into a background and several rotated text layers.

    Inverts :class:`deepinv.physics.CrosshatchTextOverlay`: given

    ``y = a + w_1 * rotate(s_1, angle_1) + ... + w_K * rotate(s_K, angle_K)``

    the network predicts all ``K+1`` sources at once, returning a tensor of shape
    ``(B, K+1, C, H, W)`` whose component 0 is the background ``a`` and whose component ``k+1`` is
    the *upright* text of layer ``k``. With the default ``n_layers=2`` that is the three images
    of the usual crosshatch setting: the clean background, the first text, and the second.

    A single U-Net backbone predicts every source, which lets it use the fact that the sources
    must add back up to the measurement. That constraint is then imposed exactly when
    ``enforce_consistency`` is set: the residual ``y - A(x)`` is folded back into the background,
    so the returned sources satisfy ``A(x) == y`` to floating-point accuracy and the network only
    has to decide how to *split* the image, never how to preserve it.

    :param int in_channels: number of channels of the image, e.g. 1 for grayscale or 3 for colour.
    :param int n_layers: number of text layers ``K`` to separate. The output has ``K+1``
        components.
    :param torch.nn.Module backbone: network mapping ``(B, C, H, W)`` to
        ``(B, (K+1)*C, H, W)``. If ``None``, a :class:`deepinv.models.UNet` is built.
    :param bool nonnegative: clamp the text layers to be non-negative, since added text only ever
        brightens the background in this model.
    :param bool enforce_consistency: fold the residual back into the background so the sources
        reproduce the measurement exactly.
    :param str, torch.device device: device to put the model on.

    |sep|

    :Examples:

        Separate a two-layer overlay:

        >>> import torch
        >>> from deepinv.models import TextLayerSeparator
        >>> from deepinv.physics import CrosshatchTextOverlay
        >>> physics = CrosshatchTextOverlay((1, 32, 32), angles=(0.0, 90.0))
        >>> model = TextLayerSeparator(in_channels=1, n_layers=2)
        >>> y = torch.rand(2, 1, 32, 32)
        >>> x_hat = model(y, physics)
        >>> x_hat.shape
        torch.Size([2, 3, 1, 32, 32])

        With ``enforce_consistency`` the sources add back up to the measurement:

        >>> bool(torch.allclose(physics.A(x_hat), y, atol=1e-4))
        True
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_layers: int = 2,
        backbone: nn.Module = None,
        nonnegative: bool = True,
        enforce_consistency: bool = True,
        device: str | torch.device = torch.device("cpu"),
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}.")

        self.in_channels = in_channels
        self.n_layers = n_layers
        self.n_components = n_layers + 1
        self.nonnegative = nonnegative
        self.enforce_consistency = enforce_consistency

        self.backbone = (
            UNet(
                in_channels=in_channels,
                out_channels=in_channels * self.n_components,
                residual=False,
                scales=3,
                device=device,
            )
            if backbone is None
            else backbone
        )
        self.to(device)

    def forward(self, y: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        r"""
        Separate a measurement into its sources.

        :param torch.Tensor y: measurement of shape ``(B, C, H, W)``.
        :param deepinv.physics.CrosshatchTextOverlay physics: the overlay operator. Required when
            ``enforce_consistency`` is set, since the residual is computed through it.
        :return: sources of shape ``(B, K+1, C, H, W)``.
        :rtype: torch.Tensor
        """
        batch, _, height, width = y.shape

        out = self.backbone(y)
        x = out.reshape(batch, self.n_components, self.in_channels, height, width)

        if self.nonnegative:
            # Added text only brightens, so the layers are non-negative; the background is free.
            x = torch.cat([x[:, :1], x[:, 1:].relu()], dim=1)

        if self.enforce_consistency:
            if physics is None:
                raise ValueError(
                    "physics must be passed when enforce_consistency is True, as the residual "
                    "y - A(x) is computed through the forward operator."
                )
            residual = y - physics.A(x)
            x = torch.cat([x[:, :1] + residual.unsqueeze(1), x[:, 1:]], dim=1)

        return x
