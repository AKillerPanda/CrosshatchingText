from __future__ import annotations

import torch
import torch.nn.functional as F

from deepinv.physics.forward import LinearPhysics, Physics
from deepinv.physics.textoverlay import (
    canvas_size,
    center_crop,
    center_pad,
    rotate,
    rotate_adjoint,
)


def estimate_substrate(
    y: torch.Tensor, window: int = 31, smooth: bool = True
) -> torch.Tensor:
    r"""
    Estimate the ink-free substrate reflectance from a multispectral measurement.

    Parchment reflectance varies slowly across the sheet, while ink is high-frequency and always
    *darker* than the support it sits on. A grey-scale dilation (a local maximum) over a window
    wider than the thickest stroke therefore erases the writing and leaves the substrate, which is
    then smoothed to remove the blockiness the dilation introduces.

    This is a starting point, not a calibration: it biases the estimate upwards wherever the window
    is too small to clear a stroke, and it will happily erase a genuinely dark region of the
    support. Prefer a measured blank area of the same sheet when one is available.

    :param torch.Tensor y: measurement of shape ``(B, C, H, W)``.
    :param int window: side of the dilation window in pixels, which should exceed the widest
        stroke. Must be odd.
    :param bool smooth: average the dilated image over the same window, so the estimate does not
        inherit the staircase edges of the maximum filter.
    :return: substrate estimate of shape ``(B, C, H, W)``.
    :rtype: torch.Tensor

    |sep|

    :Examples:

        >>> import torch
        >>> from deepinv.physics import estimate_substrate
        >>> y = torch.ones(1, 4, 32, 32)
        >>> y[:, :, 12:16, :] = 0.2          # a dark stroke across the page
        >>> substrate = estimate_substrate(y, window=9)
        >>> substrate.shape
        torch.Size([1, 4, 32, 32])
        >>> bool((substrate >= y - 1e-6).all())   # the estimate never sits below the data
        True
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}.")

    padding = window // 2
    substrate = F.max_pool2d(y, kernel_size=window, stride=1, padding=padding)
    if smooth:
        # count_include_pad=False, so the border is not darkened by the zero padding.
        substrate = F.avg_pool2d(
            substrate,
            kernel_size=window,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
    return substrate


class PalimpsestOpticalDensity(LinearPhysics):
    r"""Linear mixing of ink layers in optical-density space.

    The log-domain counterpart of :class:`deepinv.physics.PalimpsestAttenuation`, mapping ink
    densities to the optical density they produce:

    .. math::

        d_c = \sum_{k} a_{k,c} \, G_k(x_k)

    where :math:`x_k \geq 0` is the density map of ink :math:`k`, :math:`a_{k,c}` its absorption in
    spectral band :math:`c`, and :math:`G_k` an optional rotation placing that layer on the page.
    This is linear in :math:`x`, which is the whole reason for working in optical density rather
    than in reflectance.

    With ``angles=None`` the operator is exactly the linear mixing model of
    :class:`deepinv.physics.HyperSpectralUnmixing`, the ink absorption spectra playing the role of
    endmembers. It is kept separate because a palimpsest additionally needs the per-layer geometry
    :math:`G_k`, and because its measurement is an optical density derived from a reflectance image
    rather than a directly observed quantity.

    You normally obtain this operator from
    :meth:`PalimpsestAttenuation.linearize <deepinv.physics.PalimpsestAttenuation.linearize>`
    rather than building it by hand.

    :param torch.Tensor absorption: absorption spectra of shape ``(K, C)``, non-negative, one row
        per ink layer and one column per spectral band.
    :param tuple[int] img_size: spatial size ``(H, W)`` of the measurement, which the rotated
        layers are cropped down to. Required when ``angles`` is given.
    :param tuple[float], torch.Tensor angles: rotation in degrees applied to each layer, of length
        ``K``. If ``None`` (default), the layers already live in the frame of the page and no
        resampling happens.
    :param str mode: interpolation used to rotate the layers.
    :param str, torch.device device: device where the operator lives.

    |sep|

    :Examples:

        Two inks over eight bands, mixed with no geometry:

        >>> import torch
        >>> from deepinv.physics import PalimpsestOpticalDensity
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> absorption = torch.rand(2, 8)
        >>> physics = PalimpsestOpticalDensity(absorption)
        >>> x = torch.rand(1, 2, 16, 16)
        >>> physics.A(x).shape
        torch.Size([1, 8, 16, 16])

        The operator is linear, with or without the per-layer rotations. Rotated layers live on a
        canvas and the measurement is cropped out of it, so the two have different spatial sizes:

        >>> bool(physics.adjointness_test(x).abs() < 1e-4)
        True
        >>> turned = PalimpsestOpticalDensity(absorption, img_size=(16, 16), angles=(0.0, 90.0))
        >>> turned.layer_size
        (23, 23)
        >>> turned.A(torch.rand(1, 2, 23, 23)).shape
        torch.Size([1, 8, 16, 16])
        >>> bool(turned.adjointness_test(torch.rand(1, 2, 23, 23)).abs() < 1e-4)
        True
    """

    def __init__(
        self,
        absorption: torch.Tensor,
        img_size: tuple[int] | None = None,
        angles: tuple[float] | torch.Tensor | None = None,
        mode: str = "bilinear",
        device: str | torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        absorption = torch.as_tensor(absorption, dtype=torch.float32)
        if absorption.ndim != 2:
            raise ValueError(
                f"absorption must be of shape (K, C), got {tuple(absorption.shape)}."
            )
        if bool((absorption < 0).any()):
            raise ValueError("absorption must be non-negative.")
        if angles is not None and img_size is None:
            raise ValueError(
                "img_size is required when angles are given, since the rotated layers live on a "
                "canvas that is cropped down to it."
            )

        self.mode = mode
        self.img_size = None if img_size is None else tuple(img_size[-2:])
        self.register_buffer("absorption", absorption)
        self.register_buffer(
            "angles", None if angles is None else self._as_angles(angles, absorption)
        )
        self.to(device)

    @staticmethod
    def _as_angles(angles, absorption: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(angles, dtype=torch.float32).reshape(-1)
        if vector.numel() != absorption.shape[0]:
            raise ValueError(
                f"angles must have one entry per ink layer, got {vector.numel()} angles "
                f"for {absorption.shape[0]} layers."
            )
        return vector

    @property
    def n_layers(self) -> int:
        r"""Number of ink layers ``K``."""
        return self.absorption.shape[0]

    @property
    def n_bands(self) -> int:
        r"""Number of spectral bands ``C``."""
        return self.absorption.shape[1]

    @property
    def canvas(self) -> int | None:
        r"""Side ``S`` of the canvas the layers live on, or ``None`` if they are not rotated."""
        return None if self.angles is None else canvas_size(self.img_size)

    @property
    def layer_size(self) -> tuple[int, ...] | None:
        r"""
        Spatial shape of one ink layer.

        ``(S, S)`` when the layers are rotated, since a rotated layer needs a canvas wide enough
        to still cover the image, and ``(H, W)`` otherwise, when a layer already lives in the
        frame of the page.
        """
        if self.angles is not None:
            side = self.canvas
            return (side, side)
        return self.img_size

    def _warp(self, x: torch.Tensor) -> torch.Tensor:
        if self.angles is None:
            return x
        return torch.cat(
            [
                rotate(x[:, k : k + 1], self.angles[k], mode=self.mode)
                for k in range(self.n_layers)
            ],
            dim=1,
        )

    def _warp_adjoint(self, u: torch.Tensor) -> torch.Tensor:
        if self.angles is None:
            return u
        return torch.cat(
            [
                rotate_adjoint(u[:, k : k + 1], self.angles[k], mode=self.mode)
                for k in range(self.n_layers)
            ],
            dim=1,
        )

    def update_parameters(self, absorption=None, angles=None, **kwargs):
        r"""
        Set the absorption spectra and/or the per-layer rotations.

        :param torch.Tensor absorption: new absorption spectra of shape ``(K, C)``.
        :param tuple, torch.Tensor angles: new rotation of each layer, in degrees.
        """
        if absorption is not None:
            absorption = torch.as_tensor(absorption, dtype=torch.float32)
        if angles is not None:
            reference = self.absorption if absorption is None else absorption
            angles = self._as_angles(angles, reference)
        super().update_parameters(absorption=absorption, angles=angles, **kwargs)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Mix the ink layers into a per-band optical density.

        Each layer is rotated onto the page, the layers are mixed by their absorption spectra, and
        the result is cropped to the image. Mixing before cropping is the same as cropping each
        layer first, since both are linear.

        :param torch.Tensor x: ink densities of shape ``(B, K, *layer_size)``.
        :param dict kwargs: optionally ``absorption`` and ``angles``, to set them for this call.
        :return: optical density of shape ``(B, C, H, W)``.
        :rtype: torch.Tensor
        """
        self.update_parameters(**kwargs)
        if x.shape[1] != self.n_layers:
            raise ValueError(
                f"x must have {self.n_layers} ink layers, got {x.shape[1]}."
            )
        if self.layer_size is not None and tuple(x.shape[-2:]) != self.layer_size:
            raise ValueError(
                f"ink layers must be of size {self.layer_size}, got {tuple(x.shape[-2:])}. "
                "Rotated layers live on a canvas; see deepinv.physics.canvas_size."
            )

        density = torch.einsum("kc,bkhw->bchw", self.absorption, self._warp(x))
        if self.angles is None:
            return density
        return center_crop(density, self.img_size).contiguous()

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Back-project an optical density onto each ink layer.

        The adjoint of the crop is a zero-padding back onto the canvas, applied before the
        spectra and the rotations are back-projected in turn.

        :param torch.Tensor y: optical density of shape ``(B, C, H, W)``.
        :param dict kwargs: optionally ``absorption`` and ``angles``, to set them for this call.
        :return: ink densities of shape ``(B, K, *layer_size)``.
        :rtype: torch.Tensor
        """
        self.update_parameters(**kwargs)
        if y.shape[1] != self.n_bands:
            raise ValueError(
                f"y must have {self.n_bands} spectral bands, got {y.shape[1]}."
            )

        padded = y if self.angles is None else center_pad(y, canvas_size(self.img_size))
        return self._warp_adjoint(
            torch.einsum("kc,bchw->bkhw", self.absorption, padded)
        )


class PalimpsestAttenuation(Physics):
    r"""Multispectral attenuation operator for overwritten manuscripts.

    Models a palimpsest — a sheet whose original writing was erased and written over — imaged in
    several spectral bands. Ink absorbs light on the way in and on the way out, so the layers
    combine *multiplicatively* rather than additively, following Beer-Lambert:

    .. math::

        y_c = R_c \exp \left( - \sum_{k} a_{k,c} \, G_k(x_k) \right)

    where :math:`y_c` is the reflectance in band :math:`c`, :math:`R_c` the reflectance of the
    ink-free substrate, :math:`x_k \geq 0` the density map of ink :math:`k`, :math:`a_{k,c}` its
    absorption in that band, and :math:`G_k` an optional rotation placing that layer on the page —
    undertext often runs across the overtext, because the original leaves were cut and rebound
    turned.

    The unknowns are the ink densities, of shape ``(B, K, *layer_size)``: one density map per
    layer, *not* one image per layer. Each layer lives on its own domain — the page grid when it
    is not rotated, and a wider canvas when it is, so that a layer turning into the frame is
    parametrized rather than forced to zero. See :attr:`layer_size` and
    :func:`deepinv.physics.canvas_size`.

    What separates the layers is that inks of different composition absorb
    differently across bands, so a layer that is invisible in one band survives in another. That is
    the property multispectral manuscript imaging exists to exploit, and it is what makes the
    problem tractable where a single grayscale photograph would not be.

    .. note::

        Taking the negative log of :math:`y / R` turns this into a **linear** mixing model, which
        is what :meth:`optical_density` and :meth:`linearize` provide. Work in that domain: it
        gives an exact adjoint, and the whole of :mod:`deepinv.optim` becomes available. The
        nonlinearity then sits only in the change of variables, not in the operator being
        inverted.

    .. warning::

        This is a forward model for a real degradation, but not a calibrated one. Genuine
        manuscripts add parchment texture, ink bleed-through from the reverse side, non-rigid
        warping of the sheet, and faded ink whose absorption is neither uniform nor known. Treat
        results on synthetic data as evidence that the method runs, never as evidence that a real
        undertext was recovered.

    :param tuple[int] img_size: size of the measurement without batch dimension, of shape
        ``(C, H, W)``, where ``C`` is the number of spectral bands.
    :param torch.Tensor absorption: absorption spectra of shape ``(K, C)``, non-negative. If
        ``None``, ``n_layers`` spectra are built by :meth:`default_absorption`.
    :param int n_layers: number of ink layers ``K``. Ignored if ``absorption`` is given.
    :param float, torch.Tensor substrate: reflectance :math:`R` of the ink-free support, as a
        scalar, a per-band vector of shape ``(C,)``, or a map of shape ``(C, H, W)``. Estimate it
        from a measurement with :func:`deepinv.physics.estimate_substrate`.
    :param tuple[float], torch.Tensor angles: rotation in degrees of each ink layer, of length
        ``K``. If ``None`` (default), the layers already live in the frame of the page. Use this
        only when the undertext really is a rigid rotation of an upright field; real sheets warp.
    :param str mode: interpolation used to rotate the layers.
    :param str, torch.device device: device where the operator lives.

    |sep|

    :Examples:

        A two-ink palimpsest over eight bands, the undertext running across the overtext:

        >>> import torch
        >>> from deepinv.physics import PalimpsestAttenuation
        >>> physics = PalimpsestAttenuation((8, 32, 32), n_layers=2)
        >>> x = torch.zeros(1, 2, 32, 32)
        >>> x[:, 0, 8:12, :] = 1.0    # overtext stroke
        >>> x[:, 1, :, 20:24] = 1.0   # undertext stroke, perpendicular
        >>> y = physics(x)
        >>> y.shape
        torch.Size([1, 8, 32, 32])

        Ink only ever darkens the page, so the measurement never exceeds the bare substrate:

        >>> bool((y <= physics.substrate + 1e-6).all())
        True
        >>> blank = physics.A(torch.zeros(1, 2, 32, 32))
        >>> bool(torch.allclose(blank, physics.substrate.expand_as(blank)))
        True

        The negative log linearises the model exactly, which is where reconstruction happens:

        >>> density = physics.optical_density(y)
        >>> bool(torch.allclose(density, physics.linearize().A(x), atol=1e-5))
        True
        >>> bool(physics.linearize().adjointness_test(x).abs() < 1e-4)
        True

        Rotating one layer onto the other is a per-layer geometry. Each layer then lives on its
        own canvas, wide enough that turning it still covers the page, and the measurement is the
        centre crop of their mixture:

        >>> turned = PalimpsestAttenuation((8, 32, 32), n_layers=2, angles=(0.0, 90.0))
        >>> turned.layer_size
        (46, 46)
        >>> x_canvas = torch.zeros(1, 2, 46, 46)
        >>> x_canvas[:, 1, :, 20:24] = 1.0
        >>> turned.A(x_canvas).shape
        torch.Size([1, 8, 32, 32])

        It stays linear in the log domain, the crop transposing to a zero-padding:

        >>> bool(turned.linearize().adjointness_test(x_canvas).abs() < 1e-4)
        True
    """

    def __init__(
        self,
        img_size: tuple[int],
        absorption: torch.Tensor = None,
        n_layers: int = 2,
        substrate: float | torch.Tensor = 1.0,
        angles: tuple[float] | torch.Tensor | None = None,
        mode: str = "bilinear",
        device: str | torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        if len(img_size) != 3:
            raise ValueError(
                f"img_size must be of the form (C, H, W), got {tuple(img_size)}."
            )
        n_bands = img_size[0]

        if absorption is None:
            if n_layers < 1:
                raise ValueError(f"n_layers must be at least 1, got {n_layers}.")
            absorption = self.default_absorption(n_layers, n_bands)
        else:
            absorption = torch.as_tensor(absorption, dtype=torch.float32)
            if absorption.ndim == 2 and absorption.shape[1] != n_bands:
                raise ValueError(
                    f"absorption must have one column per band, got "
                    f"{absorption.shape[1]} columns for {n_bands} bands."
                )

        self.img_size = tuple(img_size)
        self.density = PalimpsestOpticalDensity(
            absorption,
            img_size=self.img_size,
            angles=angles,
            mode=mode,
            device=device,
        )
        self.register_buffer("substrate", self._as_substrate(substrate, n_bands))
        self.to(device)

    @staticmethod
    def default_absorption(n_layers: int, n_bands: int) -> torch.Tensor:
        r"""
        Build a default set of ink absorption spectra.

        Row ``k`` decays as ``exp(-b * d_k)`` over bands ``b`` linearly spaced in ``[0, 1]``, with
        the decay rates ``d_k`` spread over ``[0.5, 8]``. Slowly-decaying rows stand for inks that
        stay opaque across the whole range, fast-decaying ones for inks that turn transparent
        towards the far bands. The rows are distinct by construction, which is what makes the
        layers separable at all; the values are plausible, not measured.

        :param int n_layers: number of ink layers ``K``.
        :param int n_bands: number of spectral bands ``C``.
        :return: absorption spectra of shape ``(K, C)``.
        :rtype: torch.Tensor
        """
        bands = torch.linspace(0.0, 1.0, n_bands)
        decay = torch.linspace(0.5, 8.0, n_layers)
        return torch.exp(-bands[None, :] * decay[:, None])

    @staticmethod
    def _as_substrate(value, n_bands: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1, 1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.reshape(1, -1, 1, 1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(
                "substrate must be a scalar, of shape (C,), (C, H, W) or (B, C, H, W), got "
                f"shape {tuple(tensor.shape)}."
            )

        if tensor.shape[1] not in (1, n_bands):
            raise ValueError(
                f"substrate must have 1 or {n_bands} bands, got {tensor.shape[1]}."
            )
        if bool((tensor <= 0).any()):
            raise ValueError(
                "substrate must be strictly positive, since the optical density divides by it."
            )
        return tensor

    @property
    def n_layers(self) -> int:
        r"""Number of ink layers ``K``."""
        return self.density.n_layers

    @property
    def n_bands(self) -> int:
        r"""Number of spectral bands ``C``."""
        return self.density.n_bands

    @property
    def absorption(self) -> torch.Tensor:
        r"""Absorption spectra of shape ``(K, C)``."""
        return self.density.absorption

    @property
    def angles(self) -> torch.Tensor | None:
        r"""Rotation in degrees of each ink layer, or ``None`` if the layers are not rotated."""
        return self.density.angles

    @property
    def canvas(self) -> int | None:
        r"""Side ``S`` of the canvas the layers live on, or ``None`` if they are not rotated."""
        return self.density.canvas

    @property
    def layer_size(self) -> tuple[int, ...] | None:
        r"""
        Spatial shape of one ink layer, ``(S, S)`` when rotated and ``(H, W)`` otherwise.

        A rotated layer needs a canvas wide enough that it still covers the image after turning,
        so it does not live on the same grid as the measurement.
        """
        return self.density.layer_size

    def linearize(self) -> PalimpsestOpticalDensity:
        r"""
        The linear operator mapping ink densities to optical density.

        Together with :meth:`optical_density` this turns the problem into an ordinary linear
        inverse problem, which can be handed to :func:`deepinv.optim.optim_builder` with one prior
        per ink layer::

            d = physics.optical_density(y)
            x_hat = model(d, physics.linearize())

        :return: the log-domain operator, sharing this operator's buffers.
        :rtype: deepinv.physics.PalimpsestOpticalDensity
        """
        return self.density

    def optical_density(
        self, y: torch.Tensor, substrate: torch.Tensor = None, eps: float = 1e-6
    ) -> torch.Tensor:
        r"""
        Convert a reflectance measurement into optical density.

        Computes :math:`-\log(y / R)`, the change of variables that turns the multiplicative
        attenuation into the linear mixing model of :meth:`linearize`. The ratio is clamped below,
        since noise drives dark pixels to zero where the logarithm would diverge; clamping
        saturates the densest strokes rather than letting them dominate the fit.

        :param torch.Tensor y: reflectance of shape ``(B, C, H, W)``.
        :param torch.Tensor substrate: substrate reflectance to divide by. Defaults to this
            operator's own, which is what :meth:`A` used.
        :param float eps: floor applied to the ratio before the logarithm.
        :return: optical density of shape ``(B, C, H, W)``, non-negative wherever ``y`` does not
            exceed the substrate.
        :rtype: torch.Tensor
        """
        reference = self.substrate if substrate is None else substrate
        return -torch.log((y / reference).clamp_min(eps))

    def update_parameters(self, substrate=None, **kwargs):
        r"""
        Set the substrate reflectance, the absorption spectra and/or the per-layer rotations.

        :param float, torch.Tensor substrate: new substrate reflectance.
        :param dict kwargs: forwarded to
            :meth:`PalimpsestOpticalDensity.update_parameters <deepinv.physics.PalimpsestOpticalDensity.update_parameters>`.
        """
        if substrate is not None:
            substrate = self._as_substrate(substrate, self.n_bands).to(
                self.substrate.device
            )
            super().update_parameters(substrate=substrate)
        if kwargs:
            self.density.update_parameters(**kwargs)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Attenuate the substrate by the ink layers.

        :param torch.Tensor x: ink densities of shape ``(B, K, H, W)``, non-negative.
        :param dict kwargs: optionally ``substrate``, ``absorption`` and ``angles``, to set them
            for this call.
        :return: reflectance of shape ``(B, C, H, W)``.
        :rtype: torch.Tensor
        """
        self.update_parameters(**kwargs)
        return self.substrate * torch.exp(-self.density.A(x))
