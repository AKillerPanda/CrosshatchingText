from __future__ import annotations

import torch
import torch.nn.functional as F

from deepinv.physics.forward import LinearPhysics


def rotate(
    x: torch.Tensor, angle: float | torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    r"""
    Rotate a batch of images about their centre.

    :param torch.Tensor x: images of shape ``(B, C, H, W)``.
    :param float, torch.Tensor angle: rotation angle in degrees. A scalar rotates every image of
        the batch by the same angle; a tensor of shape ``(B,)`` gives each image its own angle.
        Passing a tensor that requires grad makes the rotation differentiable with respect to the
        angle itself.
    :param str mode: interpolation passed to :func:`torch.nn.functional.grid_sample`.
    :return: rotated images, of the same shape as ``x``. Content rotated outside the frame is
        dropped and the corners it leaves behind are filled with zeros.
    :rtype: torch.Tensor
    """
    theta = torch.as_tensor(angle, device=x.device, dtype=x.dtype).reshape(-1)
    if theta.numel() == 1:
        theta = theta.expand(x.shape[0])
    elif theta.numel() != x.shape[0]:
        raise ValueError(
            f"angle must be a scalar or have one entry per image, got {theta.numel()} "
            f"angles for a batch of {x.shape[0]}."
        )

    theta = theta * torch.pi / 180.0
    cos, sin = torch.cos(theta), torch.sin(theta)

    # The transpose of R(theta), since grid_sample maps output coordinates back to input ones.
    mat = torch.zeros(x.shape[0], 2, 3, device=x.device, dtype=x.dtype)
    mat[:, 0, 0] = cos
    mat[:, 0, 1] = sin
    mat[:, 1, 0] = -sin
    mat[:, 1, 1] = cos

    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def rotate_adjoint(
    y: torch.Tensor, angle: float | torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    r"""
    Exact adjoint of :func:`deepinv.physics.rotate`.

    Resampling is a gather, so its adjoint is a scatter and *not* a rotation by the opposite
    angle: the two differ wherever interpolation weights do not sum to one, i.e. at the
    boundary. The adjoint is obtained here as the vector-Jacobian product of ``rotate``, which
    is exact because ``rotate`` is linear in its input.

    :param torch.Tensor y: images of shape ``(B, C, H, W)``.
    :param float, torch.Tensor angle: rotation angle in degrees of the forward operator.
    :param str mode: interpolation used by the forward operator.
    :return: tensor of the same shape as ``y``.
    :rtype: torch.Tensor
    """
    with torch.enable_grad():
        v = torch.zeros_like(y, requires_grad=True)
        out = rotate(v, angle, mode=mode)
        (grad,) = torch.autograd.grad(
            out, v, grad_outputs=y, create_graph=torch.is_grad_enabled()
        )
    return grad


class CrosshatchTextOverlay(LinearPhysics):
    r"""Additive text-overlay operator, for separating crosshatched text from an image.

    Models an image carrying several layers of text, each written at its own angle:

    ``y = a + w_1 * rotate(s_1, angle_1) + ... + w_K * rotate(s_K, angle_K)``

    where ``a`` is the background image, ``s_k`` is the *upright* text field of layer ``k``, and
    ``angle_k`` is the angle that layer is written at. The unknowns are stacked into a single
    tensor ``x`` of shape ``(B, K+1, C, H, W)``, with ``x[:, 0]`` the background and
    ``x[:, k+1]`` the upright text of layer ``k``, so the operator is linear in ``x`` and this is
    an ordinary :class:`deepinv.physics.LinearPhysics` with one measurement and ``K+1`` sources.

    Parametrizing the unknowns as *upright* text rather than rotated text is deliberate: it puts
    the rotation in the operator, so a reconstruction method only ever has to model horizontal
    text, and the recovered layers come back readable.

    This is the additive counterpart of the multiplicative masks produced by
    :class:`deepinv.physics.generator.CrosshatchTextMaskGenerator`: there the text erases pixels,
    here it is added on top of them, which is what a watermark or a caption does.

    .. note::

        Rotation is applied in place on the ``(H, W)`` grid, so text rotated out of the frame is
        lost and the corners it vacates are zero. The operator is therefore not exactly
        invertible even with ``K`` known layers, which is expected for an overlay model.

    ``angles`` and ``amplitudes`` are registered buffers rather than fixed attributes, so they can
    be written by hand at any point with :meth:`update_parameters`, or passed to a single call as
    ``physics(x, angles=...)``, without rebuilding the operator. Set them from whatever you know
    about the image; nothing here estimates them.

    :param tuple[int] img_size: size of the background image, of shape `(C, H, W)`.
    :param tuple[float], torch.Tensor angles: angle in degrees of each text layer. Its length
        ``K`` sets the number of text sources, so ``x`` has ``K+1`` components.
    :param tuple[float], torch.Tensor amplitudes: per-layer weight ``w_k``. If ``None``, every
        layer has weight 1.
    :param str mode: interpolation used to rotate the layers.
    :param str, torch.device device: device where the operator lives.

    |sep|

    :Examples:

        Compose a background with two text layers at 0 and 90 degrees:

        >>> import torch
        >>> from deepinv.physics import CrosshatchTextOverlay
        >>> physics = CrosshatchTextOverlay((1, 32, 32), angles=(0.0, 90.0))
        >>> x = torch.zeros(1, 3, 1, 32, 32)   # (background, layer 1, layer 2)
        >>> x[:, 0] = 0.5                      # flat grey background
        >>> x[:, 1, :, 16] = 1.0               # a horizontal bar of "text"
        >>> y = physics(x)
        >>> y.shape
        torch.Size([1, 1, 32, 32])

        The operator is linear, so it passes the adjointness test:

        >>> bool(physics.adjointness_test(x).abs() < 1e-4)
        True

        Write the angles by hand, either persistently or for a single call:

        >>> physics.update_parameters(angles=(12.0, 78.0))
        >>> physics.angles
        tensor([12., 78.])
        >>> y = physics(x, angles=(30.0, 120.0))   # set them for this call
        >>> physics.angles
        tensor([ 30., 120.])

        Changing the number of layers just means giving the amplitudes too:

        >>> physics.update_parameters(angles=(0.0, 60.0, 120.0), amplitudes=(1.0, 0.5, 0.5))
        >>> physics.n_layers
        3
    """

    def __init__(
        self,
        img_size: tuple[int],
        angles: tuple[float] | torch.Tensor = (0.0, 90.0),
        amplitudes: tuple[float] | torch.Tensor = None,
        mode: str = "bilinear",
        device: str | torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        angles = self._as_vector(angles, "angles")
        amplitudes = (
            torch.ones_like(angles)
            if amplitudes is None
            else self._as_vector(amplitudes, "amplitudes")
        )
        self._check_lengths(angles, amplitudes)

        self.img_size = img_size
        self.mode = mode
        self.register_buffer("angles", angles)
        self.register_buffer("amplitudes", amplitudes)
        self.to(device)

    @staticmethod
    def _as_vector(value, name: str) -> torch.Tensor:
        vector = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if vector.numel() == 0:
            raise ValueError(f"{name} must have at least one entry.")
        return vector

    @staticmethod
    def _check_lengths(angles: torch.Tensor, amplitudes: torch.Tensor) -> None:
        if angles.numel() != amplitudes.numel():
            raise ValueError(
                f"amplitudes must have one entry per angle, got {amplitudes.numel()} "
                f"amplitudes for {angles.numel()} angles."
            )

    @property
    def n_layers(self) -> int:
        r"""Number of text layers ``K``, i.e. one fewer than the number of sources."""
        return self.angles.numel()

    def update_parameters(self, angles=None, amplitudes=None, **kwargs):
        r"""
        Set the angles, and optionally the amplitudes, of the text layers.

        Accepts plain tuples and lists as well as tensors, so the angles can be written by hand
        without building a new operator::

            physics.update_parameters(angles=(12.0, 78.0))

        The same values can be passed straight to a call, as ``physics(x, angles=(12.0, 78.0))``.

        :param tuple, torch.Tensor angles: new angle of each text layer, in degrees.
        :param tuple, torch.Tensor amplitudes: new weight of each text layer.
        """
        if angles is not None:
            angles = self._as_vector(angles, "angles").to(self.angles.device)
        if amplitudes is not None:
            amplitudes = self._as_vector(amplitudes, "amplitudes").to(
                self.amplitudes.device
            )

        # Changing the number of layers requires both to be given, and to agree.
        self._check_lengths(
            self.angles if angles is None else angles,
            self.amplitudes if amplitudes is None else amplitudes,
        )
        super().update_parameters(angles=angles, amplitudes=amplitudes, **kwargs)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Compose the background and the rotated text layers into one image.

        :param torch.Tensor x: sources of shape ``(B, K+1, C, H, W)``.
        :param dict kwargs: optionally ``angles`` and ``amplitudes``, to set them for this call.
        :return: image of shape ``(B, C, H, W)``.
        :rtype: torch.Tensor
        """
        self.update_parameters(**kwargs)

        if x.shape[1] != self.n_layers + 1:
            raise ValueError(
                f"x must have {self.n_layers + 1} components for "
                f"{self.n_layers} text layers, got {x.shape[1]}."
            )

        y = x[:, 0]
        for k, (angle, amplitude) in enumerate(zip(self.angles, self.amplitudes)):
            y = y + amplitude * rotate(x[:, k + 1], angle, mode=self.mode)
        return y

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Back-project a measurement onto each source.

        :param torch.Tensor y: image of shape ``(B, C, H, W)``.
        :param dict kwargs: optionally ``angles`` and ``amplitudes``, to set them for this call.
        :return: sources of shape ``(B, K+1, C, H, W)``.
        :rtype: torch.Tensor
        """
        self.update_parameters(**kwargs)

        components = [y] + [
            amplitude * rotate_adjoint(y, angle, mode=self.mode)
            for angle, amplitude in zip(self.angles, self.amplitudes)
        ]
        return torch.stack(components, dim=1)
