import pytest
import torch

from deepinv.physics import (
    HyperSpectralUnmixing,
    PalimpsestAttenuation,
    PalimpsestOpticalDensity,
    canvas_size,
    center_crop,
    center_pad,
    estimate_substrate,
)

N_BANDS = 8
IMG_SIZES = [(N_BANDS, 32, 32), (4, 24, 32)]
ANGLE_SETS = [None, (0.0, 90.0), (15.0, 105.0)]


def palimpsest(img_size=(N_BANDS, 32, 32), n_layers=2, device="cpu", **kwargs):
    """Build a PalimpsestAttenuation with the defaults shared by the tests below."""
    return PalimpsestAttenuation(img_size, n_layers=n_layers, device=device, **kwargs)


def ink(physics, batch=1, device="cpu"):
    """Two perpendicular strokes, on whatever domain this operator's layers live on."""
    height, width = physics.layer_size
    x = torch.zeros(batch, physics.n_layers, height, width, device=device)
    x[:, 0, height // 4 : height // 4 + 3, :] = 1.0
    for k in range(1, physics.n_layers):
        x[:, k, :, width // 2 : width // 2 + 3] = 1.0
    return x


# ---------------------------------------------------------------------- geometry


@pytest.mark.parametrize("img_size", [(32, 32), (24, 32), (1, 64, 40)])
def test_canvas_covers_any_rotation(img_size):
    """The canvas must reach the circumscribed circle, or a rotation clips the corners."""
    side = canvas_size(img_size)
    height, width = img_size[-2], img_size[-1]

    assert side >= (height**2 + width**2) ** 0.5 - 1
    assert side >= max(height, width)


def test_canvas_matches_generator():
    """The operator and the mask generator must render on the same canvas."""
    for height, width in [(32, 32), (64, 40), (24, 32)]:
        assert canvas_size((height, width)) == int(2.0**0.5 * max(height, width)) + 1


@pytest.mark.parametrize("img_size", [(32, 32), (24, 32)])
def test_crop_pad_are_adjoint(img_size, device):
    """center_pad is the transpose of center_crop, which A_adjoint relies on."""
    torch.manual_seed(0)
    side = canvas_size(img_size)
    u = torch.rand(2, 3, side, side, device=device)
    v = torch.rand(2, 3, *img_size, device=device)

    lhs = (center_crop(u, img_size) * v).sum()
    rhs = (u * center_pad(v, side)).sum()

    assert torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_crop_undoes_pad(device):
    v = torch.rand(2, 3, 24, 32, device=device)

    assert torch.allclose(
        center_crop(center_pad(v, canvas_size((24, 32))), (24, 32)), v
    )


def test_layer_size_follows_geometry(device):
    """Rotated layers live on the canvas; unrotated ones live on the page."""
    flat = palimpsest(device=device)
    turned = palimpsest(angles=(0.0, 90.0), device=device)

    assert flat.layer_size == (32, 32)
    assert flat.canvas is None
    assert turned.layer_size == (46, 46)
    assert turned.canvas == 46


def observed_window(side: int, size: int) -> slice:
    """The slice center_crop keeps, used to blank out what the measurement already sees."""
    start = (side - size) // 2
    return slice(start, start + size)


def test_content_outside_the_frame_still_contributes(device):
    """
    The regression this geometry exists for.

    Ink lying outside the observed window but rotating into it must reach the measurement.
    Parametrizing the layers on the image grid instead forces that content to zero, which
    asserts the page is blank there rather than admitting it was never observed.
    """
    physics = palimpsest(angles=(45.0, 45.0), device=device)
    side = physics.canvas
    window = observed_window(side, 32)

    outside = torch.ones(1, 2, side, side, device=device)
    outside[..., window, window] = 0.0

    assert bool((physics.linearize().A(outside) > 1e-6).any())


# ---------------------------------------------------------------- forward model


@pytest.mark.parametrize("img_size", IMG_SIZES)
def test_forward_shape(img_size, device):
    physics = palimpsest(img_size, device=device)
    x = ink(physics, batch=2, device=device)

    y = physics(x)

    assert y.shape == (2, img_size[0], img_size[1], img_size[2])
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("angles", ANGLE_SETS)
@pytest.mark.parametrize("img_size", IMG_SIZES)
def test_measurement_is_on_the_image_grid(angles, img_size, device):
    """Whatever domain the layers live on, the measurement is the image."""
    physics = palimpsest(img_size, angles=angles, device=device)
    y = physics.A(ink(physics, device=device))

    assert y.shape == (1, img_size[0], img_size[1], img_size[2])


def test_blank_page_returns_substrate(device):
    physics = palimpsest(device=device)
    blank = physics.A(torch.zeros(1, 2, 32, 32, device=device))

    assert torch.allclose(blank, physics.substrate.expand_as(blank))


def test_ink_only_darkens(device):
    """Beer-Lambert attenuates, so the measurement never exceeds the bare substrate."""
    physics = palimpsest(device=device)
    y = physics.A(ink(physics, device=device))

    assert bool((y <= physics.substrate + 1e-6).all())
    assert bool((y > 0).all())


def test_more_ink_is_darker(device):
    physics = palimpsest(device=device)
    x = ink(physics, device=device)

    assert bool((physics.A(2.0 * x) <= physics.A(x) + 1e-6).all())


@pytest.mark.parametrize("n_layers", (1, 2, 3))
def test_layer_count(n_layers, device):
    physics = palimpsest(n_layers=n_layers, device=device)
    x = ink(physics, device=device)

    assert physics.n_layers == n_layers
    assert physics.absorption.shape == (n_layers, N_BANDS)
    assert physics.A(x).shape == (1, N_BANDS, 32, 32)


# ------------------------------------------------------------- log-linearisation


@pytest.mark.parametrize("angles", ANGLE_SETS)
def test_optical_density_linearises_exactly(angles, device):
    """-log(A(x) / R) is exactly the linear log-domain operator applied to x."""
    physics = palimpsest(angles=angles, device=device)
    x = ink(physics, device=device)

    density = physics.optical_density(physics.A(x))

    assert torch.allclose(density, physics.linearize().A(x), atol=1e-5)


def test_optical_density_of_substrate_is_zero(device):
    physics = palimpsest(device=device)
    y = physics.substrate.expand(1, N_BANDS, 32, 32)

    assert torch.allclose(physics.optical_density(y), torch.zeros_like(y), atol=1e-6)


def test_optical_density_clamps_dark_pixels(device):
    """Noise drives dark pixels to zero, where an unclamped log would diverge."""
    physics = palimpsest(device=device)
    y = torch.zeros(1, N_BANDS, 32, 32, device=device)

    density = physics.optical_density(y)

    assert torch.isfinite(density).all()


def test_substrate_scaling_cancels(device):
    """Halving the substrate halves the measurement and leaves the density unchanged."""
    bright = palimpsest(substrate=1.0, device=device)
    dim = palimpsest(substrate=0.5, device=device)
    x = ink(bright, device=device)

    assert torch.allclose(dim.A(x), 0.5 * bright.A(x), atol=1e-6)
    assert torch.allclose(
        dim.optical_density(dim.A(x)),
        bright.optical_density(bright.A(x)),
        atol=1e-5,
    )


# -------------------------------------------------------------- linear operator


@pytest.mark.parametrize("angles", ANGLE_SETS)
@pytest.mark.parametrize("img_size", IMG_SIZES)
def test_density_operator_is_adjoint(angles, img_size, device):
    """Rotation, spectral mixing and the crop must all transpose correctly."""
    torch.manual_seed(0)
    physics = palimpsest(img_size, angles=angles, device=device)
    x = torch.rand(2, physics.n_layers, *physics.layer_size, device=device)

    assert bool(physics.linearize().adjointness_test(x).abs() < 1e-4)


def test_density_operator_matches_unmixing(device):
    """With no per-layer geometry, the log-domain operator is a linear mixing model."""
    torch.manual_seed(0)
    absorption = torch.rand(3, N_BANDS, device=device)
    density = PalimpsestOpticalDensity(absorption, device=device)
    unmixing = HyperSpectralUnmixing(M=absorption, device=device)
    x = torch.rand(2, 3, 16, 16, device=device)

    assert torch.allclose(density.A(x), unmixing.A(x), atol=1e-6)
    assert torch.allclose(
        density.A_adjoint(density.A(x)), unmixing.A_adjoint(unmixing.A(x)), atol=1e-5
    )


def test_density_operator_is_linear(device):
    torch.manual_seed(0)
    physics = PalimpsestOpticalDensity(
        torch.rand(2, N_BANDS, device=device), device=device
    )
    u = torch.rand(1, 2, 16, 16, device=device)
    v = torch.rand(1, 2, 16, 16, device=device)

    assert torch.allclose(
        physics.A(u + 2.0 * v), physics.A(u) + 2.0 * physics.A(v), atol=1e-5
    )


def test_rotation_is_actually_applied(device):
    """Turning a layer changes the measurement, i.e. the geometry is not dropped."""
    straight = palimpsest(angles=(0.0, 0.0), device=device)
    turned = palimpsest(angles=(0.0, 90.0), device=device)
    x = ink(straight, device=device)

    assert not torch.allclose(
        straight.linearize().A(x), turned.linearize().A(x), atol=1e-3
    )


# ------------------------------------------------------------------- parameters


def test_default_absorption_rows_are_distinct(device):
    """Layers are separable only because their spectra differ."""
    absorption = PalimpsestAttenuation.default_absorption(3, N_BANDS)

    assert absorption.shape == (3, N_BANDS)
    assert bool((absorption >= 0).all())
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.allclose(absorption[i], absorption[j], atol=1e-3)


def test_update_parameters(device):
    physics = palimpsest(angles=(0.0, 90.0), device=device)
    new_absorption = torch.rand(2, N_BANDS, device=device)

    physics.update_parameters(absorption=new_absorption, angles=(10.0, 100.0))

    assert torch.allclose(physics.absorption, new_absorption)
    assert torch.allclose(physics.angles, torch.tensor([10.0, 100.0], device=device))


def test_update_substrate(device):
    physics = palimpsest(device=device)
    physics.update_parameters(substrate=0.8)

    blank = physics.A(torch.zeros(1, 2, 32, 32, device=device))

    assert torch.allclose(blank, torch.full_like(blank, 0.8), atol=1e-6)


@pytest.mark.parametrize("substrate", (0.9, torch.full((N_BANDS,), 0.9)))
def test_substrate_shapes(substrate, device):
    physics = palimpsest(substrate=substrate, device=device)
    blank = physics.A(torch.zeros(1, 2, 32, 32, device=device))

    assert torch.allclose(blank, torch.full_like(blank, 0.9), atol=1e-6)


def test_spatially_varying_substrate(device):
    substrate = torch.rand(N_BANDS, 32, 32, device=device) * 0.5 + 0.5
    physics = palimpsest(substrate=substrate, device=device)

    blank = physics.A(torch.zeros(1, 2, 32, 32, device=device))

    assert torch.allclose(blank, substrate.unsqueeze(0), atol=1e-6)


# ---------------------------------------------------------------- substrate est.


def test_estimate_substrate_recovers_flat_page(device):
    y = torch.ones(1, 4, 32, 32, device=device)
    y[:, :, 12:15, :] = 0.2

    substrate = estimate_substrate(y, window=9)

    assert substrate.shape == y.shape
    assert bool((substrate >= y - 1e-6).all())
    assert torch.allclose(substrate, torch.ones_like(substrate), atol=1e-3)


def test_estimate_substrate_rejects_even_window(device):
    with pytest.raises(ValueError, match="odd"):
        estimate_substrate(torch.ones(1, 1, 8, 8, device=device), window=8)


# ---------------------------------------------------------------------- errors


def test_rejects_wrong_layer_count(device):
    physics = palimpsest(device=device)
    with pytest.raises(ValueError, match="ink layers"):
        physics.A(torch.zeros(1, 5, 32, 32, device=device))


def test_rejects_layers_on_the_wrong_domain(device):
    """A rotated layer given on the image grid is the mistake this guards against."""
    physics = palimpsest(angles=(0.0, 90.0), device=device)
    with pytest.raises(ValueError, match="canvas"):
        physics.A(torch.zeros(1, 2, 32, 32, device=device))


def test_rejects_wrong_band_count(device):
    physics = palimpsest(device=device).linearize()
    with pytest.raises(ValueError, match="spectral bands"):
        physics.A_adjoint(torch.zeros(1, 3, 32, 32, device=device))


def test_rejects_angles_without_img_size():
    with pytest.raises(ValueError, match="img_size is required"):
        PalimpsestOpticalDensity(torch.rand(2, N_BANDS), angles=(0.0, 90.0))


def test_rejects_bad_img_size():
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        PalimpsestAttenuation((32, 32))


def test_rejects_negative_absorption():
    with pytest.raises(ValueError, match="non-negative"):
        PalimpsestOpticalDensity(-torch.ones(2, N_BANDS))


def test_rejects_nonpositive_substrate():
    with pytest.raises(ValueError, match="strictly positive"):
        PalimpsestAttenuation((N_BANDS, 32, 32), substrate=0.0)


def test_rejects_mismatched_angles():
    with pytest.raises(ValueError, match="one entry per ink layer"):
        PalimpsestOpticalDensity(
            torch.rand(2, N_BANDS), img_size=(32, 32), angles=(0.0, 45.0, 90.0)
        )


def test_rejects_mismatched_absorption_bands():
    with pytest.raises(ValueError, match="one column per band"):
        PalimpsestAttenuation((N_BANDS, 32, 32), absorption=torch.rand(2, 3))
