import pytest
import torch

import deepinv as dinv
from deepinv.models import TextLayerSeparator, contrast_background
from deepinv.physics import (
    CrosshatchTextOverlay,
    canvas_size,
    center_crop,
    center_pad,
    rotate,
    rotate_adjoint,
)

IMG_SIZES = [(1, 32, 32), (3, 32, 48)]
ANGLE_SETS = [(0.0,), (0.0, 90.0), (0.0, 45.0, 90.0)]


def overlay(img_size=(1, 32, 32), angles=(0.0, 90.0), device="cpu", **kwargs):
    """Build a CrosshatchTextOverlay with defaults shared by the tests below."""
    return CrosshatchTextOverlay(img_size, angles=angles, device=device, **kwargs)


def sources(physics, batch=1, device="cpu"):
    """Random sources on the canvas this operator parametrizes its layers on."""
    return torch.rand(batch, *physics.source_size, device=device)


@pytest.mark.parametrize("angle", (0.0, 30.0, 90.0, 180.0))
def test_rotate_shape_and_identity(angle, device):
    x = torch.rand(2, 3, 32, 32, device=device)
    out = rotate(x, angle)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    if angle == 0.0:
        assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.parametrize("angle", (0.0, 30.0, 90.0))
@pytest.mark.parametrize("mode", ("bilinear", "nearest"))
def test_rotate_adjoint_is_exact(angle, mode, device):
    """<rotate(u), v> == <u, rotate_adjoint(v)> to floating point accuracy."""
    torch.manual_seed(0)
    u = torch.rand(2, 1, 16, 16, device=device)
    v = torch.rand(2, 1, 16, 16, device=device)

    lhs = (rotate(u, angle, mode=mode) * v).sum()
    rhs = (u * rotate_adjoint(v, angle, mode=mode)).sum()

    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("img_size", IMG_SIZES)
@pytest.mark.parametrize("angles", ANGLE_SETS)
def test_overlay_shapes(img_size, angles, device):
    physics = overlay(img_size, angles, device=device)
    channels, height, width = img_size

    assert physics.n_layers == len(angles)
    side = canvas_size(img_size)
    assert physics.source_size == (len(angles) + 1, channels, side, side)

    x = sources(physics, 2, device)
    y = physics.A(x)
    assert y.shape == (2, channels, height, width)

    back = physics.A_adjoint(y)
    assert back.shape == x.shape


@pytest.mark.parametrize("img_size", IMG_SIZES)
@pytest.mark.parametrize("angles", ANGLE_SETS)
def test_overlay_adjointness(img_size, angles, device):
    """The operator is linear, so it must pass deepinv's adjointness test."""
    physics = overlay(img_size, angles, device=device)
    x = sources(physics, 2, device)

    assert physics.adjointness_test(x).abs() < 1e-4


def test_overlay_is_additive(device):
    """A(x) is the background plus each rotated, weighted layer, cropped to the image."""
    angles, amplitudes = (0.0, 90.0), (0.5, 0.25)
    physics = overlay((1, 32, 32), angles, amplitudes=amplitudes, device=device)

    x = sources(physics, 2, device)
    expected = center_crop(
        x[:, 0]
        + amplitudes[0] * rotate(x[:, 1], angles[0])
        + amplitudes[1] * rotate(x[:, 2], angles[1]),
        (1, 32, 32),
    )

    assert torch.allclose(physics.A(x), expected, atol=1e-6)


def test_overlay_linearity(device):
    physics = overlay(device=device)
    x1 = sources(physics, 1, device)
    x2 = sources(physics, 1, device)

    assert torch.allclose(
        physics.A(2.0 * x1 + 3.0 * x2),
        2.0 * physics.A(x1) + 3.0 * physics.A(x2),
        atol=1e-5,
    )


def test_overlay_zero_amplitude_drops_layer(device):
    """A layer with zero amplitude cannot influence the measurement."""
    physics = overlay(angles=(0.0, 90.0), amplitudes=(1.0, 0.0), device=device)

    x = sources(physics, 1, device)
    x_other = x.clone()
    x_other[:, 2] = torch.rand_like(x[:, 2])

    assert torch.allclose(physics.A(x), physics.A(x_other), atol=1e-6)


def test_angles_are_settable_by_hand(device):
    """Angles can be written by hand, persistently or for a single call."""
    physics = overlay(angles=(0.0, 90.0), device=device)
    x = sources(physics, 1, device)

    # Persistent update, accepting a plain tuple
    physics.update_parameters(angles=(12.0, 78.0))
    assert torch.allclose(physics.angles, torch.tensor([12.0, 78.0], device=device))

    # Set for a single call, and check it really is used
    y_called = physics(x, angles=(30.0, 120.0))
    assert torch.allclose(physics.angles, torch.tensor([30.0, 120.0], device=device))

    physics_ref = overlay(angles=(30.0, 120.0), device=device)
    assert torch.allclose(y_called, physics_ref(x), atol=1e-6)


def test_angles_accept_tensors_and_lists(device):
    physics = overlay(angles=(0.0, 90.0), device=device)

    for value in ([15.0, 45.0], (15.0, 45.0), torch.tensor([15.0, 45.0])):
        physics.update_parameters(angles=value)
        assert torch.allclose(physics.angles, torch.tensor([15.0, 45.0], device=device))


def test_changing_the_number_of_layers(device):
    physics = overlay(angles=(0.0, 90.0), device=device)
    assert physics.n_layers == 2

    physics.update_parameters(angles=(0.0, 60.0, 120.0), amplitudes=(1.0, 0.5, 0.5))
    assert physics.n_layers == 3

    x = sources(physics, 1, device)
    assert physics.A(x).shape == (1, 1, 32, 32)

    # Changing angles alone must not silently leave amplitudes the wrong length
    with pytest.raises(ValueError):
        physics.update_parameters(angles=(0.0, 90.0))


def test_per_sample_angles(device):
    """A (B,) angle tensor rotates each image of the batch by its own angle."""
    x = torch.rand(2, 1, 32, 32, device=device)
    angles = torch.tensor([0.0, 90.0], device=device)

    out = rotate(x, angles)

    assert torch.allclose(out[0], rotate(x[:1], 0.0)[0], atol=1e-6)
    assert torch.allclose(out[1], rotate(x[1:], 90.0)[0], atol=1e-6)

    with pytest.raises(ValueError):
        rotate(x, torch.tensor([0.0, 45.0, 90.0], device=device))


def test_angle_is_differentiable(device):
    """The angle carries gradient, so it can be refined by autograd if wanted."""
    x = torch.rand(1, 1, 32, 32, device=device)
    angle = torch.tensor(37.0, device=device, requires_grad=True)

    rotate(x, angle).pow(2).sum().backward()

    assert angle.grad is not None
    assert torch.isfinite(angle.grad).all()
    assert angle.grad.abs() > 0


def test_overlay_errors(device):
    with pytest.raises(ValueError):
        overlay(angles=(), device=device)

    with pytest.raises(ValueError):
        overlay(angles=(0.0, 90.0), amplitudes=(1.0,), device=device)

    # Wrong number of source components
    physics = overlay(angles=(0.0, 90.0), device=device)
    with pytest.raises(ValueError):
        physics.A(torch.rand(1, 2, 1, 32, 32, device=device))


def test_overlay_canvas_geometry(device):
    """Sources live on the canvas, the measurement on the image grid."""
    physics = overlay((1, 32, 32), (0.0, 90.0), device=device)

    assert physics.canvas == 46
    assert physics.source_size == (3, 1, 46, 46)
    assert physics.A(sources(physics, 1, device)).shape == (1, 1, 32, 32)


def test_overlay_rejects_sources_on_the_image_grid(device):
    """Passing layers on the image grid is the mistake the canvas guards against."""
    physics = overlay((1, 32, 32), (0.0, 90.0), device=device)

    with pytest.raises(ValueError, match="canvas"):
        physics.A(torch.rand(1, 3, 1, 32, 32, device=device))


def test_overlay_content_outside_the_frame_reaches_the_measurement(device):
    """
    The regression the canvas exists for.

    Text lying outside the observed window but rotating into it must reach the measurement.
    Parametrizing the layers on the image grid forces that content to zero instead.
    """
    physics = overlay((1, 32, 32), (45.0, 45.0), device=device)
    side = physics.canvas
    start = (side - 32) // 2

    x = torch.zeros(1, 3, 1, side, side, device=device)
    x[:, 1] = 1.0
    x[:, 1, :, start : start + 32, start : start + 32] = (
        0.0  # blank the observed window
    )

    assert bool((physics.A(x) > 1e-6).any())


def test_overlay_adjoint_places_the_measurement_back_on_the_canvas(device):
    """A crops, so its adjoint pads: the background back-projection is the padded image."""
    physics = overlay((1, 32, 32), (0.0, 90.0), device=device)
    y = torch.rand(1, 1, 32, 32, device=device)

    back = physics.A_adjoint(y)

    assert back.shape == (1, 3, 1, 46, 46)
    assert torch.allclose(back[:, 0], center_pad(y, 46), atol=1e-6)


@pytest.mark.parametrize("n_layers", (1, 2, 3))
@pytest.mark.parametrize("in_channels", (1, 3))
def test_separator_shapes(n_layers, in_channels, device):
    angles = tuple(float(90 * k) for k in range(n_layers))
    physics = overlay((in_channels, 32, 32), angles, device=device)
    model = TextLayerSeparator(
        in_channels=in_channels, n_layers=n_layers, device=device
    )

    y = torch.rand(2, in_channels, 32, 32, device=device)
    x_hat = model(y, physics)

    side = canvas_size((in_channels, 32, 32))
    assert x_hat.shape == (2, n_layers + 1, in_channels, side, side)
    assert x_hat.dtype == y.dtype


def test_separator_enforces_consistency(device):
    """With the consistency step, the separated sources reproduce the measurement."""
    physics = overlay((1, 32, 32), (0.0, 90.0), device=device)
    model = TextLayerSeparator(in_channels=1, n_layers=2, device=device)

    y = torch.rand(2, 1, 32, 32, device=device)
    x_hat = model(y, physics)

    assert torch.allclose(physics.A(x_hat), y, atol=1e-4)


def test_separator_without_consistency(device):
    """Without the consistency step the model runs, but no longer reproduces y exactly."""
    model = TextLayerSeparator(
        in_channels=1, n_layers=2, enforce_consistency=False, device=device
    )
    y = torch.rand(2, 1, 32, 32, device=device)

    x_hat = model(y, canvas=46)  # physics is not needed here, only the canvas
    assert x_hat.shape == (2, 3, 1, 46, 46)


def test_separator_nonnegative_layers(device):
    """Text layers are clamped to be non-negative; the background stays free."""
    physics = overlay((1, 32, 32), (0.0, 90.0), device=device)
    model = TextLayerSeparator(in_channels=1, n_layers=2, device=device)

    y = torch.rand(2, 1, 32, 32, device=device)
    x_hat = model(y, physics)

    assert (x_hat[:, 1:] >= 0).all()


def test_separator_requires_physics_for_consistency(device):
    model = TextLayerSeparator(in_channels=1, n_layers=2, device=device)
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 32, 32, device=device))


def test_separator_errors(device):
    with pytest.raises(ValueError):
        TextLayerSeparator(in_channels=1, n_layers=0, device=device)


def test_separator_can_fit_a_fixed_example(device):
    """A few steps of gradient descent on one example must reduce the loss."""
    torch.manual_seed(0)
    physics = overlay((1, 32, 32), (0.0, 90.0), device=device)
    model = TextLayerSeparator(in_channels=1, n_layers=2, device=device)

    x = sources(physics, 1, device)
    y = physics.A(x)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        loss = torch.nn.functional.mse_loss(model(y, physics), x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


@pytest.mark.parametrize("in_channels", (1, 3))
def test_contrast_background(in_channels, device):
    """The rendered layer is in range and separates glyphs from the ground behind them."""
    layer = torch.zeros(2, in_channels, 16, 16, device=device)
    layer[:, :, 8, :] = 1.0  # a bright bar of "text"

    out = contrast_background(layer)

    assert out.shape == layer.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # The glyph row must differ from the untouched rows behind it
    assert not torch.allclose(out[:, :, 8, :], out[:, :, 0, :])


def test_contrast_background_handles_empty_layer(device):
    """An all-zero layer must not divide by zero."""
    out = contrast_background(torch.zeros(1, 3, 16, 16, device=device))

    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_separation_end_to_end_with_text_generator(device):
    """
    Compose real text layers through the physics and check the pipeline runs.

    The generator's ``canvas=True`` layers are exactly the domain the operator parametrizes its
    sources on, so the two halves of the pipeline agree without any resizing in between.
    """
    img_size = (1, 64, 64)
    physics = overlay(img_size, (0.0, 90.0), amplitudes=(0.4, 0.4), device=device)
    side = physics.canvas

    gen = dinv.physics.generator.CrosshatchTextMaskGenerator(
        img_size, text="DEEPINV", angles=(0.0,), device=device
    )
    upright = gen.layer_fields(canvas=True)[0]
    assert upright.shape == (side, side)

    background = torch.rand(1, 1, side, side, device=device) * 0.5
    layer = upright.expand(1, 1, side, side)
    x = torch.stack([background, layer, layer], dim=1)
    y = physics.A(x)

    # Adding text on top can only brighten the background, over the observed window
    assert (y >= center_crop(background, img_size) - 1e-6).all()

    model = TextLayerSeparator(in_channels=1, n_layers=2, device=device)
    x_hat = model(y, physics)

    assert x_hat.shape == x.shape
    assert torch.allclose(physics.A(x_hat), y, atol=1e-4)
