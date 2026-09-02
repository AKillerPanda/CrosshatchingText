from deepinv.physics.generator import (
    GaussianMaskGenerator,
    EquispacedMaskGenerator,
    RandomMaskGenerator,
    PolyOrderMaskGenerator,
)
from deepinv.physics.generator.base import seed_from_string
import pytest
import numpy as np
import torch
import deepinv as dinv
import itertools
from pathlib import Path

# Avoiding nondeterministic algorithms
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if not torch.cuda.is_available() and torch.__version__ >= "2.1.0":
    torch.use_deterministic_algorithms(True)

torch.backends.cudnn.deterministic = True

# Generators to test (make sure they appear in find_generator as well)
GENERATORS = [
    "GaussianBlurGenerator",
    "MotionBlurGenerator",
    "DiffractionBlurGenerator",
    "ProductConvolutionBlurGenerator",
    "SigmaGenerator",
]

MIXTURES = list(itertools.combinations(GENERATORS, 2))
# To test GeneratorMixture.use_batch_sampling feature, when compatible
# generators (same output keys and shapes), samples from different generators
# per batch element
MIXTURES += [("MotionBlurGenerator", "MotionBlurGenerator")]
MIXTURES += [("DiffractionBlurGenerator", "DiffractionBlurGenerator")]

SIZES = [(5, 5), (6, 6)]
NUM_CHANNELS = [1, 3]

# MRI Generators
C, T, H, W = 2, 12, 256, 512
MRI_GENERATORS = ["gaussian", "random", "uniform", "poly"]
MRI_IMG_SIZES = [(H, W), (C, H, W), (C, T, H, W), (64, 64)]
MRI_ACCELERATIONS = [4, 10, 12]
MRI_CENTER_FRACTIONS = [0, 0.04, 24 / 512]

# Inpainting/Splitting Generators
INPAINTING_IMG_SIZES = [
    (2, 64, 40),
    (2, 1000),  # This will show warning but (C, M) is valid
    (2, 3, 64, 40),
]  # (C,H,W), (C,M), (C,T,H,W)
INPAINTING_GENERATORS = ["bernoulli", "gaussian", "multiplicative"]

# Crosshatched Text Generators.
# Unlike the splitting generators above, these are not random subsampling masks with a
# split ratio: they render structured text. They feed two different problems -- step()
# gives an occlusion mask for Inpainting, layer_fields() gives the individual text
# sources for the additive CrosshatchTextOverlay separation model.
CROSSHATCH_GENERATORS = [
    "layered",
    "hatch",
    "random_shift",
    "random_angles",
    "multitext",
]
CROSSHATCH_IMG_SIZES = [(64, 40), (1, 64, 40), (3, 32, 32)]  # (H,W), (C,H,W)

DTYPES = [torch.float32, torch.float64]


# Fixture returns either None or a torch.Generator on the specified device, to test both cases:
# 1. an existing generator is passed to the physics generator
# 2. the physics generator creates its own default generator (according to the device values)
# with this fixture, generator.rng.device and device always match.
@pytest.fixture(params=[None, pytest.param("device", marks=pytest.mark.indirect)])
def rng(request, device):
    if request.param == "device":
        return torch.Generator(device=device).manual_seed(0)
    return None


def find_generator(name, size, device, dtype, psf_size=None, rng=None):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda:0
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    if name == "GaussianBlurGenerator":
        g = dinv.physics.generator.GaussianBlurGenerator(
            psf_size=size, device=device, dtype=dtype
        )
        keys = ["filter"]
    elif name == "MotionBlurGenerator":
        g = dinv.physics.generator.MotionBlurGenerator(
            psf_size=size,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        keys = ["filter"]
    elif name == "DiffractionBlurGenerator":
        g = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        keys = ["filter", "coeff", "pupil", "fc"]
    elif name == "ProductConvolutionBlurGenerator":
        g = dinv.physics.generator.ProductConvolutionBlurGenerator(
            psf_generator=dinv.physics.generator.DiffractionBlurGenerator(
                psf_size=size,
                device=device,
                dtype=dtype,
                rng=rng,
            ),
            img_size=512,
            n_eigen_psf=10,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        keys = ["filters", "multipliers"]
    elif name == "DownsamplingGenerator":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[2, 4],
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "DownsamplingGenerator2":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[2],
            psf_size=psf_size,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "DownsamplingGenerator4":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[4],
            psf_size=psf_size,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "DownsamplingGenerator[2, 4]":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[2, 4],
            psf_size=psf_size,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "SigmaGenerator":
        g = dinv.physics.generator.SigmaGenerator(device=device, dtype=dtype, rng=rng)
        keys = ["sigma"]
    elif name == "GainGenerator":
        g = dinv.physics.generator.GainGenerator(device=device, dtype=dtype, rng=rng)
        keys = ["gain"]
    else:
        raise Exception("The generator chosen doesn't exist")
    return g, size, keys


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_shape(name, size, device, dtype, rng):
    r"""
    Tests generators shape. All blur generators produce single-channel output by default;
    multi-channel (colour) output is tested separately in test_diffraction_generator.
    """

    generator, size, keys = find_generator(name, size, device, dtype, rng=rng)
    batch_size = 4

    params = generator.step(batch_size=batch_size)

    assert list(params.keys()) == keys

    if "filter" in params.keys():
        assert params["filter"].shape == (batch_size, 1, size[0], size[1])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_generation_newparams(name, device, dtype, rng):
    r"""
    Tests generators' ability to generate new parameters at each step.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, device, dtype, rng=rng)
    batch_size = 1

    if name == "GaussianBlurGenerator":
        param_key = ["filter"]
    elif name == "MotionBlurGenerator":
        param_key = ["filter"]
    elif name == "DiffractionBlurGenerator":
        param_key = ["filter"]
    elif name == "ProductConvolutionBlurGenerator":
        param_key = ["filters", "multipliers"]
    elif name == "SigmaGenerator":
        param_key = ["sigma"]

    params0 = generator.step(batch_size=batch_size, seed=0)
    params1 = generator.step(batch_size=batch_size, seed=1)

    for key in param_key:
        assert torch.any(params0[key] != params1[key])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_generation_seed(name, device, dtype, rng):
    r"""
    Tests generators consistency with the same random seed.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, device, dtype, rng=rng)
    batch_size = 1

    if name == "GaussianBlurGenerator":
        param_key = ["filter"]
    elif name == "MotionBlurGenerator":
        param_key = ["filter"]
    elif name == "DiffractionBlurGenerator":
        param_key = ["filter"]
    elif name == "ProductConvolutionBlurGenerator":
        param_key = ["filters", "multipliers"]
    elif name == "SigmaGenerator":
        param_key = ["sigma"]

    params0 = generator.step(batch_size=batch_size, seed=42)
    params1 = generator.step(batch_size=batch_size, seed=42)

    for key in param_key:
        assert torch.allclose(params0[key], params1[key])


@pytest.mark.parametrize(
    "name", sorted(set(GENERATORS).difference(set(["ProductConvolutionBlurGenerator"])))
)
@pytest.mark.parametrize("dtype", [torch.float64])
def test_average(name, device, dtype, rng):
    r"""
    Tests generators average.
    """
    size = (5, 5)
    generator, size, _ = find_generator(name, size, device, dtype, rng=rng)
    # Set generator seed for reproducibility
    generator.rng_manual_seed(0)

    n_avg = 4

    # Store the keys of a single step call for future comparison
    params = generator.step(batch_size=1, seed=0)
    keys = set(params.keys())

    for batch_size in [1, 2, n_avg]:
        batch_size = 1
        params = generator.average(5, batch_size=batch_size)
        assert isinstance(params, dict)
        assert set(params.keys()) == keys


#################################
### DOWNSAMPLING GENERATORS #####
#################################


@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("psf_size", [None, (31, 31)])
@pytest.mark.parametrize("fact", [None, 2, 4, [2, 4]])
def test_downsampling_generator(num_channels, device, dtype, psf_size, fact, rng):
    r"""
    Test downsampling generator.
    This test is different from the above ones because we do not generate a random kernel at each iteration, but
    we sample from a list.
    """
    # we need sufficiently large sizes to ensure well definedness of the operation
    size = (32, 32)

    str_fact = "" if fact is None else str(fact)

    physics = dinv.physics.Downsampling(
        img_size=(num_channels, size[0], size[1]),
        device=device,
        filter="bicubic",
        factor=4,
    )
    generator, _, _ = find_generator(
        "DownsamplingGenerator" + str_fact,
        size,
        device,
        dtype,
        psf_size=psf_size,
        rng=rng,
    )

    batch_size = (
        1 if fact is None else 128
    )  # Must be 1 as filters with different shapes can't be batched (case psf_size=None)

    if psf_size is None and batch_size > 1:
        # in this case, we have a generator that generates filters of different shapes
        with pytest.raises(ValueError):
            params = generator.step(batch_size=batch_size, seed=1)
    else:
        params = generator.step(batch_size=batch_size, seed=1)

        x = torch.randn(
            (batch_size, num_channels, size[0], size[1]),
            generator=generator.rng,
            device=device,
        )
        y = physics(x, **params)

        assert y.shape[-1] == x.shape[-1] // params["factor"].unique().item()

        if fact is not None and not isinstance(fact, list):
            assert fact == params["factor"].unique().item()


######################
### MRI GENERATORS ###
######################


@pytest.fixture
def batch_size():
    return 2


def choose_mri_generator(generator_name, img_size, acc, center_fraction, device, rng):
    if generator_name == "gaussian":
        g = GaussianMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            rng=rng,
            device=device,
        )
    elif generator_name == "random":
        g = RandomMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            rng=rng,
            device=device,
        )
    elif generator_name == "uniform":
        g = EquispacedMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            rng=rng,
            device=device,
        )
    elif generator_name == "poly":
        g = PolyOrderMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            poly_order=2,
            rng=rng,
            device=device,
        )
    return g


@pytest.mark.parametrize("generator_name", MRI_GENERATORS)
@pytest.mark.parametrize("img_size", MRI_IMG_SIZES)
@pytest.mark.parametrize("acc", MRI_ACCELERATIONS)
@pytest.mark.parametrize("center_fraction", MRI_CENTER_FRACTIONS)
def test_mri_generator(
    generator_name, img_size, batch_size, acc, center_fraction, device, rng
):
    generator = choose_mri_generator(
        generator_name, img_size, acc, center_fraction, device, rng
    )
    # test across different accs and center fractions
    H, W = img_size[-2:]
    assert W // generator.acc == (generator.n_lines + generator.n_center)

    mask = generator.step(batch_size=batch_size, seed=0)["mask"]

    if len(img_size) == 2:
        assert len(mask.shape) == 4
        C = 1
    elif len(img_size) == 3:
        assert len(mask.shape) == 4
        C = img_size[0]
    elif len(img_size) == 4:
        assert len(mask.shape) == 5
        C = img_size[0]
        assert mask.shape[2] == img_size[1]

    assert mask.shape[0] == batch_size
    assert mask.shape[1] == C
    assert mask.shape[-2:] == img_size[-2:]

    for b in range(batch_size):
        for c in range(C):
            if len(img_size) == 4:
                for t in range(img_size[1]):
                    mask[b, c, t, :, :].sum() * generator.acc == H * W
            else:
                mask[b, c, :, :].sum() * generator.acc == H * W

    mask2 = generator.step(batch_size=batch_size)["mask"]

    if generator.n_lines != 0 and generator_name != "uniform":
        assert not torch.allclose(mask, mask2)


#############################
### INPAINTING GENERATORS ###
#############################


def choose_inpainting_generator(name, img_size, split_ratio, pixelwise, device, rng):
    if name == "bernoulli":
        return dinv.physics.generator.BernoulliSplittingMaskGenerator(
            img_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=rng,
        )
    elif name == "gaussian":
        return dinv.physics.generator.GaussianSplittingMaskGenerator(
            img_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=rng,
        )
    elif name == "multiplicative":
        mri_gen = dinv.physics.generator.GaussianMaskGenerator(
            img_size=img_size,
            acceleration=2,
            device=device,
            rng=rng,
        )
        return dinv.physics.generator.MultiplicativeSplittingMaskGenerator(
            img_size=img_size,
            split_generator=mri_gen,
            device=device,
        )
    else:
        raise Exception("The generator chosen doesn't exist")


@pytest.mark.parametrize("generator_name", INPAINTING_GENERATORS)
@pytest.mark.parametrize("img_size", INPAINTING_IMG_SIZES)
@pytest.mark.parametrize("pixelwise", (False, True))
@pytest.mark.parametrize("split_ratio", (0.5,))
def test_inpainting_generators(
    generator_name, batch_size, img_size, pixelwise, split_ratio, device, rng
):
    if generator_name in ("gaussian", "multiplicative") and len(img_size) < 3:
        pytest.skip(
            "Gaussian and multiplicative splitting mask not valid for images of shape smaller than (C, H, W)"
        )

    if generator_name == "multiplicative" and not pixelwise:
        pytest.skip("Multiplicative mask test not defined for non pixelwise masking.")

    gen = choose_inpainting_generator(
        generator_name, img_size, split_ratio, pixelwise, device, rng
    )  # Assume generator always receives "correct" img_size i.e. not one with dims missing

    def correct_ratio(ratio, rtol=1e-2, atol=1e-2):
        assert torch.isclose(
            ratio,
            torch.tensor([split_ratio], device=device),
            rtol=rtol,
            atol=atol,
        )

    def correct_pixelwise(mask):
        if pixelwise:
            assert torch.all(mask[:, 0, ...] == mask[:, 1, ...])
        else:
            assert not torch.all(mask[:, 0, ...] == mask[:, 1, ...])

    # Standard generate mask
    mask1 = gen.step(batch_size=batch_size, seed=0)["mask"]
    correct_ratio(mask1.sum() / np.prod((batch_size, *img_size)))
    correct_pixelwise(mask1)

    # Standard without batch dim
    mask1 = gen.step(batch_size=None, seed=0)["mask"]
    assert tuple(mask1.shape) == tuple(img_size)
    correct_ratio(mask1.sum() / np.prod(img_size))

    # Standard mask but by passing flat input_mask of ones
    input_mask = torch.ones(batch_size, *img_size)
    # should ignore batch_size
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
    correct_ratio(mask2.sum() / input_mask.sum())
    correct_pixelwise(mask2)

    # As above but with no batch dimension in input_mask
    input_mask = torch.ones(*img_size, device=device)
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)[
        "mask"
    ]  # should use batch_size
    correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # As above but with img_size missing channel dimension (bad practice)
    # Note: Multiplicative mask must have correct input mask shape
    # Note: 1D input_mask not compatible with pixelwise
    if generator_name != "multiplicative" and not (len(img_size) <= 2 and pixelwise):
        input_mask = torch.ones(*img_size[1:], device=device)
        mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
        correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # Generate splitting mask from already subsampled mask
    # Multiplicative splitting will rarely be exact
    input_mask = torch.zeros(batch_size, *img_size, device=device)
    input_mask[..., 10:20] = 1
    mask3 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
    correct_ratio(
        mask3.sum() / input_mask.sum(),
        atol=1e-2 if generator_name != "multiplicative" else 2.5e-1,
    )
    correct_pixelwise(mask3)

    # Adapt to new img sizes
    assert gen.step(batch_size=batch_size, img_size=(73, 29))["mask"].shape[-2:] == (
        73,
        29,
    )

    # Raise error if input_mask and img_size both passed
    with pytest.raises(ValueError):
        gen.step(img_size=(20, 20), input_mask=(2, 20, 20))


def choose_crosshatch_generator(
    generator_name, img_size, device, rng, dtype=torch.float32, text="DEEPINV", **kwargs
):
    """Build the crosshatched text generator variant named ``generator_name``.

    Variant defaults are applied with ``setdefault``, so any of them can be overridden by the
    caller, e.g. ``choose_crosshatch_generator("layered", ..., angles=(0.0, 90.0))``.
    """
    kwargs.update(img_size=img_size, device=device, dtype=dtype, rng=rng)
    if generator_name == "layered":
        kwargs.setdefault("mode", "layered")
        kwargs.setdefault("angles", (0.0, 45.0, 90.0))
        return dinv.physics.generator.CrosshatchTextMaskGenerator(text=text, **kwargs)
    elif generator_name == "hatch":
        kwargs.setdefault("mode", "hatch")
        kwargs.setdefault("angles", (45.0, 135.0))
        return dinv.physics.generator.CrosshatchTextMaskGenerator(text=text, **kwargs)
    elif generator_name == "random_shift":
        kwargs.setdefault("random_shift", True)
        return dinv.physics.generator.CrosshatchTextMaskGenerator(text=text, **kwargs)
    elif generator_name == "random_angles":
        kwargs.setdefault("random_angles", True)
        return dinv.physics.generator.CrosshatchTextMaskGenerator(text=text, **kwargs)
    elif generator_name == "multitext":
        # Same text on both layers, so the generic text assertions still hold; the
        # per-layer texts are exercised by the dedicated tests below.
        kwargs.setdefault("angles", (0.0, 90.0))
        return dinv.physics.generator.MultiTextCrosshatchMaskGenerator(
            texts=(text, text), **kwargs
        )
    else:
        raise Exception("The generator chosen doesn't exist")


@pytest.mark.parametrize("generator_name", CROSSHATCH_GENERATORS)
@pytest.mark.parametrize("img_size", CROSSHATCH_IMG_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_crosshatch_generators(
    generator_name, img_size, dtype, batch_size, device, rng
):
    gen = choose_crosshatch_generator(
        generator_name, img_size, device, rng, dtype=dtype
    )  # Assume generator always receives "correct" img_size i.e. not one with dims missing

    def correct_mask(mask, shape):
        assert tuple(mask.shape) == tuple(shape)
        assert mask.dtype == dtype
        assert mask.device.type == device.type
        # Mask is binary
        assert torch.all((mask == 0) | (mask == 1))
        # Text is rendered: it removes some pixels, but neither none nor all of them
        assert 0 < (mask == 0).sum() < mask.numel()

    # Standard generate mask
    mask1 = gen.step(batch_size=batch_size, seed=0)["mask"]
    correct_mask(mask1, (batch_size, *img_size))

    # Every channel of a sample carries the same text
    if len(img_size) == 3:
        assert torch.all(mask1 == mask1[:, :1])

    # Same seed, same masks
    mask2 = gen.step(batch_size=batch_size, seed=0)["mask"]
    assert torch.equal(mask1, mask2)

    # Deterministic variants repeat the same mask across the batch
    if generator_name not in ("random_shift", "random_angles"):
        assert torch.all(mask1 == mask1[:1])

    # Standard without batch dim
    correct_mask(gen.step(batch_size=None, seed=0)["mask"], img_size)

    # Adapt to new img sizes
    correct_mask(
        gen.step(batch_size=batch_size, img_size=(37, 29))["mask"],
        (batch_size, *img_size[:-2], 37, 29),
    )


@pytest.mark.parametrize("generator_name", ("random_shift", "random_angles"))
def test_crosshatch_batch_randomness(generator_name, device, rng):
    """``random_shift`` and ``random_angles`` decorrelate the samples of a batch."""
    # A batch of 4, so that a coincidental repeat cannot pass for correct behaviour
    gen = choose_crosshatch_generator(generator_name, (1, 64, 64), device, rng)
    mask = gen.step(batch_size=4, seed=0)["mask"]

    assert not torch.all(mask == mask[:1])
    # Same seed, same batch
    assert torch.equal(mask, gen.step(batch_size=4, seed=0)["mask"])


@pytest.mark.parametrize("generator_name", CROSSHATCH_GENERATORS)
def test_crosshatch_text_is_rendered(generator_name, device, rng):
    """The rendered glyphs, and only them, drive the mask."""

    def mask_of(text):
        gen = choose_crosshatch_generator(
            generator_name, (1, 64, 64), device, rng, text=text
        )
        return gen.step(batch_size=1, seed=0)["mask"]

    # The same text always gives the same mask
    assert torch.equal(mask_of("DEEPINV"), mask_of("DEEPINV"))

    # A different text gives a different mask
    assert not torch.equal(mask_of("DEEPINV"), mask_of("IIII"))

    # A blank text removes nothing
    assert torch.all(mask_of(" ") == 1)


@pytest.mark.parametrize("generator_name", CROSSHATCH_GENERATORS)
def test_crosshatch_invert(generator_name, device, rng):
    """``invert`` swaps the glyphs and the background."""
    img_size = (1, 64, 64)
    removed = choose_crosshatch_generator(generator_name, img_size, device, rng).step(
        batch_size=1, seed=0
    )["mask"]
    kept = choose_crosshatch_generator(
        generator_name, img_size, device, rng, invert=True
    ).step(batch_size=1, seed=0)["mask"]

    # By default the text is the missing region, so the mask is 0 on the glyphs
    assert torch.equal(kept, 1.0 - removed)
    assert 0 < kept.sum() < kept.numel()


def test_crosshatch_hatch_confined_to_glyphs(device, rng):
    """In ``hatch`` mode the gratings never leave the glyphs of the unrotated text."""
    img_size = (1, 64, 64)
    text = dinv.physics.generator.CrosshatchTextMaskGenerator(
        img_size, angles=(0.0,), invert=True, device=device, rng=rng
    ).step(batch_size=1, seed=0)["mask"]
    hatch = choose_crosshatch_generator(
        "hatch", img_size, device, rng, invert=True
    ).step(batch_size=1, seed=0)["mask"]

    assert torch.all(hatch <= text)
    # The gratings carve the glyphs up rather than leaving them untouched
    assert 0 < hatch.sum() < text.sum()


@pytest.mark.parametrize("font_size", (None, 20))
def test_crosshatch_font_size(font_size, device, rng):
    """Rasterization stays binary and non-empty for bitmap and TrueType fonts."""
    gen = choose_crosshatch_generator(
        "layered", (1, 96, 96), device, rng, font_size=font_size
    )
    mask = gen.step(batch_size=1, seed=0)["mask"]

    assert torch.all((mask == 0) | (mask == 1))
    assert 0 < (mask == 0).sum() < mask.numel()


@pytest.mark.parametrize("angle", (0.0, 30.0, 45.0, 90.0, 180.0))
def test_crosshatch_rotation_matrix(angle, device):
    R = dinv.physics.generator.CrosshatchTextMaskGenerator.rotation_matrix(
        angle, device=device
    )
    eye = torch.eye(2, device=device)

    assert tuple(R.shape) == (2, 2)
    # Rotations are orthogonal with unit determinant
    assert torch.allclose(R @ R.T, eye, atol=1e-6)
    assert torch.allclose(R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0], eye[0, 0], atol=1e-6)

    # Rotating back by the opposite angle is the identity
    R_inv = dinv.physics.generator.CrosshatchTextMaskGenerator.rotation_matrix(
        -angle, device=device
    )
    assert torch.allclose(R @ R_inv, eye, atol=1e-6)


def test_crosshatch_generator_errors(device, rng):
    def build(**kwargs):
        kwargs.setdefault("img_size", (1, 64, 64))
        return dinv.physics.generator.CrosshatchTextMaskGenerator(
            device=device, rng=rng, **kwargs
        )

    # Unknown mode
    with pytest.raises(ValueError):
        build(mode="hatched")

    # img_size must be of shape (C, H, W) or (H, W)
    with pytest.raises(ValueError):
        build(img_size=(2, 1, 64, 64))

    # At least one angle is needed
    with pytest.raises(ValueError):
        build(angles=())

    # Unknown generator name
    with pytest.raises(Exception):
        choose_crosshatch_generator("bernoulli", (1, 64, 64), device, rng)


@pytest.mark.parametrize("generator_name", CROSSHATCH_GENERATORS)
def test_crosshatch_as_occlusion_mask(generator_name, batch_size, device, rng):
    """Use 1: step() is an occlusion mask, so it drops straight into Inpainting."""
    img_size = (1, 32, 32)
    gen = choose_crosshatch_generator(generator_name, img_size, device, rng)
    physics = dinv.physics.Inpainting(img_size, device=device)

    x = torch.randn(batch_size, *img_size, device=device)
    params = gen.step(batch_size=batch_size, seed=0)
    y = physics(x, **params)

    assert y.shape == x.shape
    assert torch.equal(y, x * params["mask"])


@pytest.mark.parametrize("generator_name", CROSSHATCH_GENERATORS)
@pytest.mark.parametrize("img_size", CROSSHATCH_IMG_SIZES)
def test_crosshatch_layer_fields(generator_name, img_size, device, rng):
    """Use 2: layer_fields() exposes the text sources that step() collapses together."""
    gen = choose_crosshatch_generator(generator_name, img_size, device, rng)

    gen.rng_manual_seed(0)
    layers = gen.layer_fields()

    # One layer per angle, spatial only -- no channel or batch dimension
    assert layers.shape == (len(gen.angles), *img_size[-2:])
    assert torch.all((layers == 0) | (layers == 1))

    # Every layer carries text; none is empty or completely filled
    for k in range(layers.shape[0]):
        assert 0 < layers[k].sum() < layers[k].numel()


@pytest.mark.parametrize("generator_name", CROSSHATCH_GENERATORS)
def test_crosshatch_mask_is_the_union_of_the_layers(generator_name, device, rng):
    """The two uses agree: the mask is exactly the complement of the layers' union."""
    img_size = (1, 64, 64)
    gen = choose_crosshatch_generator(generator_name, img_size, device, rng)

    gen.rng_manual_seed(0)
    layers = gen.layer_fields()

    gen.rng_manual_seed(0)
    mask = gen.batch_step()

    assert torch.equal(mask, (1.0 - layers.amax(dim=0)).expand(*img_size))


def test_crosshatch_layer_fields_adapts_to_img_size(device, rng):
    gen = choose_crosshatch_generator("layered", (1, 64, 64), device, rng)

    assert gen.layer_fields(img_size=(37, 29)).shape == (len(gen.angles), 37, 29)


def test_crosshatch_layers_differ_between_angles(device, rng):
    """Distinct angles must give distinct layers, otherwise there is nothing to separate."""
    gen = choose_crosshatch_generator(
        "layered", (1, 64, 64), device, rng, angles=(0.0, 90.0)
    )
    gen.rng_manual_seed(0)
    layers = gen.layer_fields()

    assert not torch.equal(layers[0], layers[1])


@pytest.mark.parametrize("img_size", [(1, 64, 64), (1, 64, 40)])
def test_crosshatch_layer_fields_canvas(img_size, device, rng):
    """canvas=True returns the layers on the domain the separation operators expect."""
    gen = choose_crosshatch_generator(
        "layered", img_size, device, rng, angles=(0.0, 90.0)
    )
    side = dinv.physics.canvas_size(img_size)

    cropped = gen.layer_fields()
    full = gen.layer_fields(canvas=True)

    assert cropped.shape == (2, img_size[-2], img_size[-1])
    assert full.shape == (2, side, side)
    # The default crop is exactly the centre of the canvas
    assert torch.allclose(dinv.physics.center_crop(full, img_size), cropped)


def test_multitext_layer_fields_canvas(device, rng):
    """The per-text subclass honours canvas=True the same way."""
    gen = multitext_generator(
        device, rng, texts=("DEEPINVERSE", "TEST"), angles=(0.0, 90.0)
    )
    side = dinv.physics.canvas_size((1, 64, 64))

    full = gen.layer_fields(canvas=True)

    assert full.shape == (2, side, side)
    assert torch.allclose(
        dinv.physics.center_crop(full, (1, 64, 64)), gen.layer_fields()
    )


def test_crosshatch_tile_cache_is_bounded(device, rng):
    """
    Per-call ``text`` means a caller controls the cache keys, so it must not grow without bound.

    Rendering is cached because it is the expensive part of a step, but a loop over unique
    strings would otherwise retain every tile for the life of the generator.
    """
    gen = choose_crosshatch_generator("layered", (1, 32, 32), device, rng)
    cap = gen.max_tile_cache

    for i in range(cap * 4):
        gen.step(batch_size=1, text=f"UNIQUE-{i}")

    assert len(gen._tile_cache) <= cap
    # The oldest overrides were evicted rather than retained
    assert "UNIQUE-0" not in gen._tile_cache
    assert f"UNIQUE-{cap * 4 - 1}" in gen._tile_cache


def test_crosshatch_tile_cache_still_caches(device, rng):
    """Eviction must not break the caching that makes repeated steps cheap."""
    gen = choose_crosshatch_generator("layered", (1, 32, 32), device, rng, text="KEEP")

    first = gen.step(batch_size=1)["mask"]
    assert "KEEP" in gen._tile_cache
    assert torch.equal(first, gen.step(batch_size=1)["mask"])

    # A recently used entry survives eviction pressure from later unique strings
    for i in range(gen.max_tile_cache - 1):
        gen.step(batch_size=1, text=f"OTHER-{i}")
    assert "KEEP" in gen._tile_cache


def test_multitext_tile_cache_is_bounded(device, rng):
    """The per-layer texts path shares the same cache and the same bound."""
    gen = multitext_generator(device, rng, texts=("A", "B"), angles=(0.0, 90.0))

    for i in range(gen.max_tile_cache * 3):
        gen.layer_fields(texts=(f"U{i}", f"V{i}"))

    assert len(gen._tile_cache) <= gen.max_tile_cache


def test_crosshatch_text_override_per_call(device, rng):
    """The string is a per-call argument, not something baked into the generator."""
    gen = choose_crosshatch_generator("layered", (1, 64, 64), device, rng, text="AAA")

    default = gen.step(batch_size=1)["mask"]
    other = gen.step(batch_size=1, text="ZZZZZZ")
    again = gen.step(batch_size=1, text="AAA")["mask"]

    assert not torch.equal(default, other["mask"])
    # The override is for one call only, it does not mutate the generator
    assert gen.text == "AAA"
    assert torch.equal(default, again)


def test_crosshatch_angles_override_per_call(device, rng):
    """Angles are a per-call argument too, and may differ in number from the stored ones."""
    gen = choose_crosshatch_generator(
        "layered", (1, 64, 64), device, rng, angles=(0.0, 90.0)
    )

    default = gen.step(batch_size=1)["mask"]
    turned = gen.step(batch_size=1, angles=(30.0, 120.0))["mask"]
    three = gen.layer_fields(angles=(0.0, 60.0, 120.0))

    assert not torch.equal(default, turned)
    assert gen.angles == (0.0, 90.0)
    assert three.shape[0] == 3


def test_crosshatch_explicit_angles_beat_random(device, rng):
    """random_angles must not override an angle the caller asked for explicitly."""
    gen = choose_crosshatch_generator(
        "layered", (1, 64, 64), device, rng, angles=(0.0, 90.0), random_angles=True
    )

    a = gen.layer_fields(angles=(15.0, 75.0))
    b = gen.layer_fields(angles=(15.0, 75.0))

    assert torch.equal(a, b)  # deterministic despite random_angles


@pytest.mark.parametrize("angle_range", [(0.0, 180.0), (85.0, 95.0), (0.0, 1.0)])
def test_crosshatch_angle_range(angle_range, device, rng):
    """The random draw is bounded by angle_range, not hardcoded to [0, 180)."""
    gen = choose_crosshatch_generator(
        "layered",
        (1, 32, 32),
        device,
        rng,
        angles=(0.0, 90.0),
        random_angles=True,
        angle_range=angle_range,
    )

    drawn = [a for _ in range(20) for a in gen.resolve_angles()]

    assert all(angle_range[0] <= a <= angle_range[1] for a in drawn)


def test_crosshatch_rejects_bad_angle_range(device, rng):
    with pytest.raises(ValueError, match="increasing"):
        choose_crosshatch_generator(
            "layered", (1, 32, 32), device, rng, angle_range=(90.0, 10.0)
        )


def test_multitext_texts_override_per_call(device, rng):
    """texts, and a single text applied to every layer, are both per-call arguments."""
    gen = multitext_generator(device, rng, texts=("AAA", "BBB"), angles=(0.0, 90.0))

    default = gen.layer_fields()
    swapped = gen.layer_fields(texts=("ZZZ", "YYY"))
    single = gen.layer_fields(text="QQQ")

    assert not torch.equal(default, swapped)
    assert gen.texts == ("AAA", "BBB")
    # A single text is the one-string form: every layer carries it
    assert torch.equal(single, gen.layer_fields(texts=("QQQ",)))


def test_multitext_texts_for_angles_cycles(device, rng):
    """Any number of strings spreads over any number of layers."""
    gen = multitext_generator(device, rng, texts=("A", "B"), angles=(0.0, 90.0))

    assert gen.texts_for_angles(angles=(0.0, 45.0, 90.0, 135.0)) == ("A", "B", "A", "B")
    assert gen.texts_for_angles(angles=(0.0, 45.0), texts=("Z",)) == ("Z", "Z")
    with pytest.raises(ValueError, match="At least one text"):
        gen.texts_for_angles(texts=())


def test_crosshatch_as_overlay_source(device, rng):
    """Use 2, end to end: upright layers compose additively through CrosshatchTextOverlay."""
    img_size = (1, 64, 64)
    angles, amplitudes = (0.0, 90.0), (0.5, 0.3)

    physics = dinv.physics.CrosshatchTextOverlay(
        img_size, angles=angles, amplitudes=amplitudes, device=device
    )
    side = physics.canvas

    # angles=(0,) with canvas=True gives the upright text on the domain the overlay
    # parametrizes a source on, so the generator and the operator agree exactly.
    upright = [
        choose_crosshatch_generator(
            "layered", img_size, device, rng, text=text, angles=(0.0,)
        ).layer_fields(canvas=True)[0]
        for text in ("DEEPINVERSE", "TEST")
    ]
    assert all(u.shape == (side, side) for u in upright)

    background = torch.rand(1, img_size[0], side, side, device=device) * 0.5
    x = torch.stack(
        [background] + [u.expand(1, img_size[0], side, side) for u in upright], dim=1
    )
    y = physics.A(x)

    # Text is added on top, so it can only brighten, and it must change something
    window = dinv.physics.center_crop(background, img_size)
    assert (y >= window - 1e-6).all()
    assert not torch.allclose(y, window)


def multitext_generator(device, rng, **kwargs):
    """Build a MultiTextCrosshatchMaskGenerator with defaults shared by the tests below."""
    kwargs.setdefault("img_size", (1, 64, 64))
    return dinv.physics.generator.MultiTextCrosshatchMaskGenerator(
        device=device, rng=rng, **kwargs
    )


@pytest.mark.parametrize(
    "texts, angles, expected",
    [
        (("DEEPINVERSE", "TEST"), (0.0, 90.0), ("DEEPINVERSE", "TEST")),
        (("AB", "C"), (0.0, 45.0, 90.0, 135.0), ("AB", "C", "AB", "C")),
        (("ONLY",), (0.0, 45.0), ("ONLY", "ONLY")),
    ],
)
def test_multitext_pairing(texts, angles, expected, device, rng):
    """``texts`` is paired with ``angles`` elementwise and cycled when shorter."""
    gen = multitext_generator(device, rng, texts=texts, angles=angles)

    assert gen.texts_per_angle == expected
    assert len(gen.texts_per_angle) == len(gen.angles)


def test_multitext_matches_parent_for_one_text(device, rng):
    """With a single string the subclass reproduces the parent generator exactly."""
    img_size = (1, 64, 64)
    kwargs = dict(angles=(0.0, 45.0, 90.0))

    parent = dinv.physics.generator.CrosshatchTextMaskGenerator(
        img_size, text="DEEPINV", device=device, rng=rng, **kwargs
    ).step(batch_size=1, seed=0)["mask"]
    child = multitext_generator(
        device, rng, img_size=img_size, texts=("DEEPINV",), **kwargs
    ).step(batch_size=1, seed=0)["mask"]

    assert torch.equal(parent, child)


@pytest.mark.parametrize("mode", ("layered", "hatch"))
def test_multitext_layers_differ_by_text(mode, device, rng):
    """Changing only the second string changes the mask, so each layer uses its own text."""
    kwargs = dict(angles=(0.0, 90.0), mode=mode)

    a = multitext_generator(device, rng, texts=("DEEPINVERSE", "TEST"), **kwargs).step(
        batch_size=1, seed=0
    )["mask"]
    b = multitext_generator(device, rng, texts=("DEEPINVERSE", "OTHER"), **kwargs).step(
        batch_size=1, seed=0
    )["mask"]

    assert not torch.equal(a, b)
    assert torch.all((a == 0) | (a == 1))
    assert 0 < (a == 0).sum() < a.numel()


def test_multitext_text_lengths(device, rng):
    """Strings of very different lengths tile independently and both leave their mark."""
    # A blank second layer must remove strictly less than a rendered one
    blank = multitext_generator(
        device, rng, texts=("DEEPINVERSE", " "), angles=(0.0, 90.0)
    ).step(batch_size=1, seed=0)["mask"]
    short = multitext_generator(
        device, rng, texts=("DEEPINVERSE", "I"), angles=(0.0, 90.0)
    ).step(batch_size=1, seed=0)["mask"]
    long = multitext_generator(
        device, rng, texts=("DEEPINVERSE", "TESTING123"), angles=(0.0, 90.0)
    ).step(batch_size=1, seed=0)["mask"]

    # More text in the second layer removes more pixels
    assert (blank == 0).sum() < (short == 0).sum() < (long == 0).sum()


def test_multitext_generator_errors(device, rng):
    # At least one text is needed
    with pytest.raises(ValueError):
        multitext_generator(device, rng, texts=())

    # Parent validation still applies
    with pytest.raises(ValueError):
        multitext_generator(device, rng, mode="hatched")

    with pytest.raises(ValueError):
        multitext_generator(device, rng, angles=())


@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_inpainting_generator_random_ratio(num_channels, device, dtype, rng):
    # NOTE elements of this test are now redundant given above tests
    size = (100, 100)  # we take it large to have significant statistical numbers after
    physics = dinv.physics.Inpainting(
        (num_channels, size[0], size[1]), 0.9, device=device
    )

    split_ratio = 0.6
    generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        (num_channels, size[0], size[1]),
        split_ratio=split_ratio,
        device=device,
        dtype=dtype,
        rng=rng,
    )
    batch_size = 2
    params = generator.step(batch_size=batch_size)

    mask = params["mask"]
    assert mask.shape == (batch_size, num_channels, size[0], size[1])

    experimental_split_ratio = (mask[0] == 1).sum() / mask[0].numel()
    assert abs(experimental_split_ratio.item() - split_ratio) < 1e-2

    # check forward
    x = torch.randn(
        (batch_size, num_channels, size[0], size[1]),
        generator=generator.rng,
        device=device,
    )
    y = physics(x, **params)
    experimental_split_ratio_obs = 1 - (y[0] == 0).sum() / y[0].numel()
    assert torch.allclose(
        experimental_split_ratio, experimental_split_ratio_obs, rtol=1e-4
    )

    # now we do the same with each element in the batch for random_split_ratio
    min_split_ratio = 0.001
    max_split_ratio = 0.5
    generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        (num_channels, size[0], size[1]),
        split_ratio=split_ratio,
        random_split_ratio=True,
        min_split_ratio=min_split_ratio,
        max_split_ratio=max_split_ratio,
        device=device,
        rng=rng,
    )
    batch_size = 2
    params = generator.step(batch_size=batch_size, seed=0)

    mask = params["mask"]
    assert mask.shape == (batch_size, num_channels, size[0], size[1])

    x = torch.randn(
        (batch_size, num_channels, size[0], size[1]),
        generator=generator.rng,
        device=device,
    )
    y = physics(x, **params)

    list_exp_split_ratio = []
    for b in range(batch_size):
        experimental_split_ratio = (mask[b] == 1).sum() / mask[b].numel()
        assert experimental_split_ratio.item() < max_split_ratio + 1e-2
        assert experimental_split_ratio.item() > min_split_ratio - 1e-2

        # check forward
        experimental_split_ratio_obs = 1 - (y[b] == 0).sum() / y[b].numel()
        assert torch.allclose(
            experimental_split_ratio, experimental_split_ratio_obs, rtol=1e-3
        )

        list_exp_split_ratio.append(experimental_split_ratio)

    # check that split ratios are different between batches
    assert abs(list_exp_split_ratio[0] - list_exp_split_ratio[1]) > 1e-2


def test_string_seed():
    # Dummy long paths
    paths = [f"{'deepinv/' * 10}{p}" for p in Path("deepinv/tests").glob("*.py")]
    seeds = [seed_from_string(p) for p in paths]

    # Assert unique seeds
    assert len(set(seeds)) == len(seeds)

    # Assert seed in correct range for manual_seed
    for s in seeds:
        assert -0x8000_0000_0000_0000 < s < 0xFFFF_FFFF_FFFF_FFFF

    # Assert generators different
    states = [torch.Generator().manual_seed(s).get_state() for s in seeds]
    assert len(set(states)) == len(states)


@pytest.mark.parametrize("apodize", [True, False])
@pytest.mark.parametrize("random_rotate", [True, False])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("convention", ["noll", "ansi"])
@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize(
    "fc", [None, 0.2, (0.15, 0.2), torch.tensor([[0.10, 0.11], [0.2, 0.21]])]
)
@pytest.mark.parametrize("coeff", [None, torch.zeros(2, 35)])
def test_diffraction_generator(
    device,
    apodize,
    random_rotate,
    center,
    convention,
    is_3d,
    rng,
    fc,
    coeff,
):
    r"""
    Test diffraction generator.
    """

    dtype = torch.float32
    zernike_index = tuple(range(1, 36))  # All Zernike index up to 7th order
    pupil_size = (256, 256)
    if is_3d:
        size = (5, 5, 5)
        generator = dinv.physics.generator.DiffractionBlurGenerator3D(
            psf_size=size,
            device=device,
            zernike_index=zernike_index,
            index_convention=convention,
            apodize=apodize,
            random_rotate=random_rotate,
            dtype=dtype,
            pupil_size=pupil_size,
            rng=rng,
        )

    else:
        pupil_size = (256, 256)
        size = (5, 5)
        generator = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            zernike_index=zernike_index,
            index_convention=convention,
            apodize=apodize,
            random_rotate=random_rotate,
            center=center,
            dtype=dtype,
            pupil_size=pupil_size,
            rng=rng,
        )

    batch_sizes = (1, 2)
    expected_keys = set(
        ["filter", "coeff", "pupil", "fc"]
        + (["angle"] if random_rotate else [])
        + (["coeff_tilt_x", "coeff_tilt_y"] if ((not is_3d) and center) else [])
    )
    for batch_size in batch_sizes:
        params = generator.step(
            batch_size=batch_size,
            seed=0,
            focal_length=0.004,
            aperture_diameter=0.002,
            apodize=apodize,
            random_rotate=random_rotate,
            fc=fc,
            coeff=coeff.to(device) if coeff is not None else None,
        )

        if fc is not None:
            if isinstance(fc, float):
                num_channels_out = 1
                batch_size_out = batch_size
            else:
                fc_tensor = torch.as_tensor(fc)
                if fc_tensor.ndim == 1:
                    fc_tensor = fc_tensor[None, :].expand(batch_size, -1)
                batch_size_out, num_channels_out = fc_tensor.shape
        else:
            batch_size_out = batch_size
            num_channels_out = 1

        if coeff is not None:
            if coeff.ndim == 2:
                batch_size_out = coeff.shape[0]
            elif coeff.ndim == 3:
                batch_size_out, num_channels_out = coeff.shape[:2]

        # print(fc, batch_size_out, num_channels_out)
        # print(params["filter"].shape, (batch_size_out, num_channels_out, *size))

        # Test keys and shapes
        assert set(params.keys()) == expected_keys
        assert params["filter"].shape == (batch_size_out, num_channels_out, *size)
        assert params["coeff"].shape == (
            batch_size_out,
            num_channels_out,
            len(zernike_index),
        )
        assert params["pupil"].shape == (batch_size_out, num_channels_out, *pupil_size)
        if random_rotate:
            assert params["angle"].shape == (batch_size_out,)
        if (not is_3d) and center:
            assert params["coeff_tilt_x"].shape == (batch_size_out, num_channels_out, 1)
            assert params["coeff_tilt_y"].shape == (batch_size_out, num_channels_out, 1)

        # Test generator consistency when coeff is None
        params2 = generator.step(
            batch_size=batch_size,
            seed=0,
            fc=fc,
        )
        if coeff is None:
            for key in params.keys():
                assert torch.allclose(params[key], params2[key])

        # Test generator variability when coeff is None
        params3 = generator.step(
            batch_size=batch_size,
            seed=1,
            fc=fc,
        )
        if coeff is None:
            for key in params.keys():
                if key == "fc":
                    assert torch.allclose(params[key], params3[key])
                else:
                    assert not torch.allclose(params[key], params3[key])

        # test raising ValueError when incompatible shapes
        if (
            fc is None
            and coeff is None
            and not apodize
            and not random_rotate
            and convention == "noll"
        ):
            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=2,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(3, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=3, K)  (B_f != B_c)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=2,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(3, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=2, K)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=5,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(1, 35).to(device),
                )  # (B_f=2, C_f=2) vs (1, K)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=5,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(3, 2, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=3, C_c=2, K)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=5,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(2, 3, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=2, C_c=3, K)
            assert "does not match" in str(excinfo.value)

    # Test centering effect if center is True and psf size is large enough
    # Test only for 2D case as centering in 3D case is still under development

    # Helper function to compute the barycenter of a PSF
    def _barycenter(h):
        Ny, Nx = h.shape[-2:]
        centerx = (Nx / 2.0) - 0.5
        centery = (Ny / 2.0) - 0.5
        x = torch.arange(0, h.shape[-1]).to(h.device)
        y = torch.arange(0, h.shape[-2]).to(h.device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        com_x = (X[None, None, :, :] * h).sum(dim=(-2, -1)) - centerx
        com_y = (Y[None, None, :, :] * h).sum(dim=(-2, -1)) - centery
        return com_x, com_y

    if (not is_3d) and center and (not apodize) and (not random_rotate):
        batch_sizes = (1, 2)
        size = (71, 71)
        pupil_size = (256, 256)
        generator = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            zernike_index=zernike_index,
            index_convention=convention,
            apodize=apodize,
            random_rotate=random_rotate,
            center=center,
            dtype=dtype,
            pupil_size=pupil_size,
            rng=rng,
        )
        for batch_size in batch_sizes:
            generated_psf = generator.step(
                batch_size=batch_size,
                seed=0,
                focal_length=0.004,
                aperture_diameter=0.002,
                apodize=apodize,
                random_rotate=random_rotate,
            )["filter"]
            com_x, com_y = _barycenter(generated_psf)
            assert torch.all(torch.round(com_x.abs(), decimals=1) <= 0.1)
            assert torch.all(torch.round(com_y.abs(), decimals=1) <= 0.1)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("isotropic", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gaussian_blur_generator(device, dim, isotropic, batch_size):
    r"""
    Validate GaussianBlurGenerator behaviors across 1D/2D/3D:
    - isotropic vs anisotropic sigma handling
    - float or tuple for sigma_min/max
    - float or tuple for angle_min/max
    """
    torch.manual_seed(0)

    # choose psf size according to dimension
    if dim == 1:
        psf_size = (7,)
    elif dim == 2:
        psf_size = (7, 7)
    else:
        psf_size = (5, 5, 5)

    if dim == 1:
        if isotropic:
            pytest.skip("Isotropic setting not relevant for 1D Gaussian blur.")
        # In 1D, isotropic should be ignored and sigma_min/max accept single float/integer or length-1 tuple
        generator = dinv.physics.generator.GaussianBlurGenerator(
            psf_size=psf_size,
            sigma_min=0.5,
            sigma_max=1,
            device=device,
        )
        params = generator.step(batch_size=batch_size, seed=0)
        assert params["filter"].shape == (batch_size, 1, *psf_size)

        # providing length-1 tuple should also work
        generator = dinv.physics.generator.GaussianBlurGenerator(
            psf_size=psf_size,
            sigma_min=(0.5,),
            sigma_max=(1,),
            device=device,
        )
        params = generator.step(batch_size=batch_size, seed=0)
        assert params["filter"].shape == (batch_size, 1, *psf_size)

        # providing length-2 tuple should raise error
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=True,
                sigma_min=(0.5, 1.1),
                sigma_max=3.0,
                device=device,
            )

    elif dim == 2:
        # In 2D, generator can accept float, integer, length-1 or length-2 tuple for sigma_min/max. If different than length-2 tuple, the same min/max will be applied to both dimensions.

        for sigma_min, sigma_max in zip(
            [0.5, (0.5,), (0.5, 0.6)], [1.0, (1.0,), (1.0, 1.1)], strict=True
        ):
            generator = dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=isotropic,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                angle_min=0.0,
                angle_max=(torch.pi,),
                device=device,
            )
            params = generator.step(batch_size=batch_size, seed=0)
            assert params["filter"].shape == (batch_size, 1, *psf_size)

        if isotropic:
            # check that the providing filter is indeed isotropic
            center = tuple(s // 2 for s in psf_size)
            for b in range(batch_size):
                assert torch.isclose(
                    params["filter"][b, 0, center[0] + 2, center[1] + 2],
                    params["filter"][b, 0, center[0] + 2, center[1] - 2],
                )
                assert torch.isclose(
                    params["filter"][b, 0, center[0] - 2, center[1] - 2],
                    params["filter"][b, 0, center[0] - 2, center[1] - 2],
                )
                assert torch.isclose(
                    params["filter"][b, 0, center[0] - 2, center[1] - 2],
                    params["filter"][b, 0, center[0] - 2, center[1] + 2],
                )

        # providing length-2 tuple for angle_min should raise error
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=isotropic,
                sigma_min=0.5,
                sigma_max=2.0,
                angle_min=(0.0, 0.5),
                angle_max=(1.0),
                device=device,
            )
        # providing length-2 tuple for angle_max should raise error
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=True,
                sigma_min=0.5,
                sigma_max=2.0,
                angle_min=(0.0),
                angle_max=(1.0, 1.5),
                device=device,
            )
        # angle_min should be less than angle_max
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=True,
                sigma_min=0.5,
                sigma_max=2.0,
                angle_min=(1.5),
                angle_max=(0.5),
                device=device,
            )

        # Angle constructor validation: 2D only accepts single float/integer or length-1 tuple for angle_min/max, not length-2 tuple
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size, angle_min=(0.1, 0.2), angle_max=(0.2, 0.3)
            )

    elif dim == 3:
        # In 3D, generator can accept float, integer, length-1 or length-3 tuple for sigma_min/max. If different than length-3 tuple, the same min/max will be applied to all dimensions.

        for sigma_min, sigma_max in zip(
            [0.5, (0.5,), (0.5, 0.6, 0.7)], [1.0, (1.0,), (1.0, 1.1, 1.2)], strict=True
        ):
            generator = dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=isotropic,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                angle_min=(-torch.pi, 0.0, 0.0),
                angle_max=(torch.pi, 0.5 * torch.pi, 2 * torch.pi),
                device=device,
            )
            params = generator.step(batch_size=batch_size, seed=0)
            assert params["filter"].shape == (batch_size, 1, *psf_size)

        # Angle constructor validation: 3D must accept length-3
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size, angle_min=(0.1, 0.2)
            )

    # Single sigma for the whole batch -> pass an explicit sigma tensor with identical rows
    sigma_same = torch.tensor([[1.23] * dim] * batch_size, device=device)
    params_single = generator.step(batch_size=batch_size, sigma=sigma_same, seed=0)
    filt_single = params_single["filter"]
    if batch_size > 1:
        assert torch.allclose(filt_single[0], filt_single[1])

    # Different sigma per sample -> pass per-sample sigma tensor
    if batch_size > 1:
        sig0 = [(0.6 + 0.1 * i) for i in range(dim)]
        sig1 = [(1.6 + 0.1 * i) for i in range(dim)]
        sigma_tensor = torch.tensor([sig0, sig1], device=device, dtype=torch.float32)
        params_diff = generator.step(batch_size=batch_size, sigma=sigma_tensor, seed=0)
        filt_diff = params_diff["filter"]
        assert not torch.allclose(filt_diff[0], filt_diff[1])

    # Angle handling: for 2D and 3D, passing different angles per batch should change kernels
    if dim == 2 and batch_size > 1:
        # angle should change the kernel only when sigma is anisotropic
        sigma_aniso = torch.tensor([[0.6, 1.2]] * batch_size, device=device)
        angle_tensor = torch.tensor([0.0, 1.0], device=device)
        p_angle = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_aniso, seed=0
        )
        f_angle = p_angle["filter"]
        assert not torch.allclose(f_angle[0], f_angle[1])

        # check that if sigma is isotropic, angle does not change the kernel
        sigma_iso = torch.tensor([[0.9, 0.9]] * batch_size, device=device)
        p_angle_iso = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_iso, seed=0
        )
        f_angle_iso = p_angle_iso["filter"]
        assert torch.allclose(f_angle_iso[0], f_angle_iso[1])

    if dim == 3 and batch_size > 1:
        sigma_aniso = torch.tensor([[0.6, 0.8, 1.2]] * batch_size, device=device)
        angle_tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.7]], device=device)
        p_angle = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_aniso, seed=0
        )
        f_angle = p_angle["filter"]
        assert not torch.allclose(f_angle[0], f_angle[1])

        # check that if sigma is isotropic, angle does not change the kernel
        sigma_iso = torch.tensor([[0.9, 0.9, 0.9]] * batch_size, device=device)
        p_angle_iso = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_iso, seed=0
        )
        f_angle_iso = p_angle_iso["filter"]
        assert torch.allclose(f_angle_iso[0], f_angle_iso[1])


@pytest.mark.parametrize("generators", MIXTURES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_batch_sampling", [True, False])
def test_generator_mixture(generators, size, dtype, use_batch_sampling, device, rng):
    generator_pair = []
    for name in generators:
        g, _, _ = find_generator(name, size, device, dtype, rng=rng)
        generator_pair.append(g)

    mixture = dinv.physics.generator.GeneratorMixture(
        generator_pair,
        [0.5, 0.5],
        use_batch_sampling=use_batch_sampling,
        device=device,
        rng=rng,
        verbose=True,
    )

    # When two generators belong to the same class and have same output keys and shapes
    # use_batch_sampling must be True if specified
    if type(generator_pair[0]) == type(generator_pair[1]):
        assert mixture.use_batch_sampling == use_batch_sampling

        # Check that the mixture functions properly when use_batch_sampling is True
        # and all params from the batch are from the same generator (force it by using batch_size=1)
        params = mixture.step(batch_size=1, seed=0)

    params = mixture.step(batch_size=4, seed=0)
    assert isinstance(params, dict)

    # Check the set keys of produced by the mixture are the same as the keys of the individual generators
    assert set(params.keys()).intersection(
        set.union(*[set(g.step(batch_size=1, seed=0).keys()) for g in generator_pair])
    ) == set(params.keys())


#################################
### CONFOCAL BLUR GENERATOR 3D ##
#################################


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "lambda_ill,lambda_coll,expected_channels",
    [
        (489e-9, 525e-9, 1),  # single-channel (scalar wavelengths)
        ([489e-9, 561e-9], [525e-9, 620e-9], 2),  # two-channel (list wavelengths)
    ],
)
def test_confocal_blur_generator_3d(
    device, batch_size, lambda_ill, lambda_coll, expected_channels
):
    r"""
    Test ConfocalBlurGenerator3D output shapes and keys for single- and multi-channel cases.
    """
    psf_size = (5, 11, 11)
    zernike_index = (3,)  # minimal: one coefficient for speed

    generator = dinv.physics.generator.ConfocalBlurGenerator3D(
        psf_size=psf_size,
        zernike_index=zernike_index,
        lambda_ill=lambda_ill,
        lambda_coll=lambda_coll,
        device=device,
    )

    params = generator.step(batch_size=batch_size, seed=0)

    expected_keys = {
        "filter",
        "coeff_ill",
        "coeff_coll",
        "pupil_ill",
        "pupil_coll",
        "fc_ill",
        "fc_coll",
    }
    assert set(params.keys()) == expected_keys
    assert params["filter"].shape == (batch_size, expected_channels, *psf_size)
    assert params["fc_ill"].shape == (batch_size, expected_channels)
    assert params["fc_coll"].shape == (batch_size, expected_channels)

    # Reproducibility
    params2 = generator.step(batch_size=batch_size, seed=0)
    assert torch.allclose(params["filter"], params2["filter"])


########################################
### DIFFRACTION USED_ZERNIKE_INDEX TEST #
########################################


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_used", [3, 10])
def test_diffraction_used_zernike_index(device, batch_size, n_used):
    r"""
    Test DiffractionBlurGenerator.step(used_zernike_index=...) feature.

    Verifies:
    - output shape is (B, 1, H, W) regardless of subset size
    - coeff shape last dim equals n_used
    - different subsets produce different PSFs (not degenerate)
    - passing indices outside self.zernike_index raises ValueError
    """
    psf_size = (15, 15)
    full_index = list(range(3, 37))  # 34 Noll indices

    generator = dinv.physics.generator.DiffractionBlurGenerator(
        psf_size=psf_size,
        zernike_index=full_index,
        device=device,
    )

    used = full_index[:n_used]
    params = generator.step(batch_size=batch_size, seed=0, used_zernike_index=used)

    assert params["filter"].shape == (batch_size, 1, *psf_size)
    assert params["coeff"].shape[-1] == n_used

    # Different subset → different PSF
    other_used = full_index[-n_used:]
    params_other = generator.step(
        batch_size=batch_size, seed=0, used_zernike_index=other_used
    )
    assert not torch.allclose(params["filter"], params_other["filter"])

    # Passing an index not in self.zernike_index must raise
    with pytest.raises(ValueError, match="not in self.zernike_index"):
        generator.step(
            batch_size=1, used_zernike_index=[1, 2]
        )  # 1,2 not in range(3,37)
