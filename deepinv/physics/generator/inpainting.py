from __future__ import annotations
from typing import TYPE_CHECKING
from warnings import warn
import torch
import torch.nn.functional as F
from deepinv.physics.generator.base import PhysicsGenerator
from deepinv.physics.functional.rand import random_choice

if TYPE_CHECKING:
    from deepinv.physics.generator.mri import BaseMaskGenerator


class BernoulliSplittingMaskGenerator(PhysicsGenerator):
    """Base generator for splitting/inpainting masks.

    Generates binary masks with an approximate given split ratio, according to a Bernoulli distribution. Can be used either for generating random inpainting masks for :class:`deepinv.physics.Inpainting`, or random splitting masks for :class:`deepinv.loss.SplittingLoss`.

    Optional pass in input_mask to subsample this mask given the split ratio. For mask ratio to be almost exactly as specified, use this option with a flat mask of ones as input.

    |sep|

    :Examples:

        Generate random mask

        >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
        >>> gen = BernoulliSplittingMaskGenerator((1, 3, 3), split_ratio=0.6)
        >>> gen.step(batch_size=2)["mask"].shape
        torch.Size([2, 1, 3, 3])

        Generate splitting mask from given input_mask

        >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
        >>> from deepinv.physics import Inpainting
        >>> physics = Inpainting((1, 100, 100), 0.9)
        >>> gen = BernoulliSplittingMaskGenerator((1, 100, 100), split_ratio=0.6)
        >>> gen.step(batch_size=2, input_mask=physics.mask)["mask"].shape
        torch.Size([2, 1, 100, 100])

        Generate splitting mask from given `input_mask` with random split ratio for each sample in the batch

        >>> gen = BernoulliSplittingMaskGenerator((1, 100, 100), split_ratio=0.6, random_split_ratio=True, min_split_ratio=0.1, max_split_ratio=0.9)
        >>> mask = gen.step(batch_size=2, input_mask=physics.mask, seed=10)["mask"]
        >>> (mask[0] == 0).sum()/mask[0].numel()  # 0.1 < split_ratio < 0.9
        tensor(0.5782)
        >>> (mask[1] == 0).sum()/mask[1].numel()  # 0.1 < split_ratio < 0.9
        tensor(0.2905)

        Generate splitting mask with new 2D shape than that given at initialization

        >>> gen.step(img_size=(71, 73))["mask"].shape
        torch.Size([1, 1, 71, 73])

    :param tuple[int] img_size: size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, M) or (M,).
        Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
    :param float split_ratio: ratio of values to be kept.
    :param bool pixelwise: Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
    :param bool random_split_ratio: if True, `split_ratio` is randomly sampled from `[min_split_ratio, max_split_ratio]` at each step.
    :param float min_split_ratio: minimum split ratio. Only used if `random_split_ratio` is True.
    :param float max_split_ratio: maximum split ratio. Only used if `random_split_ratio` is True.
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.dtype dtype: the data type of the generated parameters
    :param torch.Generator rng: torch random number generator.
    """

    def __init__(
        self,
        img_size: tuple[int],
        split_ratio: float,
        pixelwise: bool = True,
        random_split_ratio: bool = False,
        min_split_ratio: float = 0.0,
        max_split_ratio: float = 1.0,
        device: str | torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, device=device, dtype=dtype, rng=rng, **kwargs)
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.pixelwise = pixelwise
        self.random_split_ratio = random_split_ratio
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio

    def step(
        self,
        batch_size=1,
        input_mask: torch.Tensor = None,
        img_size: tuple | None = None,
        seed: int = None,
        **kwargs,
    ) -> dict:
        r"""
        Generate a random mask.

        If ``input_mask`` is None, generates a standard random mask that can be used for :class:`deepinv.physics.Inpainting`.
        If ``input_mask`` is specified, splits the input mask into subsets given the split ratio.

        :param int batch_size: batch_size. If None, no batch dimension is created. If input_mask passed and has its own batch dimension > 1, batch_size is ignored.
        :param torch.Tensor, None input_mask: optional mask to be split. If None, all pixels are considered. If not None, only pixels where mask==1 are considered. input_mask shape can optionally include a batch dimension.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
        :param int seed: the seed for the random number generator.

        :return: dictionary with key **'mask'**: tensor of size ``(batch_size, *img_size)`` with values in {0, 1}.
        :rtype: dict
        """
        self.rng_manual_seed(seed)

        if input_mask is not None and img_size is not None:
            raise ValueError("Only input_mask or img_size can be passed, but not both.")

        if isinstance(input_mask, torch.Tensor) and len(input_mask.shape) > len(
            self.img_size
        ):
            input_mask = input_mask.to(self.device)
            if input_mask.shape[0] > 1:
                # Batch dim exists in input_mask and it's > 1
                batch_size = input_mask.shape[0]
            else:
                # Singular batch dim exists in input_mask so use batch_size
                # Removes batch dimensions
                input_mask = input_mask[0]

        if batch_size is not None:
            # Create each mask in batch independently
            outs = []
            for b in range(batch_size):
                inp = None
                if isinstance(input_mask, torch.Tensor) and len(input_mask.shape) > len(
                    self.img_size
                ):
                    inp = input_mask[b]
                elif isinstance(input_mask, torch.Tensor):
                    inp = input_mask
                outs.append(
                    self.batch_step(input_mask=inp, img_size=img_size, **kwargs)
                )
            mask = torch.stack(outs)
        else:
            mask = self.batch_step(input_mask=input_mask, img_size=img_size, **kwargs)

        return {"mask": mask}

    def check_pixelwise(self, input_mask=None) -> bool:
        r"""Check if pixelwise can be used given input_mask dimensions and img_size dimensions"""
        pixelwise = self.pixelwise

        if pixelwise and len(self.img_size) == 2:
            warn(
                "Generating pixelwise mask assumes channel in first dimension. For 2D images (i.e. of shape (H,W)) ensure img_size is at least 3D (i.e. C,H,W). However, for img_size of shape (C,M), this will work as expected."
            )
        elif pixelwise and len(self.img_size) == 1:
            warn("For 1D img_size, pixelwise must be False.")
            pixelwise = False

        if (
            isinstance(input_mask, torch.Tensor) and input_mask.numel() > 1
        ):  # Input mask is properly specified
            if pixelwise:
                if len(input_mask.shape) == 1:
                    warn("input_mask is only 1D so pixelwise cannot be used.")
                    return False
                elif len(input_mask.shape) == 2 and len(input_mask.shape) < len(
                    self.img_size
                ):
                    # When input_mask 2D, this can either be shape C,M or H,W.
                    # When input_mask C,M, img_size will also be C,M (as passed in from SplittingLoss) and pixelwise can be used safely.
                    # When input_mask H,W but img_size higher-dimensional e.g. C,H,W, then pixelwise should be set to False as it will happen anyway.
                    return False
                elif not all(
                    torch.equal(input_mask[i], input_mask[0])
                    for i in range(1, input_mask.shape[0])
                ):
                    warn("To use pixelwise, all channels must be same.")
                    return False

        return pixelwise

    def batch_step(
        self, input_mask: torch.Tensor = None, img_size: tuple | None = None
    ) -> dict:
        r"""
        Create one batch of splitting mask.

        :param torch.Tensor, None input_mask: optional mask to be split. If ``None``, all pixels are considered. If not ``None``, only pixels where ``mask==1`` are considered. Batch dimension should not be included in shape.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
        :return: mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
        """
        pixelwise = self.check_pixelwise(input_mask)
        img_size = (
            self.img_size if img_size is None else self.img_size[:-2] + img_size[-2:]
        )

        if self.random_split_ratio:
            self.split_ratio = (
                torch.rand(1, generator=self.rng, **self.factory_kwargs)
                * (self.max_split_ratio - self.min_split_ratio)
                + self.min_split_ratio
            )

        if isinstance(input_mask, torch.Tensor) and input_mask.numel() > 1:
            input_mask = input_mask.to(self.device)
            # Sample indices from given input mask
            if pixelwise:
                idx = input_mask[0, ...].nonzero(as_tuple=False)
            else:
                idx = input_mask.nonzero(as_tuple=False)

            shuff = idx[
                torch.randperm(len(idx), generator=self.rng, device=self.device)
            ]
            idx_out = shuff[: int(self.split_ratio * len(idx))].t()

            mask = torch.zeros_like(input_mask)

            if pixelwise:
                mask = mask[0, ...]
                mask[tuple(idx_out)] = 1
                mask = torch.stack([mask] * input_mask.shape[0])
            else:
                mask[tuple(idx_out)] = 1

        else:
            # Sample pixels from a uniform distribution as input_mask is not given
            mask = torch.ones(img_size, device=self.device)
            aux = torch.rand(img_size, generator=self.rng, device=self.device)
            if not pixelwise:
                mask[aux > self.split_ratio] = 0
            else:
                mask[:, aux[0, ...] > self.split_ratio] = 0

        return mask


class MultiplicativeSplittingMaskGenerator(BernoulliSplittingMaskGenerator):
    r"""Multiplicative splitting mask generator.

    Randomly generates binary masks using the given `physics_generator`, and multiplies the `input_mask` (i.e. mask that is used to create accelerated measurements).

    Given an acceleration mask :math:`M` sampled from a known distribution, this generator provides masks :math:`M'=M_1 \circ M` with :math:`M_1` sampled from `split_generator`,
    which is typically the same distribution as :math:`M`.

    .. seealso::

        :class:`deepinv.loss.mri.WeightedSplittingLoss`
            K-weighted splitting loss proposed in :footcite:t:`millard2023theoretical`,
            where this splitting mask generator is used for self-supervised learning.

    |sep|

    :Examples:

        >>> from deepinv.physics.generator import GaussianMaskGenerator, MultiplicativeSplittingMaskGenerator
        >>> physics_generator = GaussianMaskGenerator((1, 128, 128), acceleration=4)
        >>> orig_mask = physics_generator.step(batch_size=2)["mask"]
        >>> split_generator = GaussianMaskGenerator((1, 128, 128), acceleration=2)
        >>> mask_generator = MultiplicativeSplittingMaskGenerator((1, 128, 128), split_generator)
        >>> mask_generator.step(batch_size=2, input_mask=orig_mask)["mask"].shape
        torch.Size([2, 1, 128, 128])

    .. note::

        :class:`deepinv.physics.generator.MultiplicativeSplittingMaskGenerator` calls the `super().step()` function of :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` to generate the splitting mask. During initialization, we force `self` to share the same random number generator as `self.split_generator` to correctly propagate seeding to the `self.split_generator` when using `seed` argument in `step`.

    :param tuple[int] img_size: size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, T, H, W).
        Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
    :param deepinv.physics.generator.BaseMaskGenerator split_generator: mask generator used for multiplicative splitting
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    """

    def __init__(
        self,
        img_size: tuple[int],
        split_generator: BaseMaskGenerator,
        device: str | torch.device = torch.device("cpu"),
        **kwargs,
    ):
        if "split_ratio" in kwargs:
            warn(
                "split_ratio argument is ignored in MultiplicativeSplittingMaskGenerator as the split ratio is determined by the split_generator."
            )
            kwargs.pop("split_ratio")
        if "pixelwise" in kwargs:
            warn(
                "pixelwise argument is ignored in MultiplicativeSplittingMaskGenerator as the splitting is determined by the split_generator."
            )
            kwargs.pop("pixelwise")
        if "rng" in kwargs:
            warn(
                "rng argument is ignored in MultiplicativeSplittingMaskGenerator as the random number generator is shared with split_generator to ensure reproducibility when using seed in BernoulliSplittingMaskGenerator.step."
            )
            kwargs.pop("rng")

        super().__init__(
            img_size=img_size,
            split_ratio=0.0,  # unused
            pixelwise=True,  # unused
            rng=split_generator.rng,  # use same rng as split generator to ensure reproducibility when using seed in BernoulliSplittingMaskGenerator.step,
            device=device,
            **kwargs,
        )
        self.split_generator = split_generator

    def batch_step(
        self, input_mask: torch.Tensor = None, img_size: tuple | None = None
    ) -> dict:
        r"""
        Create one batch of splitting mask.

        :param torch.Tensor, None input_mask: optional mask to be split. If ``None``, all pixels are considered. If not ``None``, only pixels where ``mask==1`` are considered. Batch dimension should not be included in shape.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
        :return: mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
        """
        if isinstance(input_mask, torch.Tensor) and input_mask.numel() > 1:
            mask = self.split_generator.step(
                batch_size=1, img_size=input_mask.shape[-2:]
            )["mask"].squeeze(0)

            if input_mask.shape[-2:] == mask.shape[-2:]:
                return mask * input_mask.to(self.device)
            else:
                raise ValueError(
                    f"Input mask should be same shape as generated mask, but input has shape {input_mask.shape} and generated has shape {mask.shape}"
                )
        else:
            mask = self.split_generator.step(batch_size=1, img_size=img_size)[
                "mask"
            ].squeeze(0)
            return mask


class GaussianSplittingMaskGenerator(BernoulliSplittingMaskGenerator):
    """Randomly generate Gaussian splitting/inpainting masks.

    Generates binary masks with an approximate given split ratio, where samples are weighted according to a spatial Gaussian distribution, where pixels near the center are less likely to be kept.
    This mask is used for measurement splitting for MRI in :footcite:t:`yaman2020self`.

    Can be used either for generating random inpainting masks for :class:`deepinv.physics.Inpainting`, or random splitting masks for :class:`deepinv.loss.SplittingLoss`.

    Optional pass in input_mask to subsample this mask given the split ratio.

    Handles both 2D mask (i.e. [C, H, W] from :footcite:t:`yaman2020self` and 2D+time dynamic mask (i.e. [C, T, H, W] from :footcite:t:`acar2021self` generation. Does not handle 1D data (e.g. of shape [C, M])

    |sep|

    :Examples:

        Randomly split input mask using Gaussian weighting

        >>> from deepinv.physics.generator import GaussianSplittingMaskGenerator
        >>> from deepinv.physics import Inpainting
        >>> physics = Inpainting((1, 3, 3), 0.9)
        >>> gen = GaussianSplittingMaskGenerator((1, 3, 3), split_ratio=0.6, center_block=0)
        >>> gen.step(batch_size=2, input_mask=physics.mask)["mask"].shape
        torch.Size([2, 1, 3, 3])

    See :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` for further examples.

    :param tuple[int] img_size: size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, T, H, W).
        Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
    :param float split_ratio: ratio of values to be kept (i.e. ones).
    :param bool pixelwise: Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
    :param float std_scale: scale parameter of 2D Gaussian, in pixels.
    :param int, tuple[int] center_block: size of block in image center that is always kept for MRI autocalibration signal. Either int for square block or 2-tuple (h, w)
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.Generator rng: random number generator.
    :param torch.dtype dtype: the data type of the generated parameters
    """

    def __init__(
        self,
        img_size: tuple[int],
        split_ratio: float,
        pixelwise: bool = True,
        std_scale: float = 4.0,
        center_block: tuple[int] | int = (8, 8),
        device: torch.device = torch.device("cpu"),
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            img_size=img_size,
            split_ratio=split_ratio,
            pixelwise=pixelwise,
            device=device,
            rng=rng,
            **kwargs,
        )
        if len(img_size) < 3:
            raise ValueError(
                "img_size should be at least of shape (C, H, W). Gaussian splitting mask does not support signals of shape (C, M)."
            )
        self.std_scale = std_scale
        self.center_block = (
            (center_block, center_block)
            if isinstance(center_block, int)
            else center_block
        )

    def get_pdf(self, shape):
        """
        Generate a Gaussian distribution.

        :param tuple shape: (nx, ny) dimensions.
        :return: Gaussian Tensor of shape (nx, ny)
        """
        nx, ny = shape
        centerx, centery = nx // 2, ny // 2

        x, y = torch.meshgrid(
            torch.arange(0, nx, 1, device=self.device),
            torch.arange(0, ny, 1, device=self.device),
            indexing="ij",
        )

        gaussian = torch.exp(
            -(
                (x - centerx) ** 2 / (2 * (nx / self.std_scale) ** 2)
                + (y - centery) ** 2 / (2 * (ny / self.std_scale) ** 2)
            )
        )
        return gaussian

    def batch_step(
        self,
        input_mask: torch.Tensor = None,
        img_size: tuple | None = None,
    ) -> dict:
        r"""
        Create one batch of splitting mask using Gaussian distribution.

        Adapted from https://github.com/byaman14/SSDU/blob/main/masks/ssdu_masks.py from SSDU :footcite:t:`yaman2020self`.

        :param torch.Tensor, None input_mask: optional mask to be split. If ``None``, all pixels are considered. If not ``None``, only pixels where ``mask==1`` are considered. Batch dimension should not be included in shape.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
        :return: mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
        """
        pixelwise = self.check_pixelwise()
        _T = self.img_size[1] if len(self.img_size) > 3 else 1
        _C = self.img_size[0] if not pixelwise else 1

        # Create blank input mask if not specified. Create with time dim even if we only want static mask
        if not isinstance(input_mask, torch.Tensor) or input_mask.numel() <= 1:
            img_size = img_size if img_size is not None else self.img_size
            input_mask = torch.ones(_C, _T, *img_size[-2:], device=self.device)

        if len(input_mask.shape) < len(self.img_size):
            # Missing channel dim, so create it
            no_channel_dim = True
            input_mask = input_mask.unsqueeze(0)
            _C = 1
        else:
            no_channel_dim = False

        if len(input_mask.shape) == 3:
            # Create time dim even if we only want static mask
            input_mask = input_mask.unsqueeze(1)

        if pixelwise:
            # Only use one channel (they are all the same...)
            input_mask = input_mask[[0], ...]

        nx, ny = input_mask.shape[-2:]
        centerx, centery = nx // 2, ny // 2

        # Create PDF
        gaussian = self.get_pdf((nx, ny))

        prob_mask = input_mask * gaussian[..., :, :]  # 2D prob map

        prob_mask[
            ...,
            centerx - self.center_block[0] // 2 : centerx + self.center_block[0] // 2,
            centery - self.center_block[1] // 2 : centery + self.center_block[1] // 2,
        ] = 0

        norm_prob = prob_mask / prob_mask.sum(dim=(-2, -1), keepdim=True)

        # Fill output mask
        mask_out = torch.zeros_like(input_mask).flatten(-2)

        for c in range(_C):
            for t in range(_T):
                ind = random_choice(
                    nx * ny,
                    size=(input_mask[c, t, :, :].sum() * (1 - self.split_ratio))
                    .ceil()
                    .int()
                    .item(),
                    p=norm_prob[c, t, :, :].flatten(),
                    replace=False,
                    rng=self.rng,
                )
                mask_out[c, t, ind] = 1

        # Invert mask for output and handle dimensions
        mask_out = input_mask - mask_out.unflatten(-1, (nx, ny))

        if len(self.img_size) == 3:
            mask_out = mask_out[:, 0, ...]  # no actual time dim

        if self.pixelwise and not no_channel_dim:
            mask_out = torch.cat([mask_out] * self.img_size[0], dim=0)

        return mask_out


class Phase2PhaseSplittingMaskGenerator(BernoulliSplittingMaskGenerator):
    """Phase2Phase splitting mask generator for dynamic data.

    To be exclusively used with :class:`deepinv.loss.mri.Phase2PhaseLoss`.
    Splits dynamic data (i.e. data of shape (B, C, T, H, W)) into even and odd phases in the T dimension.

    Used in :footcite:t:`eldeniz2021phase2phase`.

    If input_mask not passed, a blank input mask is used instead.

    :param tuple[int] img_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W).
        Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.Generator rng: unused.
    """

    def __init__(
        self,
        img_size: tuple[int],
        device: torch.device = "cpu",
        rng: torch.Generator = None,
    ):
        super().__init__(
            img_size=img_size,
            split_ratio=None,
            pixelwise=None,
            device=device,
            rng=rng,
        )

    def batch_step(
        self, input_mask: torch.Tensor = None, img_size: tuple | None = None
    ) -> dict:
        r"""
        Create one batch of splitting mask.

        :param torch.Tensor, None input_mask: optional mask to be split. If ``None``, all pixels are considered. If not ``None``, only pixels where ``mask==1`` are considered. Batch dimension should not be included in shape.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
        :return: mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
        """
        if len(self.img_size) != 4:
            raise ValueError("Default img_size must be of shape (C, T, H, W)")

        if input_mask is not None and input_mask.shape != self.img_size:
            raise ValueError("input_mask must be same shape as default img_size")

        if not isinstance(input_mask, torch.Tensor) or input_mask.numel() <= 1:
            img_size = (
                self.img_size
                if img_size is None
                else self.img_size[:-2] + img_size[-2:]
            )
            input_mask = torch.ones(img_size, device=self.device)

        mask_out = torch.zeros_like(input_mask)
        mask_out[:, ::2] = input_mask[:, ::2]
        return mask_out


class Artifact2ArtifactSplittingMaskGenerator(Phase2PhaseSplittingMaskGenerator):
    """Artifact2Artifact splitting mask generator for dynamic data.

    To be exclusively used with :class:`deepinv.loss.mri.Artifact2ArtifactLoss`.
    Randomly selects a chunk from dynamic data (i.e. data of shape (B, C, T, H, W)) in the T dimension and puts zeros in the rest of the mask.

    Artifact2Artifact was introduced by :footcite:t:`liu2020rare`.

    If input_mask not passed, a blank input mask is used instead.

    :param tuple[int] img_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W).
        Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
    :param int, tuple[int] split_size: time-length of chunk. Must divide ``img_size[1]`` exactly. If ``tuple``, one is randomly selected each time.
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.Generator rng: torch random number generator.
    """

    def __init__(
        self,
        img_size: tuple[int],
        split_size: int | tuple[int] = 2,
        device: torch.device = "cpu",
        rng: torch.Generator = None,
    ):
        super().__init__(img_size, device, rng=rng)
        self.split_size = split_size
        self.prev_idx = None
        self.prev_split_size = None

    def batch_step(
        self,
        input_mask: torch.Tensor = None,
        img_size: tuple | None = None,
        persist_prev: bool = False,
    ) -> dict:
        r"""
        Create one batch of splitting mask.

        :param torch.Tensor, None input_mask: optional mask to be split. If ``None``, all pixels are considered. If not ``None``, only pixels where ``mask==1`` are considered. Batch dimension should not be included in shape.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
        :param bool persist_prev: if `True`, the selected chunk will be different from the previous time it was called. This is used so input chunk is compared to a different output chunk. Default to `False`.
        :return: mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
        """

        def rand_select(arr):
            return arr[
                torch.randint(
                    len(arr), (1,), generator=self.rng, device=self.device
                ).item()
            ]

        # Do Phase2Phase step to check input dimensions
        _ = super().batch_step(input_mask=input_mask, img_size=None)

        if not isinstance(input_mask, torch.Tensor) or input_mask.numel() <= 1:
            img_size = (
                self.img_size
                if img_size is None
                else self.img_size[:-2] + img_size[-2:]
            )
            input_mask = torch.ones(img_size, device=self.device)

        # Choose split_size
        split_size = self.split_size
        if isinstance(self.split_size, (tuple, list)):
            if persist_prev:
                split_size = self.prev_split_size
            else:
                self.prev_split_size = split_size = rand_select(self.split_size)

        # Randomly select one chunk. Don't select previous chunk if leave_prev_idx is True
        idxs = list(range(input_mask.shape[1] // split_size))
        if persist_prev:
            idxs.remove(self.prev_idx)

        self.prev_idx = idx = rand_select(idxs)

        mask_out = torch.zeros_like(input_mask)
        mask_out[:, split_size * idx : split_size * (idx + 1)] = input_mask[
            :, split_size * idx : split_size * (idx + 1)
        ]
        return mask_out


class CrosshatchTextMaskGenerator(PhysicsGenerator):
    r"""Generator for crosshatched text inpainting masks.

    Renders a line of text, tiles it over the image plane, and overlays several copies of the
    tiled field rotated by a set of angles. The result is a crosshatch pattern made of text,
    mimicking the text-overlay degradation used in text-removal inpainting benchmarks.

    Each copy is rotated about the image centre and the copies are combined by a pixelwise
    maximum, so a pixel belongs to the crosshatch as soon as it is covered by the text in at
    least one of the rotated copies. The rotation is applied with
    :func:`torch.nn.functional.grid_sample` on a square canvas of side
    ``int(sqrt(2) * max(H, W)) + 1``, large enough that no corner of the image is left
    uncovered by any rotation.

    Two modes are available:

    - ``mode="layered"`` (default): the crosshatch is built from the rotated text itself, i.e. the
      strokes of the hatch *are* the glyphs.
    - ``mode="hatch"``: the unrotated tiled text acts as a stencil which is filled with straight
      line gratings rotated by each angle, i.e. classic crosshatch shading confined to the glyphs.

    By default the text is *removed*, that is the mask is 0 on the glyphs and 1 elsewhere, which is
    the usual setup for text-removal inpainting with :class:`deepinv.physics.Inpainting`. Set
    ``invert=True`` to keep only the text instead.

    .. note::
        Text rasterization requires `Pillow <https://pillow.readthedocs.io>`_, which is installed
        alongside ``torchvision``. If ``font_path`` is not given, Pillow's built-in font is used, so
        the exact glyph shapes depend on the installed Pillow version. Pass ``font_path`` to pin a
        specific TrueType font for reproducible glyphs across machines.

    :param tuple[int] img_size: size of the mask without batch dimension, of shape `(C, H, W)` or `(H, W)`.
    :param str text: text to render into the mask.
    :param tuple[float] angles: rotation angles in degrees applied to the text (``mode="layered"``)
        or to the hatch lines (``mode="hatch"``). Defaults to ``(0.0, 45.0, 90.0)``.
    :param str mode: either ``"layered"`` (crosshatch made of rotated text) or ``"hatch"``
        (text stencil filled with rotated line gratings).
    :param int font_size: size of the rendered font in pixels. If ``None``, Pillow's default size is used.
    :param str font_path: path to a TrueType font file. If ``None``, Pillow's built-in font is used.
    :param int text_spacing: padding in pixels added around the text before tiling, controlling how
        densely the text repeats.
    :param int hatch_spacing: period in pixels of the line grating. Only used if ``mode="hatch"``.
    :param int hatch_width: thickness in pixels of the lines of the grating. Only used if ``mode="hatch"``.
    :param bool invert: if ``False`` (default), glyph pixels are set to 0 and the background to 1, so
        the text is the missing region. If ``True``, only the text is kept.
    :param bool random_angles: if ``True``, ``len(angles)`` angles are sampled uniformly in
        ``[0, 180)`` at each step instead of using ``angles``.
    :param bool random_shift: if ``True``, the tiled text field is randomly translated for each
        sample in the batch, so that the batch elements differ.
    :param str, torch.device device: device where the mask is stored (default: 'cpu').
    :param torch.dtype dtype: the data type of the generated mask.
    :param torch.Generator rng: torch random number generator.
    
    |sep|
    
    :Examples:

        Generate a crosshatched text mask made of text rotated by 0, 45 and 90 degrees:

        >>> from deepinv.physics.generator import CrosshatchTextMaskGenerator
        >>> gen = CrosshatchTextMaskGenerator((1, 64, 64), text="DEEPINV", angles=(0.0, 45.0, 90.0))
        >>> mask = gen.step(batch_size=2)["mask"]
        >>> mask.shape
        torch.Size([2, 1, 64, 64])
        >>> bool(((mask == 0) | (mask == 1)).all())  # the mask is binary
        True

        The rotation matrix used to rotate the text:

        >>> R = CrosshatchTextMaskGenerator.rotation_matrix(90.0)
        >>> bool(torch.allclose(R, torch.tensor([[0.0, -1.0], [1.0, 0.0]]), atol=1e-6))
        True

        Fill the glyphs with crosshatch shading instead, using lines at 45 and 135 degrees:

        >>> gen = CrosshatchTextMaskGenerator((1, 64, 64), mode="hatch", angles=(45.0, 135.0))
        >>> gen.step()["mask"].shape
        torch.Size([1, 1, 64, 64])

        Use the mask to build a text-removal inpainting problem:

        >>> from deepinv.physics import Inpainting
        >>> physics = Inpainting((1, 64, 64))
        >>> x = torch.randn(1, 1, 64, 64)
        >>> y = physics(x, **gen.step(batch_size=1))
        >>> y.shape
        torch.Size([1, 1, 64, 64])
    """

    def __init__(
        self,
        img_size: tuple[int],
        text: str = "DEEPINV",
        angles: tuple[float] = (0.0, 45.0, 90.0),
        mode: str = "layered",
        font_size: int = None,
        font_path: str = None,
        text_spacing: int = 4,
        hatch_spacing: int = 8,
        hatch_width: int = 2,
        invert: bool = False,
        random_angles: bool = False,
        random_shift: bool = False,
        device: str | torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, device=device, dtype=dtype, rng=rng, **kwargs)

        if mode not in ("layered", "hatch"):
            raise ValueError(f"mode must either be 'layered' or 'hatch', got '{mode}'.")
        if len(img_size) not in (2, 3):
            raise ValueError(
                f"img_size must be of shape (C, H, W) or (H, W), got {img_size}."
            )
        if len(angles) == 0:
            raise ValueError("At least one angle must be given.")

        self.img_size = img_size
        self.text = text
        self.angles = tuple(float(a) for a in angles)
        self.mode = mode
        self.font_size = font_size
        self.font_path = font_path
        self.text_spacing = text_spacing
        self.hatch_spacing = hatch_spacing
        self.hatch_width = hatch_width
        self.invert = invert
        self.random_angles = random_angles
        self.random_shift = random_shift
        self._tile_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def rotation_matrix(
        angle: float,
        device: str | torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        r"""
        Build the 2x2 rotation matrix of a given angle.

        The matrix is ``[[cos, -sin], [sin, cos]]``, so it rotates counterclockwise in a
        right-handed frame.

        :param float angle: rotation angle in degrees.
        :param str, torch.device device: device on which to build the matrix.
        :param torch.dtype dtype: data type of the matrix.
        :return: tensor of shape ``(2, 2)``.
        :rtype: torch.Tensor
        """
        theta = torch.as_tensor(angle, device=device, dtype=dtype) * torch.pi / 180.0
        cos, sin = torch.cos(theta), torch.sin(theta)
        return torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])

    def _render_text(self, text: str = None) -> torch.Tensor:
        r"""
        Rasterize a string into a binary 2D tensor using Pillow.

        Results are cached per string, as a tile only depends on the text and on
        ``font_size``, ``font_path`` and ``text_spacing``, which are fixed at construction.

        :param str text: string to rasterize. If ``None``, ``self.text`` is used.
        :return: tensor of shape ``(h, w)`` with values in {0, 1}, where 1 marks the glyphs.
        :rtype: torch.Tensor
        """
        text = self.text if text is None else text
        if text in self._tile_cache:
            return self._tile_cache[text]

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Pillow is required by CrosshatchTextMaskGenerator to rasterize text. "
                "Install it with `pip install pillow`."
            ) from e

        if self.font_path is not None:
            font = ImageFont.truetype(self.font_path, self.font_size or 20)
        elif self.font_size is not None:
            try:
                font = ImageFont.load_default(self.font_size)
            except TypeError:  # pragma: no cover - Pillow < 10.1
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Measure the text before drawing it so that no glyph is clipped.
        left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(
            (0, 0), text, font=font
        )
        pad = self.text_spacing
        # textbbox may return floats for TrueType fonts, but Pillow wants integer sizes.
        width = int(max(right - left, 1)) + 2 * pad
        height = int(max(bottom - top, 1)) + 2 * pad

        image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(image).text((pad - left, pad - top), text, fill=1, font=font)

        tile = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        self._tile_cache[text] = tile.reshape(height, width).to(**self.factory_kwargs)
        return self._tile_cache[text]

    def _tile(
        self, tile: torch.Tensor, size: int, shift: tuple = (0, 0)
    ) -> torch.Tensor:
        r"""
        Repeat a 2D tile until it covers a ``(size, size)`` canvas.

        :param torch.Tensor tile: 2D tensor to repeat.
        :param int size: side of the square canvas.
        :param tuple shift: ``(dy, dx)`` translation applied before cropping.
        :return: tensor of shape ``(size, size)``.
        :rtype: torch.Tensor
        """
        height, width = tile.shape
        reps = (-(-size // height) + 1, -(-size // width) + 1)
        canvas = tile.repeat(reps)
        canvas = torch.roll(canvas, shifts=shift, dims=(0, 1))
        return canvas[:size, :size]

    def _grating(self, size: int) -> torch.Tensor:
        r"""
        Build a horizontal line grating of period ``hatch_spacing`` and thickness ``hatch_width``.

        :param int size: side of the square canvas.
        :return: tensor of shape ``(size, size)`` with values in {0, 1}.
        :rtype: torch.Tensor
        """
        rows = torch.arange(size, device=self.device)
        lines = (rows % self.hatch_spacing < self.hatch_width).to(**self.factory_kwargs)
        return lines.unsqueeze(1).expand(size, size).contiguous()

    def _rotate(self, canvas: torch.Tensor, angle: float) -> torch.Tensor:
        r"""
        Rotate a square canvas by ``angle`` degrees about its centre.

        The sampling grid is built from the transpose of the rotation matrix, since
        :func:`torch.nn.functional.grid_sample` maps output coordinates back to input
        coordinates.

        :param torch.Tensor canvas: 2D tensor of shape ``(size, size)``.
        :param float angle: rotation angle in degrees.
        :return: rotated tensor of shape ``(size, size)``.
        :rtype: torch.Tensor
        """
        rot = self.rotation_matrix(angle, **self.factory_kwargs)
        theta = torch.zeros(1, 2, 3, **self.factory_kwargs)
        theta[:, :, :2] = rot.t()

        canvas = canvas[None, None, ...]
        grid = F.affine_grid(theta, list(canvas.shape), align_corners=False)
        rotated = F.grid_sample(
            canvas, grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
        return rotated[0, 0]

    def layer_fields(self, img_size: tuple | None = None) -> torch.Tensor:
        r"""
        Return the individual text layers, before they are combined into a single mask.

        Layer ``k`` is the text field rotated by ``angles[k]`` (``mode="layered"``), or the
        unrotated text stencil filled with the grating rotated by ``angles[k]``
        (``mode="hatch"``). :meth:`batch_step` is the pixelwise maximum of these layers.
        Useful to build a separation problem, where each layer is a source to recover.

        :param tuple img_size: if not ``None``, generate layers of this 2D shape and override
            the ``img_size`` attribute, must be of form `(H, W)`.
        :return: tensor of shape ``(len(angles), H, W)`` with values in {0, 1}, where 1 marks
            the glyphs of that layer.
        :rtype: torch.Tensor
        """
        size = self.img_size if img_size is None else self.img_size[:-2] + img_size[-2:]
        height, width = size[-2], size[-1]

        # Canvas large enough that any rotation still covers the whole image.
        side = int(2.0**0.5 * max(height, width)) + 1

        if self.random_angles:
            angles = (
                torch.rand(len(self.angles), generator=self.rng, **self.factory_kwargs)
                * 180.0
            ).tolist()
        else:
            angles = self.angles

        shift = (0, 0)
        if self.random_shift:
            shift = tuple(
                torch.randint(
                    0, side, (2,), generator=self.rng, device=self.device
                ).tolist()
            )

        tile = self._render_text()
        text_field = self._tile(tile, side, shift=shift)

        if self.mode == "layered":
            layers = [self._rotate(text_field, angle) for angle in angles]
        else:
            layers = [
                text_field * self._rotate(self._grating(side), angle)
                for angle in angles
            ]

        # Centre crop each layer back to the image size.
        top = (side - height) // 2
        left = (side - width) // 2
        return torch.stack(layers)[
            :, top : top + height, left : left + width
        ].contiguous()

    def batch_step(self, img_size: tuple | None = None) -> torch.Tensor:
        r"""
        Create one crosshatched text mask, without batch dimension.

        :param tuple img_size: if not ``None``, generate a mask of this 2D shape and override the
            ``img_size`` attribute, must be of form `(H, W)`.
        :return: mask of shape given either by ``img_size`` or the class attribute ``img_size``.
        :rtype: torch.Tensor
        """
        size = self.img_size if img_size is None else self.img_size[:-2] + img_size[-2:]
        height, width = size[-2], size[-1]

        # A pixel is crosshatched as soon as one layer covers it.
        field = self.layer_fields(img_size=img_size).amax(dim=0)

        mask = field if self.invert else 1.0 - field

        if len(size) == 3:
            mask = mask.expand(size[0], height, width)

        return mask.contiguous()

    def step(
        self,
        batch_size: int = 1,
        seed: int = None,
        img_size: tuple | None = None,
        **kwargs,
    ) -> dict:
        r"""
        Generate a batch of crosshatched text masks.

        The masks of a batch are identical unless ``random_angles`` or ``random_shift`` is set.

        :param int batch_size: batch size. If ``None``, no batch dimension is created.
        :param int seed: the seed for the random number generator.
        :param tuple img_size: if not ``None``, generate masks of this 2D image shape and override
            the ``img_size`` attribute, must be of form `(H, W)`.
        :return: dictionary with key **'mask'**: tensor of size ``(batch_size, *img_size)`` with
            values in {0, 1}.
        :rtype: dict
        """
        self.rng_manual_seed(seed)

        if batch_size is None:
            return {"mask": self.batch_step(img_size=img_size)}

        masks = [self.batch_step(img_size=img_size) for _ in range(batch_size)]
        return {"mask": torch.stack(masks)}


class MultiTextCrosshatchMaskGenerator(CrosshatchTextMaskGenerator):
    r"""Crosshatched mask whose layers each carry their own text.

    Same construction as :class:`deepinv.physics.generator.CrosshatchTextMaskGenerator`, except
    that every rotation angle gets its own string instead of repeating a single one. This lets a
    long word run in one direction and a short one in another, e.g. ``"DEEPINVERSE"`` horizontally
    and ``"TEST"`` vertically.

    ``texts`` and ``angles`` are paired elementwise, and ``texts`` is cycled when it is shorter
    than ``angles``, so ``texts=("A", "B")`` with four angles alternates A, B, A, B. Each string is
    rasterized and tiled independently, so the strings may differ in length: a longer word simply
    repeats less often across the image plane.

    :param tuple[int] img_size: size of the mask without batch dimension, of shape `(C, H, W)` or `(H, W)`.
    :param tuple[str] texts: strings to render, paired with ``angles`` and cycled if shorter.
    :param tuple[float] angles: rotation angles in degrees, one per layer.
    :param str mode: either ``"layered"`` (crosshatch made of rotated text) or ``"hatch"``
        (each text stencil filled with a line grating rotated by its own angle).
    :param kwargs: any other argument of :class:`deepinv.physics.generator.CrosshatchTextMaskGenerator`.

    |sep|

    :Examples:

        A long word across the image and a short one down it:

        >>> from deepinv.physics.generator import MultiTextCrosshatchMaskGenerator
        >>> gen = MultiTextCrosshatchMaskGenerator(
        ...     (1, 64, 64), texts=("DEEPINVERSE", "TEST"), angles=(0.0, 90.0)
        ... )
        >>> mask = gen.step(batch_size=2)["mask"]
        >>> mask.shape
        torch.Size([2, 1, 64, 64])
        >>> bool(((mask == 0) | (mask == 1)).all())  # the mask is binary
        True

        Fewer strings than angles: the strings are cycled.

        >>> gen = MultiTextCrosshatchMaskGenerator(
        ...     (1, 64, 64), texts=("AB", "C"), angles=(0.0, 45.0, 90.0, 135.0)
        ... )
        >>> gen.texts_per_angle
        ('AB', 'C', 'AB', 'C')

        A single string reproduces the parent generator exactly.

        >>> from deepinv.physics.generator import CrosshatchTextMaskGenerator
        >>> kwargs = dict(angles=(0.0, 45.0), text_spacing=2)
        >>> one = CrosshatchTextMaskGenerator((1, 32, 32), text="AB", **kwargs)
        >>> many = MultiTextCrosshatchMaskGenerator((1, 32, 32), texts=("AB",), **kwargs)
        >>> bool(torch.equal(one.step()["mask"], many.step()["mask"]))
        True
    """

    def __init__(
        self,
        img_size: tuple[int],
        texts: tuple[str] = ("DEEPINVERSE", "TEST"),
        angles: tuple[float] = (0.0, 90.0),
        **kwargs,
    ):
        if len(texts) == 0:
            raise ValueError("At least one text must be given.")

        # The parent keeps a single ``text``; use the first one so its own helpers stay usable.
        super().__init__(img_size, text=texts[0], angles=angles, **kwargs)
        self.texts = tuple(texts)

    @property
    def texts_per_angle(self) -> tuple:
        r"""
        The string used by each layer, i.e. ``texts`` cycled to the length of ``angles``.

        :return: tuple of strings, of the same length as ``angles``.
        :rtype: tuple
        """
        return tuple(
            self.texts[k % len(self.texts)] for k in range(len(self.angles))
        )

    def layer_fields(self, img_size: tuple | None = None) -> torch.Tensor:
        r"""
        Return the individual text layers, one per angle, each with its own text.

        Same as :meth:`deepinv.physics.generator.CrosshatchTextMaskGenerator.layer_fields`,
        except layer ``k`` is tiled from ``texts_per_angle[k]`` instead of a single string.

        :param tuple img_size: if not ``None``, generate layers of this 2D shape and override
            the ``img_size`` attribute, must be of form `(H, W)`.
        :return: tensor of shape ``(len(angles), H, W)`` with values in {0, 1}.
        :rtype: torch.Tensor
        """
        size = self.img_size if img_size is None else self.img_size[:-2] + img_size[-2:]
        height, width = size[-2], size[-1]

        # Canvas large enough that any rotation still covers the whole image.
        side = int(2.0**0.5 * max(height, width)) + 1

        if self.random_angles:
            angles = (
                torch.rand(len(self.angles), generator=self.rng, **self.factory_kwargs)
                * 180.0
            ).tolist()
        else:
            angles = self.angles

        shift = (0, 0)
        if self.random_shift:
            shift = tuple(
                torch.randint(
                    0, side, (2,), generator=self.rng, device=self.device
                ).tolist()
            )

        # Each layer is tiled from its own text, so the strings may have different lengths.
        layers = []
        for text, angle in zip(self.texts_per_angle, angles):
            text_field = self._tile(self._render_text(text), side, shift=shift)
            if self.mode == "layered":
                layers.append(self._rotate(text_field, angle))
            else:
                layers.append(text_field * self._rotate(self._grating(side), angle))

        # Centre crop each layer back to the image size.
        top = (side - height) // 2
        left = (side - width) // 2
        return torch.stack(layers)[
            :, top : top + height, left : left + width
        ].contiguous()
