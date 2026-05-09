import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import rawpy
from typing import NamedTuple, Optional
from RawHandler.utils import get_exif_data, sparse_representation
from RawHandler.utils import (
    get_xyz_to_colorspace,
    transform_colorspace_to_rggb,
    pixel_unshuffle,
    pixel_shuffle,
    safe_crop,
)
from RawHandler.MetaDataHandler import MetaDataHandler


# Define a NamedTuple for the core metadata required by BaseRawHandler for processing
class CoreRawMetadata(NamedTuple):
    black_level_per_channel: np.ndarray
    white_level: int
    rgb_xyz_matrix: np.ndarray
    raw_pattern: np.ndarray
    camera_white_balance: np.ndarray
    iheight: int
    iwidth: int


class BaseRawHandler:
    """
    Base class for handling raw image pixel data.

    Args:
        pixel_array (np.array): A 2D NumPy array representing the raw pixel data.
        core_metadata (CoreRawMetadata): A NamedTuple containing essential metadata for processing.
        full_metadata (Optional[FullRawMetadata]): A Dict containing additional, general metadata.
        rawpy_object (optional): A rawpy object.
    """

    def __init__(
        self,
        pixel_array: np.ndarray,
        core_metadata: CoreRawMetadata,
        full_metadata: MetaDataHandler,
        colorspace: str = "lin_rec2020",
        rawpy_object = None
    ):
        if not isinstance(pixel_array, np.ndarray):
            raise TypeError("pixel_array must be a NumPy array.")
        if not isinstance(core_metadata, CoreRawMetadata):
            raise TypeError("core_metadata must be an instance of CoreRawMetadata.")
        self.rawpy_object = rawpy_object
        self.raw = pixel_array
        self.core_metadata = core_metadata
        self.full_metadata = full_metadata 
        self.colorspace = colorspace

    def compute_linear(self):
        camera_linear = (
            self.rawpy_object.postprocess(
                user_wb=[1, 1, 1, 1],
                output_color=rawpy.ColorSpace.raw,
                no_auto_bright=True,
                use_camera_wb=False,
                use_auto_wb=False,
                gamma=(1, 1),
                user_flip=0,
                output_bps=16,
                no_auto_scale=True,
            )
        )
        camera_linear = camera_linear / self.core_metadata.white_level
        return camera_linear.transpose(2, 0, 1)

    def _remove_masked_pixels(self, img: np.ndarray) -> np.ndarray:
        """Removes masked pixels from the image based on core_metadata.iheight and core_metadata.iwidth."""
        return img[:, 0 : self.core_metadata.iheight, 0 : self.core_metadata.iwidth]

    def flip(self, axis=1):
        raw = np.flip(self.raw, axis=axis)
        if axis == 1:
            self.raw = safe_crop(raw, dx=1, dy=0)
        else:
            self.raw = safe_crop(raw, dx=0, dy=1)

    def rotate(self, k=1):
        raw = np.rot90(self.raw, k=k)
        if k == 1:
            self.raw = safe_crop(raw, dx=0, dy=1)
        if k == 2:
            self.raw = safe_crop(raw, dx=1, dy=1)
        if k == 3:
            self.raw = safe_crop(raw, dx=1, dy=0)

    def _input_handler(self, dims=None) -> np.ndarray:
        """
        Crops bayer array.
        """
        img = np.expand_dims(self.raw, axis=0)
        img = self._remove_masked_pixels(img)
        if dims is not None:
            if len(dims) != 4:
                raise ValueError(
                    f"Arguments must be length 0 or 4, found length {dims}."
                )
            # Center on Bayer grid
            h1, h2, w1, w2 = dims
            h1, h2, w1, w2 = list(map(lambda x: x - x % 2, [h1, h2, w1, w2]))
            return img[:, h1:h2, w1:w2]
        else:
            return img

    def _adjust_bayer_bw_levels(self, dims=None, clip=False) -> np.ndarray:
        """
        Adjusts black and white levels of Bayer data.
        """
        img = self._input_handler(dims=dims)
        img = img.astype(np.float32)

        bayer_map = self._make_bayer_map(img)
        for channel in range(4):
            channel_mask = bayer_map == channel
            img[channel_mask] -= self.core_metadata.black_level_per_channel[channel]
            img[channel_mask] *= 1.0 / (
                self.core_metadata.white_level
                - self.core_metadata.black_level_per_channel[channel]
            )
        if clip:
            img = np.clip(img, 0, 1)
        return img

    def _make_bayer_map(self, bayer: np.ndarray) -> np.ndarray:
        """Creates a Bayer channel map."""
        channel_map = np.zeros_like(bayer, dtype=int)
        channel_map[0, 0::2, 0::2] = 0  # Red
        channel_map[0, 0::2, 1::2] = 1  # Green (G1)
        channel_map[0, 1::2, 0::2] = 3  # Green (G2)
        channel_map[0, 1::2, 1::2] = 2  # Blue
        return channel_map

    def rgb_colorspace_transform(self, colorspace=None, xyz_to_colorspace=None) -> np.ndarray:
        """Return the 3×3 matrix that converts camera RGB → target colourspace.

        The camera's ``rgb_xyz_matrix`` (from rawpy) is the RGB → CIE XYZ
        conversion.  The transform is:

            target_RGB = camera_RGB @ (xyz_to_target @ rgb_xyz_matrix).T

        Parameters
        ----------
        colorspace : str
            Target colourspace (e.g. ``"sRGB"``, ``"Display P3"``, ``"XYZ"``,
            ``"camera"``).  Defaults to the instance ``colorspace`` attribute.
        xyz_to_colorspace : np.ndarray, optional
            A custom 3×3 XYZ → linear RGB matrix. If provided, ``colorspace``
            is ignored (except ``"camera"`` and ``"XYZ"`` shortcuts).

        Returns
        -------
        np.ndarray
            3×3 transformation matrix.
        """
        colorspace = colorspace or self.colorspace
        if colorspace == "camera":
            return np.eye(3)

        camera_rgb_to_xyz = self.core_metadata.rgb_xyz_matrix[:3]

        if xyz_to_colorspace is None:
            if colorspace == "XYZ":
                return camera_rgb_to_xyz
            xyz_to_colorspace = get_xyz_to_colorspace(colorspace)

        return xyz_to_colorspace @ camera_rgb_to_xyz

    def apply_colorspace_transform(
        self,
        dims=None,
        xyz_to_colorspace: np.ndarray = None,
        colorspace=None,
        clip=False,
    ) -> np.ndarray:
        """
        Converts or returns rggb data converted into specified RGB colorspace.

        For non-RGB / perceptual colourspaces (CIELAB, Oklab, etc.),
        use :meth:`to_perceptual` instead.
        """
        img = self._adjust_bayer_bw_levels(dims=dims)
        rggb = pixel_unshuffle(img, 2)
        transform = self.rgb_colorspace_transform(
            colorspace=colorspace, xyz_to_colorspace=xyz_to_colorspace
        )
        rggb_transform = transform_colorspace_to_rggb(transform)
        orig_dims = rggb.shape
        transformed = (rggb_transform @ rggb.reshape(4, -1)).reshape(orig_dims)
        if clip:
            transformed = np.clip(transformed, 0, 1)
        return pixel_shuffle(transformed, 2)

    def downsize(self, min_preview_size=256, colorspace=None, clip=False) -> np.ndarray:
        H, W = self.raw.shape
        W_steps, H_steps = H // min_preview_size - 1, W // min_preview_size - 1
        steps = min(W_steps, H_steps)
        raw = self.apply_colorspace_transform(colorspace=colorspace, clip=clip)[0]
        rggb = pixel_unshuffle(np.expand_dims(raw, 0), 2)[:, ::steps, ::steps]
        mosaic = pixel_shuffle(rggb, 2)
        return mosaic

    def generate_thumbnail(
        self,
        min_preview_size=256,
        colorspace=None,
        clip=False,
        demosaicing_func=demosaicing_CFA_Bayer_bilinear,
    ) -> np.ndarray:
        img = self.downsize(
            min_preview_size=min_preview_size, colorspace=colorspace, clip=clip
        )
        img = demosaicing_func(img)
        return img

    def as_rgb(
        self,
        colorspace=None,
        dims=None,
        demosaicing_func=demosaicing_CFA_Bayer_bilinear,
        clip=False,
    ) -> np.ndarray:
        bayer = self.apply_colorspace_transform(colorspace=colorspace, dims=dims)
        rgb = demosaicing_func(bayer)
        if clip:
            rgb = np.clip(rgb, 0, 1)
        return rgb.transpose(2, 0, 1)

    def as_rggb(self, colorspace=None, dims=None, clip=False) -> np.ndarray:
        bayer = self.apply_colorspace_transform(colorspace=colorspace, dims=dims)
        rggb = pixel_unshuffle(bayer, 2)
        if clip:
            rggb = np.clip(rggb, 0, 1)
        return rggb

    def as_sparse(
        self, colorspace=None, dims=None, clip=False, pattern="RGGB", cfa_type="bayer"
    ) -> np.ndarray:
        bayer = self.apply_colorspace_transform(colorspace=colorspace, dims=dims)
        sparse = sparse_representation(bayer[0], pattern=pattern, cfa_type=cfa_type)
        if clip:
            sparse = np.clip(sparse, 0, 1)
        return sparse


class RawHandler:
    """
    Factory class to create BaseRawHandler instances from raw image files.
    This class handles rawpy specific parsing for pixel data and core metadata,
    and uses exifread for extracting general EXIF metadata.

    Args:
        path (string): Path to raw file.
    """

    def __new__(cls, path: str, **kwargs):
        # Use rawpy for raw pixel data and core processing metadata
        rawpy_object = rawpy.imread(path)
        raw_image = rawpy_object.raw_image_visible
        assert rawpy_object.color_desc.decode() == "RGBG", (
            "Only raw files with Bayer patterns are supported currently."
        )

        bayer_pattern = "".join(
            map(lambda idx: "RGBG"[idx], rawpy_object.raw_pattern.flatten())
        )
        # Adjust raw_image based on Bayer pattern to align with RGGB
        CROP_OFFSETS = {
            "RGGB": (0, 0),
            "BGGR": (1, 1),
            "GBRG": (0, 1),
            "GRBG": (1, 0),
        }

        dx, dy = CROP_OFFSETS.get(bayer_pattern, (None, None))
        if dx is None:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        raw_image = safe_crop(raw_image, dx=dx, dy=dy)
        # Ensure sensor shape is even
        if raw_image.shape[0] % 2 != 0:
            raw_image = raw_image[:-1]
        if raw_image.shape[1] % 2 != 0:
            raw_image = raw_image[:, :-1]
        # Extract Core Metadata for BaseRawHandler's processing logic
        core_metadata = CoreRawMetadata(
            black_level_per_channel=rawpy_object.black_level_per_channel,
            white_level=rawpy_object.white_level,
            rgb_xyz_matrix=rawpy_object.rgb_xyz_matrix,
            raw_pattern=rawpy_object.raw_pattern,
            camera_white_balance=np.array(rawpy_object.camera_whitebalance),
            iheight=rawpy_object.sizes.iheight,
            iwidth=rawpy_object.sizes.iwidth,
        )

        # Extract Full (General) Metadata using exifread
        metadata = MetaDataHandler(path)
        return BaseRawHandler(
            pixel_array=raw_image,
            core_metadata=core_metadata,
            full_metadata=metadata,
            rawpy_object = rawpy_object,
            **kwargs,
        )
