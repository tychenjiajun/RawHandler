import numpy as np
import rawpy
from typing import NamedTuple, Optional
from RawHandler.utils import sparse_representation_three_channel
from RawHandler.MetaDataHandler import MetaDataHandler
from RawHandler.dng_utils import to_dng
from typing import Literal, Tuple

from RawHandler.utils import (
    get_xyz_to_colorspace,
    pixel_unshuffle,
    sparse_representation_and_mask,
)


# Define a NamedTuple for the core metadata required by BaseRawHandler for processing
class CoreRawMetadata(NamedTuple):
    black_level_per_channel: np.ndarray
    white_level: int
    rgb_xyz_matrix: np.ndarray
    raw_pattern: np.ndarray
    camera_white_balance: np.ndarray
    iheight: int
    iwidth: int


class BaseRawHandlerRawpy:
    """
    Base class for handling raw image pixel data.

    Args:
        pixel_array (np.array): A 2D NumPy array representing the raw pixel data.
        core_metadata (CoreRawMetadata): A NamedTuple containing essential metadata for processing.
        full_metadata (Optional[FullRawMetadata]): A class wrapping exiv2 to handle metadata information.
    """

    def __init__(
        self,
        rawpy_object: rawpy.RawPy,
        core_metadata: CoreRawMetadata,
        full_metadata: Optional[dict] = None,
        colorspace: Literal[
            "camera", "XYZ", "sRGB", "AdobeRGB", "lin_rec2020"
        ] = "lin_rec2020",
    ):
        if not isinstance(core_metadata, CoreRawMetadata):
            raise TypeError("core_metadata must be an instance of CoreRawMetadata.")

        self.rawpy_object = rawpy_object
        self.core_metadata = core_metadata
        self.full_metadata = full_metadata if full_metadata is not None else None
        self.colorspace = colorspace
        self.camera_linear = None

    def compute_linear(self, subtract_black=False):
        if not subtract_black:
            self.camera_linear = (
                self.rawpy_object.postprocess(
                    user_wb=[1, 1, 1, 1],
                    output_color=rawpy.ColorSpace.raw,
                    no_auto_bright=True,
                    use_camera_wb=False,
                    use_auto_wb=False,
                    gamma=(1, 1),
                    user_flip=0,
                    output_bps=16,
                    user_black=0,
                    no_auto_scale=True,
                )
                / self.core_metadata.white_level
            ).transpose(2, 0, 1)
        else:
            self.camera_linear = (
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
                    / self.core_metadata.white_level
                ).transpose(2, 0, 1)  

    # orig_dims = camera_linear.shape
    # rgb_to_xyz = self.core_metadata.rgb_xyz_matrix[:3]
    # camera_linear = (rgb_to_xyz @ camera_linear.reshape(3, -1)).reshape(orig_dims)
    # self.camera_linear = camera_linear

    def _input_handler(self, dims=None, safe_crop=0) -> np.ndarray:
        """
        Crops linear array.
        """
        if self.camera_linear is None:
            self.compute_linear()
        if dims is not None:
            h1, h2, w1, w2 = dims
            if safe_crop:
                h1, h2, w1, w2 = list(
                    map(lambda x: x - x % safe_crop, [h1, h2, w1, w2])
                )
            return self.camera_linear[:, h1:h2, w1:w2]
        else:
            return self.camera_linear

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
        safe_crop=0,
        xyz_to_colorspace: np.ndarray = None,
        colorspace=None,
        clip=False,
    ) -> np.ndarray:
        """
        Converts or returns linear data converted into specified colorspace.
        """
        camera_linear = self._input_handler(dims=dims, safe_crop=safe_crop)
        rgb_transform = self.rgb_colorspace_transform(
            colorspace=colorspace, xyz_to_colorspace=xyz_to_colorspace
        )
        orig_dims = camera_linear.shape
        transformed = (rgb_transform @ camera_linear.reshape(3, -1)).reshape(orig_dims)
        if clip:
            transformed = np.clip(transformed, 0, 1)
        return transformed

    def compute_mask_and_sparse(
        self, dims=None, safe_crop=0, divide_by_wl=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_img = self.rawpy_object.raw_image_visible

        if dims is not None:
            h1, h2, w1, w2 = dims
            if safe_crop:
                # Replaced lambda/map with bitwise/integer math for speed
                h1 -= h1 % safe_crop
                h2 -= h2 % safe_crop
                w1 -= w1 % safe_crop
                w2 -= w2 % safe_crop
            raw_img = raw_img[h1:h2, w1:w2]
            # Roll the pattern to align with crop
            pattern = np.roll(
                self.core_metadata.raw_pattern, shift=(-h1, -w1), axis=(0, 1)
            )
        else:
            pattern = self.core_metadata.raw_pattern
        # Compute sparse representation on the (potentially smaller) image
        sparse, mask = sparse_representation_and_mask(raw_img, pattern)

        # Scale by white level
        if divide_by_wl:
            # Multiply by reciprocal is often faster than division
            sparse = sparse * (1.0 / self.core_metadata.white_level)

        return sparse, mask

    def downsize(
        self, min_preview_size=256, colorspace=None, clip=False, safe_crop=0
    ) -> np.ndarray:
        _, H, W = self.camera_linear.shape
        W_steps, H_steps = H // min_preview_size - 1, W // min_preview_size - 1
        steps = min(W_steps, H_steps)
        c_first_linear = self.apply_colorspace_transform(
            colorspace=colorspace, clip=clip, safe_crop=safe_crop
        )[0]
        c_first_linear = c_first_linear[:, ::steps, ::steps]
        return c_first_linear

    def generate_thumbnail(
        self,
        min_preview_size=256,
        colorspace=None,
        clip=False,
        safe_crop=0,
    ) -> np.ndarray:
        c_first_linear = self.downsize(
            min_preview_size=min_preview_size,
            colorspace=colorspace,
            clip=clip,
            safe_crop=safe_crop,
        )
        return c_first_linear

    def as_rgb(
        self,
        colorspace=None,
        dims=None,
        clip=False,
        safe_crop=0,
    ) -> np.ndarray:
        c_first_linear = self.apply_colorspace_transform(
            colorspace=colorspace, dims=dims, safe_crop=safe_crop
        )
        if clip:
            c_first_linear = np.clip(c_first_linear, 0, 1)
        return c_first_linear

    def as_sparse(
        self,
        colorspace=None,
        dims=None,
        clip=False,
        safe_crop=0,
        pattern="RGGB",
        cfa_type="bayer",
    ) -> np.ndarray:
        c_first_linear = self.apply_colorspace_transform(
            colorspace=colorspace, dims=dims, safe_crop=safe_crop
        )
        sparse = sparse_representation_three_channel(
            c_first_linear, pattern=pattern, cfa_type=cfa_type
        )
        if clip:
            sparse = np.clip(sparse, 0, 1)
        return sparse

    def as_cfa(self, **kwargs) -> np.ndarray:
        sparse = self.as_sparse(**kwargs)
        return sparse.sum(axis=0, keepdims=True)

    def as_rggb(self, cfa_type="bayer", **kwargs) -> np.ndarray:
        cfa = self.as_CFA(**kwargs)
        if cfa_type == "bayer":
            rggb = pixel_unshuffle(cfa, 2)
        else:
            rggb = pixel_unshuffle(cfa, 6)
        return rggb

    def to_dng(self, filepath, uint_img=None, user_black_level=None):
        try:
            to_dng(self, filepath, uint_img=uint_img, user_black_level=user_black_level)
            return True
        except Exception as e:
            print(e)
            return False


class RawHandlerRawpy:
    """
    Factory class to create BaseRawHandlerRawpy instances from raw image files.
    This class handles rawpy specific parsing for pixel data and core metadata,
    and uses exifread for extracting general EXIF metadata.

    Args:
        path (string): Path to raw file.
    """

    def __new__(cls, path: str, **kwargs):
        # Use rawpy for raw pixel data and core processing metadata
        rawpy_object = rawpy.imread(path)

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

        # Extract Metadata using exiv2
        metadata = MetaDataHandler(path)

        return BaseRawHandlerRawpy(
            rawpy_object=rawpy_object,
            core_metadata=core_metadata,
            full_metadata=metadata,
            **kwargs,
        )
