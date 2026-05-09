"""Unit tests for get_xyz_to_colorspace and related colourspace utilities."""

import numpy as np
import pytest
from RawHandler.utils import (
    get_xyz_to_colorspace,
    get_colorspace_to_xyz,
    make_colorspace_matrix,
    transform_colorspace_to_rggb,
)
from RawHandler.RawHandler import BaseRawHandler, CoreRawMetadata
from RawHandler.MetaDataHandler import MetaDataHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["sRGB", "Adobe RGB (1998)", "Display P3", "ITU-R BT.2020",
                         "ACEScg", "ProPhoto RGB", "Apple RGB", "Don RGB 4"])
def rgb_colourspace(request):
    """A selection of representative RGB colourspaces."""
    return request.param


@pytest.fixture
def identity_xyz():
    """Identity XYZ values (white point at luminance 1.0)."""
    return np.array([[0.9504, 1.0, 1.0888]])  # D65


@pytest.fixture
def minimal_handler():
    """Create a minimal BaseRawHandler with known camera RGB→XYZ matrix."""
    # Use sRGB primaries as the "camera" for deterministic testing
    camera_rgb_to_xyz = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ])
    core = CoreRawMetadata(
        black_level_per_channel=np.array([0.0, 0.0, 0.0, 0.0]),
        white_level=65535,
        rgb_xyz_matrix=camera_rgb_to_xyz,
        raw_pattern=np.array([[0, 1], [3, 2]]),
        camera_white_balance=np.array([1.0, 1.0, 1.0, 1.0]),
        iheight=10,
        iwidth=10,
    )
    meta = MetaDataHandler.__new__(MetaDataHandler)
    meta._tags = {}
    raw = np.random.rand(10, 10).astype(np.float32)
    return BaseRawHandler(raw, core, meta, colorspace="lin_rec2020")


# ---------------------------------------------------------------------------
# get_xyz_to_colorspace
# ---------------------------------------------------------------------------

class TestGetXYZToColorspace:
    """Tests for get_xyz_to_colorspace()."""

    def test_returns_3x3_matrix(self, rgb_colourspace):
        M = get_xyz_to_colorspace(rgb_colourspace)
        assert M.shape == (3, 3)

    def test_legacy_identity(self):
        M = get_xyz_to_colorspace("identity")
        np.testing.assert_array_almost_equal(M, np.eye(3), decimal=10)

    def test_legacy_adobe_rgb_alias(self):
        """Legacy 'AdobeRGB' should resolve to 'Adobe RGB (1998)'."""
        M1 = get_xyz_to_colorspace("AdobeRGB")
        M2 = get_xyz_to_colorspace("Adobe RGB (1998)")
        np.testing.assert_array_almost_equal(M1, M2, decimal=10)

    def test_legacy_lin_rec2020_alias(self):
        """Legacy 'lin_rec2020' should resolve to 'ITU-R BT.2020'."""
        M1 = get_xyz_to_colorspace("lin_rec2020")
        M2 = get_xyz_to_colorspace("ITU-R BT.2020")
        np.testing.assert_array_almost_equal(M1, M2, decimal=10)

    def test_unknown_colourspace_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            get_xyz_to_colorspace("NonExistentSpace")

    def test_inverse_roundtrip(self, rgb_colourspace):
        """XYZ→RGB matrix should be the inverse of RGB→XYZ matrix."""
        M_xyz_to_rgb = get_xyz_to_colorspace(rgb_colourspace)
        M_rgb_to_xyz = get_colorspace_to_xyz(rgb_colourspace)
        product = M_xyz_to_rgb @ M_rgb_to_xyz
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=10)

    def test_known_srgb_values(self):
        """Verify sRGB matrix matches known reference values."""
        M = get_xyz_to_colorspace("sRGB")
        # Reference: IEC 61966-2-1 / sRGB standard
        expected = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570],
        ])
        np.testing.assert_array_almost_equal(M, expected, decimal=3)

    def test_known_adobe_rgb_values(self):
        """Verify Adobe RGB (1998) matrix matches known reference values."""
        M = get_xyz_to_colorspace("Adobe RGB (1998)")
        expected = np.array([
            [ 2.04159, -0.56501, -0.34473],
            [-0.96924,  1.87597,  0.04156],
            [ 0.01344, -0.11836,  1.01517],
        ])
        np.testing.assert_array_almost_equal(M, expected, decimal=4)

    def test_wide_gamut_matrices_are_valid(self, rgb_colourspace):
        """Wide-gamut colourspaces should produce well-conditioned matrices."""
        M = get_xyz_to_colorspace(rgb_colourspace)
        # Matrix should be invertible
        assert np.linalg.cond(M) < 1e10, f"Matrix for {rgb_colourspace} is ill-conditioned"
        # Determinant should be non-zero
        assert abs(np.linalg.det(M)) > 1e-10

    def test_srgb_rgb_to_xyz_consistency(self):
        """RGB→XYZ→RGB roundtrip should return the original values."""
        rgb = np.array([[0.5, 0.3, 0.1]])
        M_xyz_to_rgb = get_xyz_to_colorspace("sRGB")
        M_rgb_to_xyz = get_colorspace_to_xyz("sRGB")
        xyz = rgb @ M_rgb_to_xyz.T
        rgb_back = xyz @ M_xyz_to_rgb.T
        np.testing.assert_array_almost_equal(rgb, rgb_back, decimal=10)

    def test_cross_colourspace_conversion(self):
        """Convert sRGB→XYZ→Display P3 should be consistent."""
        srgb = np.array([[0.8, 0.6, 0.4]])
        M_srgb = get_colorspace_to_xyz("sRGB")
        M_p3 = get_xyz_to_colorspace("Display P3")
        xyz = srgb @ M_srgb.T
        p3 = xyz @ M_p3.T
        # P3 values should be different from sRGB (different gamut)
        assert not np.allclose(srgb, p3, atol=1e-10)
        # But all values should be in a reasonable range (not NaN or Inf)
        assert np.all(np.isfinite(p3))

    def test_make_colorspace_matrix_composes_correctly(self):
        """make_colorspace_matrix should compose rgb_to_xyz with xyz_to_colorspace."""
        rgb_to_xyz = np.array([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ])
        M = make_colorspace_matrix(rgb_to_xyz, colorspace="sRGB")
        # Should equal xyz_to_sRGB @ rgb_to_xyz
        xyz_to_srgb = get_xyz_to_colorspace("sRGB")
        expected = xyz_to_srgb @ rgb_to_xyz
        np.testing.assert_array_almost_equal(M, expected, decimal=10)

    def test_transform_colorspace_to_rggb_matches_3x3(self):
        """The 4×4 RGGB transform should embed the 3×3 transform correctly."""
        M3 = get_xyz_to_colorspace("sRGB")
        M4 = transform_colorspace_to_rggb(M3)
        assert M4.shape == (4, 4)
        # The 3×3 embedded in 4×4 should be consistent:
        # RGGB has two green channels that each get half the green weight
        t00, t01, t02 = M3[0, 0], M3[0, 1], M3[0, 2]
        t10, t11, t12 = M3[1, 0], M3[1, 1], M3[1, 2]
        t20, t21, t22 = M3[2, 0], M3[2, 1], M3[2, 2]
        assert M4[0, 0] == t00
        assert M4[0, 1] == pytest.approx(t01 / 2)
        assert M4[0, 2] == pytest.approx(t01 / 2)
        assert M4[0, 3] == t02
        assert M4[1, 1] == t11
        assert M4[2, 2] == t11
        assert M4[3, 3] == t22


# ---------------------------------------------------------------------------
# rgb_colorspace_transform (BaseRawHandler method)
# ---------------------------------------------------------------------------

class TestRGBColorspaceTransform:
    """Tests for BaseRawHandler.rgb_colorspace_transform()."""

    def test_camera_returns_identity(self, minimal_handler):
        """colorspace='camera' should return the 3×3 identity matrix."""
        M = minimal_handler.rgb_colorspace_transform(colorspace="camera")
        np.testing.assert_array_almost_equal(M, np.eye(3), decimal=10)

    def test_xyz_returns_camera_rgb_to_xyz(self, minimal_handler):
        """colorspace='XYZ' should return the camera's RGB→XYZ matrix directly."""
        M = minimal_handler.rgb_colorspace_transform(colorspace="XYZ")
        expected = minimal_handler.core_metadata.rgb_xyz_matrix[:3]
        np.testing.assert_array_almost_equal(M, expected, decimal=10)

    def test_default_uses_instance_colorspace(self, minimal_handler):
        """No argument should fall back to the instance's colourspace attribute."""
        # Instance was created with colorspace="lin_rec2020" (→ ITU-R BT.2020)
        M = minimal_handler.rgb_colorspace_transform()
        camera_rgb_to_xyz = minimal_handler.core_metadata.rgb_xyz_matrix[:3]
        xyz_to_bt2020 = get_xyz_to_colorspace("ITU-R BT.2020")
        expected = xyz_to_bt2020 @ camera_rgb_to_xyz
        np.testing.assert_array_almost_equal(M, expected, decimal=10)

    def test_override_colorspace_parameter(self, minimal_handler):
        """Passing a colourspace argument should override the instance default."""
        M = minimal_handler.rgb_colorspace_transform(colorspace="sRGB")
        camera_rgb_to_xyz = minimal_handler.core_metadata.rgb_xyz_matrix[:3]
        xyz_to_srgb = get_xyz_to_colorspace("sRGB")
        expected = xyz_to_srgb @ camera_rgb_to_xyz
        np.testing.assert_array_almost_equal(M, expected, decimal=10)

    def test_srgb_is_identity_when_camera_is_srgb(self):
        """If camera RGB primaries == sRGB and target == sRGB, transform should be identity."""
        xyz_to_srgb = get_xyz_to_colorspace("sRGB")
        rgb_to_xyz = np.linalg.inv(xyz_to_srgb)  # sRGB RGB→XYZ
        core = CoreRawMetadata(
            black_level_per_channel=np.array([0.0, 0.0, 0.0, 0.0]),
            white_level=65535,
            rgb_xyz_matrix=rgb_to_xyz,
            raw_pattern=np.array([[0, 1], [3, 2]]),
            camera_white_balance=np.array([1.0, 1.0, 1.0, 1.0]),
            iheight=10,
            iwidth=10,
        )
        meta = MetaDataHandler.__new__(MetaDataHandler)
        meta._tags = {}
        raw = np.random.rand(10, 10).astype(np.float32)
        handler = BaseRawHandler(raw, core, meta, colorspace="sRGB")
        M = handler.rgb_colorspace_transform(colorspace="sRGB")
        # xyz_to_sRGB @ rgb_to_xyz = I
        np.testing.assert_array_almost_equal(M, np.eye(3), decimal=10)

    def test_adobe_rgb_transform(self, minimal_handler):
        """camera(sRGB) → Adobe RGB (1998) should produce a valid transform."""
        M = minimal_handler.rgb_colorspace_transform(colorspace="Adobe RGB (1998)")
        camera_rgb_to_xyz = minimal_handler.core_metadata.rgb_xyz_matrix[:3]
        xyz_to_adobe = get_xyz_to_colorspace("Adobe RGB (1998)")
        expected = xyz_to_adobe @ camera_rgb_to_xyz
        np.testing.assert_array_almost_equal(M, expected, decimal=10)
        assert np.linalg.cond(M) < 1e10

    def test_acescg_transform(self, minimal_handler):
        """camera(sRGB) → ACEScg should produce a valid transform."""
        M = minimal_handler.rgb_colorspace_transform(colorspace="ACEScg")
        camera_rgb_to_xyz = minimal_handler.core_metadata.rgb_xyz_matrix[:3]
        xyz_to_acescg = get_xyz_to_colorspace("ACEScg")
        expected = xyz_to_acescg @ camera_rgb_to_xyz
        np.testing.assert_array_almost_equal(M, expected, decimal=10)
        assert np.all(np.isfinite(M))

    def test_wide_gamut_transforms(self, minimal_handler, rgb_colourspace):
        """All RGB colourspaces should produce well-conditioned transforms."""
        M = minimal_handler.rgb_colorspace_transform(colorspace=rgb_colourspace)
        assert M.shape == (3, 3)
        assert np.linalg.cond(M) < 1e10
        assert abs(np.linalg.det(M)) > 1e-10
        assert np.all(np.isfinite(M))

    def test_transform_applied_to_camera_rgb(self, minimal_handler):
        """Apply the transform to camera RGB values → should get target RGB."""
        camera_rgb = np.array([[0.8, 0.6, 0.4]])
        M = minimal_handler.rgb_colorspace_transform(colorspace="Display P3")
        target_rgb = camera_rgb @ M.T
        # Verify: camera_rgb → XYZ → Display P3
        camera_rgb_to_xyz = minimal_handler.core_metadata.rgb_xyz_matrix[:3]
        xyz = camera_rgb @ camera_rgb_to_xyz.T
        xyz_to_p3 = get_xyz_to_colorspace("Display P3")
        expected = xyz @ xyz_to_p3.T
        np.testing.assert_array_almost_equal(target_rgb, expected, decimal=10)

    def test_unknown_colourspace_raises(self, minimal_handler):
        """An unknown colourspace should raise ValueError."""
        with pytest.raises(ValueError, match="invalid"):
            minimal_handler.rgb_colorspace_transform(colorspace="NonExistent")
