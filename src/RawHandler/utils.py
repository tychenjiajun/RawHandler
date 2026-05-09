import numpy as np
from itertools import product
import colour


def download_file_requests(url, local_filename):
    import requests

    """
    Downloads a file from a given URL using the requests library.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The desired local filename to save the downloaded file as.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"File '{local_filename}' downloaded successfully from '{url}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def get_loss(bayer1, bayer2):
    return ((bayer1 - bayer2) ** 2).mean()


def check_if_crop_is_valid(shape, crop_edges):
    if (crop_edges[:2] < 0).any() or (crop_edges[:2] > shape[0]).any():
        return False
    if (crop_edges[-2:] < 0).any() or (crop_edges[-2:] > shape[1]).any():
        return False
    return True


def align_images(
    rh1, rh2, dims, offset=(0, 0, 0, 0), max_iters=100, step_sizes=[16, 8, 4, 2]
):
    offset = np.array(offset)
    bayer1 = rh1.input_handler(dims=dims)
    img_shape = rh1.raw.shape[-2:]

    loss = get_loss(bayer1, rh2.input_handler(dims=dims + offset))

    for step_size in step_sizes:
        directions = [
            np.array((step_size, step_size, 0, 0)),
            np.array((-step_size, -step_size, 0, 0)),
            np.array((0, 0, step_size, step_size)),
            np.array((0, 0, -step_size, -step_size)),
        ]
        for _ in range(max_iters):
            starting_offset = offset.copy()
            for step_dir in directions:
                crop_edges = dims + offset + step_dir
                # Do not update if step would create an invalid crop
                if not check_if_crop_is_valid(img_shape, crop_edges):
                    continue
                temp_loss = get_loss(bayer1, rh2.input_handler(dims=crop_edges))
                if temp_loss < loss:
                    offset += step_dir
                    loss = temp_loss
            if np.all(starting_offset == offset):
                break  # No improvement for this step size
    return offset


def transform_colorspace_to_rggb(transform):
    """
    Transforms 3x3 color space transform to work with rggb color spaces.
    Args:
        transform (np.array): 3x3 numpy array that defines the colorspace transform.

    Returns:
        new_transform (np.array): 4x4 array for rggb data.
    """
    t = transform

    t00, t01, t02 = t[0, 0], t[0, 1], t[0, 2]
    t10, t11, t12 = t[1, 0], t[1, 1], t[1, 2]
    t20, t21, t22 = t[2, 0], t[2, 1], t[2, 2]

    new_transform = np.block(
        [
            [t00, t01 / 2, t01 / 2, t02],
            [t10, t11, 0.0, t12],
            [t10, 0.0, t11, t12],
            [t20, t21 / 2, t21 / 2, t22],
        ]
    )
    return new_transform


def get_xyz_to_colorspace(colorspace):
    """Return the 3×3 XYZ → *linear* RGB matrix for *colorspace*.

    Supports all 96 RGB colourspaces from colour-science:
    https://colour.readthedocs.io/en/develop/generated/colour.RGB_COLOURSPACES.html

    Legacy aliases for backward compatibility:
        "identity" → identity matrix
        "AdobeRGB" → "Adobe RGB (1998)"
        "lin_rec2020" → "ITU-R BT.2020"
    """
    alias_map = {
        "identity": "identity",
        "AdobeRGB": "Adobe RGB (1998)",
        "lin_rec2020": "ITU-R BT.2020",
    }
    resolved = alias_map.get(colorspace, colorspace)

    if resolved == "identity":
        return np.eye(3)

    # colour.XYZ_to_RGB accepts a colourspace name string and transforms
    # the given XYZ values. We transform the identity basis to extract the matrix.
    xyz_basis = np.eye(3)  # columns are [1,0,0], [0,1,0], [0,0,1] in XYZ
    rgb = colour.XYZ_to_RGB(xyz_basis, colourspace=resolved)
    return rgb.T  # shape (3, 3)


def get_colorspace_to_xyz(colorspace):
    xyz_to_colorspace = get_xyz_to_colorspace(colorspace)
    return np.linalg.inv(xyz_to_colorspace)


def make_colorspace_matrix(
    rgb_to_xyz, colorspace="lin_rec2020", xyz_to_colorspace=None
):
    """
    Computes the combination of the rgb to xyz converstion, and a convertion from xyz to the specified colorspace.
    Args:
        xyz_to_colorspace (np.array): Specify your own 3x3 matrix to convert to a colorspace. This arguement gets overwritten by the 'colorspace' arguement. (Optional)
        colorspace (str): Name of predefined colorspace: 'sRGB', 'AdobeRGB', 'lin_rec2020'. (Default 'lin_rec2020')
    Returns:
        transform (np.array): 3x3 array for rggb data.
    """
    xyz_to_colorspace = xyz_to_colorspace or get_xyz_to_colorspace(colorspace)
    transform = xyz_to_colorspace @ rgb_to_xyz
    return transform


def get_exif_data(raw_file_path):
    import exifread

    try:
        with open(raw_file_path, "rb") as f:
            tags = exifread.process_file(f)
            return tags
    except Exception as e:
        print(f"Error reading EXIF data from {raw_file_path}: {e}")
        return None


def get_bounds(M):
    corners = np.array(list(product([0, 1], repeat=3)))  # 8 RGB corners
    transformed = corners @ M.T
    min_vals = transformed.min(axis=0)
    max_vals = transformed.max(axis=0)
    return min_vals, max_vals


def normalize_adobe_rgb(img, min_vals, max_vals):
    return (img - min_vals[:, None, None]) / (max_vals - min_vals + 1e-8)[:, None, None]


def pixel_unshuffle(x, r):
    C, H, W = x.shape
    x = (
        x.reshape(C, H // r, r, W // r, r)
        .transpose(0, 2, 4, 1, 3)
        .reshape(C * r**2, H // r, W // r)
    )
    return x


def pixel_shuffle(x, r):
    C, H, W = x.shape
    x = (
        x.reshape(C // r**2, r, r, H, W)
        .transpose(0, 3, 1, 4, 2)
        .reshape(C // r**2, H * r, W * r)
    )
    return x


def get_min_max(rh, colorspace):
    transform = rh.rgb_colorspace_transform(colorspace=colorspace)
    min_vals, max_vals = get_bounds(transform)
    return min(min_vals), max(max_vals)


def scale_0_to_1(rh, image, colorspace):
    min_val, max_val = get_min_max(rh, colorspace)
    img = (image - min_val) / (max_val - min_val)
    return img


def reverse_scale_0_to_1(rh, image, colorspace):
    min_val, max_val = get_min_max(rh, colorspace)
    img = image * (max_val - min_val) + min_val
    return img


def linear_to_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)


def linear_to_srgb_torch(x):
    import torch

    a = 0.055
    threshold = 0.0031308
    low = 12.92 * x
    high = (1 + a) * torch.pow(x.clamp(min=1e-8), 1 / 2.4) - a
    return torch.where(x <= threshold, low, high)


def safe_crop(img: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    h, w = img.shape[:2]

    # Compute slice boundaries explicitly
    y0 = dy
    y1 = h - dy if dy > 0 else h
    x0 = dx
    x1 = w - dx if dx > 0 else w

    return img[y0:y1, x0:x1]


def sparse_representation_three_channel(image, pattern="RGGB", cfa_type="bayer"):
    """
    Make a sparse representation of a C, H, W image.

    Args:
        image: numpy array (3, H, W) image.
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer.
        cfa_type: "bayer" or "xtrans".

    Returns:
        rgb: numpy array (3, H, W, 3).
    """
    C, H, W = image.shape

    if cfa_type == "bayer":
        # Generate sparse R, G, B channels
        sparse = np.zeros((C, H, W), dtype=image.dtype)

        masks = {
            "RGGB": np.array([["R", "G"], ["G", "B"]]),
            "BGGR": np.array([["B", "G"], ["G", "R"]]),
            "GRBG": np.array([["G", "R"], ["B", "G"]]),
            "GBRG": np.array([["G", "B"], ["R", "G"]]),
        }
        cmap = {"R": 0, "G": 1, "B": 2}
        mask = masks[pattern]

        for i in range(2):
            for j in range(2):
                ch = cmap[mask[i, j]]
                sparse[ch, i::2, j::2] = image[ch, i::2, j::2]

    elif cfa_type == "xtrans":
        sparse = np.zeros((3, H, W), dtype=image.dtype)

        xtrans_pattern = np.array(
            [
                ["G", "B", "R", "G", "R", "B"],
                ["R", "G", "G", "B", "G", "G"],
                ["B", "G", "G", "R", "G", "G"],
                ["G", "R", "B", "G", "B", "R"],
                ["B", "G", "G", "R", "G", "G"],
                ["R", "G", "G", "B", "G", "G"],
            ]
        )
        cmap = {"R": 0, "G": 1, "B": 2}

        for i in range(6):
            for j in range(6):
                ch = cmap[xtrans_pattern[i, j]]
                sparse[ch, i::6, j::6] = image[ch, i::6, j::6]
    return sparse


def sparse_representation(cfa, pattern="RGGB", cfa_type="bayer"):
    """
    Make a sparse representation of a CFA.

    Args:
        cfa: numpy array (H, W), single-channel CFA image.
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer.
        cfa_type: "bayer" or "xtrans".

    Returns:
        rgb: numpy array (3, H, W, 3).
    """
    H, W = cfa.shape

    if cfa_type == "bayer":
        # Generate sparse R, G, B channels
        sparse = np.zeros((3, H, W), dtype=cfa.dtype)

        masks = {
            "RGGB": np.array([["R", "G"], ["G", "B"]]),
            "BGGR": np.array([["B", "G"], ["G", "R"]]),
            "GRBG": np.array([["G", "R"], ["B", "G"]]),
            "GBRG": np.array([["G", "B"], ["R", "G"]]),
        }
        cmap = {"R": 0, "G": 1, "B": 2}
        mask = masks[pattern]

        for i in range(2):
            for j in range(2):
                ch = cmap[mask[i, j]]
                sparse[ch, i::2, j::2] = cfa[i::2, j::2]

    elif cfa_type == "xtrans":
        sparse = np.zeros((3, H, W), dtype=cfa.dtype)

        xtrans_pattern = np.array(
            [
                ["G", "B", "R", "G", "R", "B"],
                ["R", "G", "G", "B", "G", "G"],
                ["B", "G", "G", "R", "G", "G"],
                ["G", "R", "B", "G", "B", "R"],
                ["B", "G", "G", "R", "G", "G"],
                ["R", "G", "G", "B", "G", "G"],
            ]
        )
        cmap = {"R": 0, "G": 1, "B": 2}

        for i in range(6):
            for j in range(6):
                ch = cmap[xtrans_pattern[i, j]]
                sparse[ch, i::6, j::6] = cfa[i::6, j::6]
    return sparse


def sparse_representation_and_mask(cfa, pattern):
    H, W = cfa.shape
    ph, pw = pattern.shape
    # If two green channels, set both to 1
    pattern[pattern == 3] = 1
    # Create the output arrays
    sparse = np.zeros((3, H, W), dtype=cfa.dtype)
    mask = np.zeros((3, H, W), dtype=np.uint8)

    # Tile the pattern to match the CFA shape
    full_pattern = np.tile(pattern, (H // ph + 1, W // pw + 1))
    full_pattern = full_pattern[:H, :W]

    # Vectorized assignment for each channel (R, G, B)
    for ch in range(3):
        ch_mask = full_pattern == ch
        mask[ch] = ch_mask
        sparse[ch] = cfa * ch_mask
    return sparse, mask
