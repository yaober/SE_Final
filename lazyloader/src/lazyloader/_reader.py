import numpy as np

def rgb2hed(rgb):
    """
    Convert an RGB image to HED color space.

    Parameters
    ----------
    rgb : array
        RGB image.

    Returns
    -------
    hed : array
        HED image.
    """
    # Stain separation matrix
    # These values are hardcoded for convenience
    rgb_from_hed = np.array([[0.650, 0.0722, 0], 
                             [0.704, 0.990, 0], 
                             [0.286, 0.105, 0]])
    hed_from_rgb = np.linalg.inv(rgb_from_hed)

    # Convert RGB to OD (Optical Density) space
    rgb = rgb / 255.0
    od = -np.log(rgb + 1e-6)

    # Convert OD to HED
    hed = np.dot(od, hed_from_rgb.T)

    return hed


import dask.array as da

def dask_rgb2hed(rgb):
    """
    Convert an RGB image to HED color space using Dask.

    Parameters
    ----------
    rgb : dask.array
        RGB image.

    Returns
    -------
    hed : dask.array
        HED image.
    """
    rgb_from_hed = np.array([[0.650, 0.0722, 0], 
                             [0.704, 0.990, 0], 
                             [0.286, 0.105, 0]])
    hed_from_rgb = np.linalg.inv(rgb_from_hed)

    # Convert RGB to OD (Optical Density) space
    rgb = rgb / 255.0
    od = -da.log(rgb + 1e-6)

    # Convert OD to HED
    hed = da.tensordot(od, hed_from_rgb.T, axes=(-1, -1))

    return hed

def threshold_hed(hed, threshold=0.5):
    """
    Apply a threshold to the HED image to identify nuclei.

    Parameters
    ----------
    hed : dask.array
        HED image.
    threshold : float
        Threshold value.

    Returns
    -------
    labels : dask.array
        Binary image with 1 indicating nuclei.
    """
    h_channel = hed[:, :, 0]
    labels = h_channel > threshold

    return labels

import numpy as np
import openslide
import dask
import dask.array as da
from openslide import deepzoom as dz
import logging

logging.basicConfig(level=logging.DEBUG)


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list) or not path.endswith(".svs"):
        return None

    return read_svs_file


def read_svs_file(path):
    """Take a path and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided.
    """
    tile_size = 512
    overlap = 0

    slide = openslide.open_slide(path)
    zoom_gen = dz.DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
    num_levels = zoom_gen.level_count

    @dask.delayed(pure=True)
    def fetch_tile(level, col, row):
        tile = zoom_gen.get_tile(level, (col, row))
        return np.array(tile).transpose((1, 0, 2))

    def dask_rgb2hed(rgb):
        rgb_from_hed = np.array([[0.650, 0.0722, 0], 
                                 [0.704, 0.990, 0], 
                                 [0.286, 0.105, 0]])
        hed_from_rgb = np.linalg.inv(rgb_from_hed)
        rgb = rgb / 255.0
        od = -da.log(rgb + 1e-6)
        hed = da.tensordot(od, hed_from_rgb.T, axes=(-1, -1))
        return hed

    def threshold_hed(hed, threshold=0.5):
        h_channel = hed[:, :, 0]
        labels = h_channel > threshold
        return labels

    pyramid = []
    label_pyramid = []

    for level in reversed(range(num_levels)):
        level_dims = zoom_gen.level_dimensions[level]
        num_tiles_x, num_tiles_y = zoom_gen.level_tiles[level]

        if num_tiles_x <= 1 or num_tiles_y <= 1:
            print(f"Ignoring Level {level} with dimensions: {level_dims}")
            continue

        print(f"Reading Level {level} with dimensions: {level_dims}")

        sample_tile_shape = fetch_tile(level, 0, 0).shape.compute()
        rows = range(num_tiles_y - 1)
        cols = range(num_tiles_x - 1)

        level_data = da.concatenate(
            [
                da.concatenate(
                    [
                        da.from_delayed(
                            fetch_tile(level, col, row),
                            sample_tile_shape,
                            np.uint8,
                        )
                        for row in rows
                    ],
                    allow_unknown_chunksizes=False,
                    axis=1,
                )
                for col in cols
            ],
            allow_unknown_chunksizes=False,
        )

        pyramid.append(level_data)

        if level == num_levels - 1:
            hed = dask_rgb2hed(level_data)
            labels = threshold_hed(hed)
            label_pyramid.append(labels)
        else:
            label_pyramid.append(da.zeros_like(level_data[:, :, 0], dtype=bool))

    add_kwargs = {
        "multiscale": True,
        "contrast_limits": [0, 255],
    }
    layer_type = "image"

    label_kwargs = {
        "multiscale": True,
        "color": "red",
        "name": "Nuclei Labels",
    }
    label_type = "labels"

    return [(pyramid, add_kwargs, layer_type), (label_pyramid, label_kwargs, label_type)]
