"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""

import numpy as np
import openslide
import dask
import dask.array as da
from openslide import deepzoom as dz
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize


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
    if isinstance(path, list):
        # A path list does not make sense for this plugin
        return None

    # otherwise we return the *function* that can read ``path``.
    if path.endswith(".svs"):
        return reader_function

    # otherwise we return the *function* that can read ``path``.
    return None


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    tile_size = 512
    overlap = 0
    model = StarDist2D.from_pretrained("2D_versatile_he")

    slide = openslide.open_slide(path)
    gen = dz.DeepZoomGenerator(
        slide, tile_size=tile_size, overlap=overlap, limit_bounds=True
    )
    num_levels = gen.level_count

    @dask.delayed(pure=True)
    def get_tile(level, column, row):
        tile = gen.get_tile(level, (column, row))
        return np.array(tile).transpose((1, 0, 2))

    @dask.delayed(pure=True)
    def model_eval(tile, level, num_levels):
        if level == num_levels - 1:
            tile_norm = normalize(tile[:, :, :3])
            labels, _ = model.predict_instances(tile_norm)
            labels = render_label(labels)
        else:
            labels = np.zeros((*tile.shape[:2], 4), np.float32)
        return labels

    myPyramid = []
    myPyramidPred = []
    # Loop through each level and read the image data
    for level in reversed(range(num_levels)):
        # Get dimensions for the current level
        level_dimensions = gen.level_dimensions[level]

        sample_tile_shape = get_tile(level, 0, 0).shape.compute()
        n_tiles_x, n_tiles_y = gen.level_tiles[level]
        # Remove last tile because it has a custom size.
        if n_tiles_x <= 1 or n_tiles_y <= 1:
            print(
                f"Ignoring Level {level} with dimensions: {level_dimensions}"
            )
            continue
        else:
            print(f"Reading Level {level} with dimensions: {level_dimensions}")
        rows = range(n_tiles_y - 1)
        cols = range(n_tiles_x - 1)

        col_tile = []
        col_pred = []
        for col in cols:
            row_tile = []
            row_pred = []
            for row in rows:
                tile = get_tile(level, col, row)
                tile_pred = model_eval(tile, level, num_levels)
                row_tile.append(
                    da.from_delayed(tile, sample_tile_shape, np.uint8)
                )
                row_pred.append(
                    da.from_delayed(
                        tile_pred, (*sample_tile_shape[:2], 4), np.float32
                    )
                )
            row_tile = da.concatenate(
                row_tile, axis=1, allow_unknown_chunksizes=False
            )
            row_pred = da.concatenate(
                row_pred, axis=1, allow_unknown_chunksizes=False
            )
            col_tile.append(row_tile)
            col_pred.append(row_pred)
        myPyramid.append(
            da.concatenate(col_tile, allow_unknown_chunksizes=False)
        )
        myPyramidPred.append(
            da.concatenate(col_pred, allow_unknown_chunksizes=False)
        )

    # optional kwargs for the corresponding viewer.add_* method
    img_kwargs = {
        "multiscale": True,
        "contrast_limits": [0, 255],
    }
    label_kwargs = {
        "name": "label",
        "multiscale": True,
        "contrast_limits": [0, 1],
    }

    return [
        (myPyramid, img_kwargs, "image"),
        (myPyramidPred, label_kwargs, "image"),
    ]