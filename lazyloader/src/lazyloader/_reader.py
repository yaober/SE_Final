"""
This module implements a reader plugin for napari that reads SVS files using
the openslide library and dask lazy loading.
"""

import numpy as np
import openslide
import dask
import dask.array as da
from openslide import deepzoom as dz


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str
        Path to file.

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

    return None


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str
        Path to file

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

    slide = openslide.open_slide(path)
    gen = dz.DeepZoomGenerator(
        slide, tile_size=tile_size, overlap=overlap, limit_bounds=True
    )
    num_levels = gen.level_count

    # lazy load the tile data
    @dask.delayed(pure=True)
    def get_tile(level, column, row):
        tile = gen.get_tile(level, (column, row))
        return np.array(tile).transpose((1, 0, 2))

    myPyramid = []
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

        arr = da.concatenate(
            [
                da.concatenate(
                    [
                        da.from_delayed(
                            get_tile(level, col, row),
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

        # Append the numpy array to the list
        myPyramid.append(arr)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {
        "multiscale": True,
        "contrast_limits": [0, 255],
    }
    layer_type = "image"

    return [(myPyramid, add_kwargs, layer_type)]