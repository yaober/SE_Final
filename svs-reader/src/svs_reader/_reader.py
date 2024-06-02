"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""

import numpy as np
import openslide


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
    slide = openslide.OpenSlide(path)
    num_levels = slide.level_count

    myPyramid = []
    # Loop through each level and read the image data
    for level in range(num_levels):
        # Get dimensions for the current level
        level_dimensions = slide.level_dimensions[level]
        print(f"Reading Level {level} with dimensions: {level_dimensions}")

        # Read the entire region for the current level
        region = slide.read_region((0, 0), level, level_dimensions)

        # Convert the region to a numpy array
        region_array = np.array(region)

        # Append the numpy array to the list
        myPyramid.append(region_array)

    slide.close()
    add_kwargs = {
        "multiscale": True,
        "contrast_limits": [0, 255],
    }
    layer_type = "image"  # optional, default is "image"
    return [(myPyramid, add_kwargs, layer_type)]