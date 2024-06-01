# src/svs_reader/_reader.py

import numpy as np
import openslide

def svs_reader_function(path):
    """
    Read an SVS file and return a list of numpy arrays representing the image pyramid.
    
    Parameters
    ----------
    path : str
        Path to the SVS file.

    Returns
    -------
    data : list of numpy.ndarray
        List of numpy arrays, each representing a different level of the image pyramid.
    """
    slide = openslide.OpenSlide(path)
    myPyramid = []

    for level in range(slide.level_count):
        dims = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, dims)
        img_np = np.array(img)
        myPyramid.append(img_np)

    slide.close()
    return myPyramid

# You may need to adjust the function signature to match the expected reader signature
def napari_get_reader(path):
    if isinstance(path, list):
        path = path[0]
    if not path.endswith('.svs'):
        return None
    return svs_reader_function
