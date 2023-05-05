from skimage import morphology
from skimage import measure
import numpy as np


def proposed_mask(label_rgba_values: np.array, mag1c_values: np.array) -> np.array:
    # existing label is everything is not transparent in the rgba label
    existing_label = label_rgba_values[-1] != 0

    mag1c_thresholded = mag1c_values[0] >= 200
    mag1c_thresholded_opening = morphology.binary_dilation(
        morphology.binary_opening(mag1c_thresholded, morphology.disk(1)), morphology.disk(1))

    # Apply connected components stuff, this produces an image where pixels with the same value are connected
    mag1c_thresholded_opening_connected_components = measure.label(mag1c_thresholded_opening, background=0)

    connected_components_with_labels = np.unique(mag1c_thresholded_opening_connected_components[existing_label & (
                mag1c_thresholded_opening_connected_components != 0)])

    # Select all the connected components that intersects with labels
    mag1c_connected_components_pixel_plumes = np.any(
        mag1c_thresholded_opening_connected_components[..., np.newaxis] == connected_components_with_labels, axis=-1)

    # keep only pixels with mag1c>200
    mag1c_connected_components_pixel_plumes &= mag1c_thresholded

    return mag1c_connected_components_pixel_plumes