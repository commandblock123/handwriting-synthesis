from __future__ import print_function
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


# Define the alphabet including a null character, space, punctuation, digits, and letters.
# Also added '/' as it appears in some datasets. Check if model vocabulary matches exactly.
# Let's stick to the original provided alphabet for compatibility unless model is retrained with expanded vocab.
alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]
alphabet_ord = list(map(ord, alphabet)) # Get ASCII codes
# Create mapping from character to its index in the alphabet. Unknown chars map to 0 (null).
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))
# Create mapping from index back to ASCII code.
num_to_alpha = dict(enumerate(alphabet_ord))

MAX_STROKE_LEN = 1200
MAX_CHAR_LEN = 75


def align(coords):
    """
    Corrects for global slant/offset in handwriting strokes using linear regression.
    Args:
        coords: Numpy array of shape (N, 2) or (N, 3) representing stroke coordinates (x, y, [eos]).
    Returns:
        Numpy array of shape (N, 2) with aligned coordinates.
    """
    coords_xy = np.copy(coords[:, :2])
    X = coords_xy[:, 0].reshape(-1, 1)
    Y = coords_xy[:, 1].reshape(-1, 1)

    # Prepare input matrix for linear regression (X = [1, x])
    X_padded = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

    # Calculate regression coefficients (offset, slope) using the normal equation
    try:
        # beta = inv(X^T * X) * X^T * Y
        beta = np.linalg.inv(X_padded.T.dot(X_padded)).dot(X_padded.T).dot(Y).squeeze()
        offset, slope = beta[0], beta[1]
    except np.linalg.LinAlgError:
        # Handle cases where X^T * X is singular (e.g., vertical line)
        offset, slope = 0, 0 # No correction if calculation fails

    # Calculate rotation angle (theta) from the slope
    theta = np.arctan(slope)
    # Create 2D rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    # Apply rotation and remove offset
    aligned_coords = np.dot(coords_xy, rotation_matrix) - offset
    return aligned_coords


def skew(coords, degrees):
    """
    Skews stroke coordinates horizontally by a given angle in degrees.
    Args:
        coords: Numpy array of shape (N, 2) or (N, 3).
        degrees: Skew angle in degrees.
    Returns:
        Numpy array of the same shape with skewed coordinates.
    """
    coords_copy = np.copy(coords)
    theta = degrees * np.pi / 180.0 # Convert degrees to radians
    # Skew transformation matrix
    A = np.array([[np.cos(-theta), 0], [np.sin(-theta), 1]])
    # Apply transformation to x, y coordinates
    coords_copy[:, :2] = np.dot(coords_copy[:, :2], A)
    return coords_copy


def stretch(coords, x_factor, y_factor):
    """
    Stretches stroke coordinates along the x and y axes.
    Args:
        coords: Numpy array of shape (N, 2) or (N, 3).
        x_factor: Scaling factor for the x-axis.
        y_factor: Scaling factor for the y-axis.
    Returns:
        Numpy array with stretched coordinates.
    """
    coords_copy = np.copy(coords)
    coords_copy[:, :2] *= np.array([x_factor, y_factor])
    return coords_copy


def add_noise(coords, scale):
    """
    Adds Gaussian noise to stroke coordinates (excluding the first point).
    Args:
        coords: Numpy array of shape (N, 2) or (N, 3).
        scale: Standard deviation of the Gaussian noise.
    Returns:
        Numpy array with added noise.
    """
    coords_copy = np.copy(coords)
    if len(coords_copy) > 1:
        noise = np.random.normal(loc=0.0, scale=scale, size=coords_copy[1:, :2].shape)
        coords_copy[1:, :2] += noise
    return coords_copy


def encode_ascii(ascii_string):
    """
    Encodes an ASCII string into a numpy array of integer indices based on the alphabet map.
    Unknown characters are mapped to index 0. Appends 0 at the end.
    """
    # Map each character to its index using alpha_to_num dictionary
    encoded_list = list(map(lambda char: alpha_to_num[char], ascii_string))
    return np.array(encoded_list + [0], dtype=np.int32)


def denoise(coords):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """
    # Split coordinates into sub-strokes based on eos flag
    # np.where finds indices where eos is 1. Add 1 to include the eos point in the previous stroke.
    split_indices = np.where(coords[:, 2] == 1)[0] + 1
    strokes = np.split(coords, split_indices[:-1], axis=0) # Use split_indices[:-1] to avoid empty array at end

    smoothed_strokes = []
    for stroke in strokes:
        if len(stroke) == 0: continue # Skip empty strokes

        # Savitzky-Golay filter requires window length < number of points
        # Use min(7, len(stroke)) if len(stroke) is odd, else min(7, len(stroke)-1)
        window_length = 7
        polyorder = 3

        # Adjust window length if stroke is too short
        if len(stroke) <= polyorder:
             # Cannot apply filter if length <= polyorder, keep original
             smoothed_strokes.append(stroke)
             continue
        # Ensure window_length is odd and less than stroke length
        effective_window_length = min(window_length, len(stroke))
        if effective_window_length % 2 == 0:
             effective_window_length -= 1
        # Ensure window length is greater than polyorder
        effective_window_length = max(effective_window_length, polyorder + 1 + (polyorder % 2)) # Make it odd and > polyorder


        if effective_window_length <= polyorder or effective_window_length > len(stroke):
              print(f"Warning: Skipping denoise for short stroke (len={len(stroke)})")
              smoothed_strokes.append(stroke)
              continue


        try:
            # Apply filter to x and y coordinates separately
            x_smooth = savgol_filter(stroke[:, 0], effective_window_length, polyorder, mode='nearest')
            y_smooth = savgol_filter(stroke[:, 1], effective_window_length, polyorder, mode='nearest')

            # Combine smoothed coordinates with original eos flag
            smoothed_stroke = np.stack([x_smooth, y_smooth, stroke[:, 2]], axis=-1)
            smoothed_strokes.append(smoothed_stroke)

        except ValueError as e:
             print(f"Warning: savgol_filter failed for stroke (len={len(stroke)}, win={effective_window_length}, poly={polyorder}). Error: {e}. Keeping original.")
             smoothed_strokes.append(stroke)


    # Combine the smoothed sub-strokes back into a single array
    if not smoothed_strokes: return np.zeros((0, 3)) # Handle case where input was empty
    smoothed_coords = np.vstack(smoothed_strokes)
    return smoothed_coords


def interpolate(coords, factor=2):
    """
    interpolates strokes using cubic spline
    """
    if factor <= 1: return coords # No interpolation needed

    split_indices = np.where(coords[:, 2] == 1)[0] + 1
    strokes = np.split(coords, split_indices[:-1], axis=0)

    interpolated_strokes = []
    for stroke in strokes:
        if len(stroke) == 0: continue

        # Interpolation requires at least 4 points for cubic spline (k=3)
        if len(stroke) > 3:
             try:
                 # Create time steps (parameter t) for interpolation
                 t_original = np.arange(len(stroke))
                 # Create interpolation functions for x and y
                 f_x = interp1d(t_original, stroke[:, 0], kind='cubic', bounds_error=False, fill_value="extrapolate")
                 f_y = interp1d(t_original, stroke[:, 1], kind='cubic', bounds_error=False, fill_value="extrapolate")

                 # Create new time steps for higher density
                 num_new_points = int(len(stroke) * factor)
                 t_new = np.linspace(0, len(stroke) - 1, num_new_points)

                 # Calculate interpolated coordinates
                 x_new = f_x(t_new)
                 y_new = f_y(t_new)

                 # Combine new x, y coordinates
                 xy_interpolated = np.stack([x_new, y_new], axis=-1)

                 # Create new eos flags: 0 for all interpolated points except the very last one
                 eos_new = np.zeros((len(xy_interpolated), 1), dtype=coords.dtype)
                 eos_new[-1] = 1.0 # Mark end of the interpolated sub-stroke

                 interpolated_stroke = np.concatenate([xy_interpolated, eos_new], axis=1)
                 interpolated_strokes.append(interpolated_stroke)

             except ValueError as e:
                 print(f"Warning: Interpolation failed for stroke (len={len(stroke)}). Error: {e}. Keeping original.")
                 interpolated_strokes.append(stroke) # Keep original if interpolation fails
        else:
            # If stroke is too short for cubic spline, keep it as is
            interpolated_strokes.append(stroke)

    if not interpolated_strokes: return np.zeros((0, 3))
    interpolated_coords = np.vstack(interpolated_strokes)
    return interpolated_coords


def normalize(offsets):
    """
    normalizes strokes to median unit norm
    """
    offsets_copy = np.copy(offsets)
    # Calculate Euclidean norm (magnitude) of each offset vector [dx, dy]
    norms = np.linalg.norm(offsets_copy[:, :2], axis=1)
    # Find the median norm (ignore zero norms to avoid division by zero)
    non_zero_norms = norms[norms > 1e-6] # Use a small epsilon
    if len(non_zero_norms) > 0:
         median_norm = np.median(non_zero_norms)
         if median_norm > 1e-6: # Avoid division by very small median
              # Scale dx, dy so that the median norm becomes 1
              offsets_copy[:, :2] /= median_norm
    return offsets_copy


def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    if len(coords) == 0: return np.zeros((0, 3))

    dx_dy = coords[1:, :2] - coords[:-1, :2]

    eos = coords[1:, 2:3]

    offsets_rest = np.concatenate([dx_dy, eos], axis=1)

    first_offset = np.array([[0, 0, 1.0]], dtype=coords.dtype)

    offsets = np.concatenate([first_offset, offsets_rest], axis=0)
    return offsets


def offsets_to_coords(offsets):
    """
    Converts relative offsets [dx, dy, eos] back to absolute coordinates [x, y, eos].
    Assumes the first point starts at (0, 0).
    Args:
        offsets: Numpy array of shape (N, 3).
    Returns:
        Numpy array of shape (N, 3) representing coordinates.
    """
    if len(offsets) == 0: return np.zeros((0, 3))

    # Calculate cumulative sum of dx, dy to get absolute x, y positions
    # cumsum assumes the sequence starts from 0 implicitly.
    coords_xy = np.cumsum(offsets[:, :2], axis=0)

    # Keep the original eos flags
    eos = offsets[:, 2:3] # Shape (N, 1)

    # Combine absolute coordinates and eos flags
    coords = np.concatenate([coords_xy, eos], axis=1)
    return coords


def draw(
        offsets,
        ascii_seq=None,
        align_strokes=True,
        denoise_strokes=True,
        interpolation_factor=None,
        save_file=None
):
    """
    Draws handwriting strokes using Matplotlib.
    Args:
        offsets: Numpy array of shape (N, 3) representing stroke offsets [dx, dy, eos].
        ascii_seq: Optional string or sequence of character codes to display as title.
        align_strokes: Boolean, whether to apply alignment correction.
        denoise_strokes: Boolean, whether to apply Savitzky-Golay smoothing.
        interpolation_factor: Optional integer factor for spline interpolation.
        save_file: Optional filename to save the plot image. If None, shows plot interactively.
    """
    if len(offsets) == 0:
        print("Warning: No offsets provided to draw.")
        return

    # Convert offsets to coordinates for drawing
    strokes = offsets_to_coords(offsets)

    # Apply optional preprocessing
    if denoise_strokes:
        strokes = denoise(strokes)
    if interpolation_factor is not None and interpolation_factor > 1:
        strokes = interpolate(strokes, factor=interpolation_factor)
    if align_strokes:
        # Apply alignment to the coordinates
        aligned_xy = align(strokes[:, :2])
        strokes = np.concatenate([aligned_xy, strokes[:, 2:3]], axis=1)

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(12, 3)) # Adjust figure size as needed

    # --- Draw Strokes ---
    # Iterate through points, drawing line segments between points where eos=0
    current_stroke_points = []
    for x, y, eos in strokes:
        current_stroke_points.append((x, y))
        # If end-of-stroke flag is 1, plot the completed stroke segment
        if eos == 1:
            if len(current_stroke_points) > 1: # Need at least 2 points to draw a line
                coords_x, coords_y = zip(*current_stroke_points)
                ax.plot(coords_x, coords_y, 'k-') # Plot with black lines
            # Reset for the next stroke segment
            current_stroke_points = []

    # Plot any remaining points if the last point didn't have eos=1
    if len(current_stroke_points) > 1:
        coords_x, coords_y = zip(*current_stroke_points)
        ax.plot(coords_x, coords_y, 'k-')

    # --- Axes and Appearance ---
    # Set reasonable plot limits based on data or fixed values
    # ax.set_xlim(-50, 600) # Example fixed limits
    # ax.set_ylim(-40, 40)
    ax.set_aspect('equal') # Ensure correct aspect ratio for handwriting

    # Hide ticks and labels for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.tick_params(...) # Original code had more detailed tick removal

    # Add title if provided
    if ascii_seq is not None:
        if not isinstance(ascii_seq, str):
             # If sequence of codes, try to convert back to string
             try:
                 ascii_seq = ''.join(map(chr, [num_to_alpha.get(code, ord('?')) for code in ascii_seq if code in num_to_alpha]))
             except:
                 ascii_seq = "Cannot decode title" # Fallback
        plt.title(ascii_seq)

    # Save or show plot
    if save_file is not None:
        try:
            plt.savefig(save_file, bbox_inches='tight', dpi=150) # Save with tight bounding box
            print('Plot saved to {}'.format(save_file))
        except Exception as e:
            print(f"Error saving plot to {save_file}: {e}")
    else:
        plt.show() # Show interactively

    plt.close(fig) # Close the figure to free memory