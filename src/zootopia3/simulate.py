import numpy as np
import numpy.typing as npt
from numpy.dtypes import StringDType
from random import randint
from .shapes import Shape
from .waves import create_sinusoid

def simulate_sinusoids(num_each,
                        amplitudes,
                        frequencies,
                        phases,
                        vertical_shifts,
                        length=50,
                        ) -> npt.NDArray[np.uint8]:
    '''Simulate the data of many sine waves.

    Parameters
    ----------
    amplitudes : list[float]
    frequencies : list[float]
    phases : list[float]
    vertical_shifts : list[float]

    Returns
    -------
        Two-dimensional array, where each row is a sine wave

    '''
    itr = 0
    X = np.empty(
        (num_each * len(amplitudes) * len(frequencies) * len(phases) * len(vertical_shifts), 
            length
        ),
        dtype=np.float32
    )
    for _ in range(num_each):
        for amplitude in amplitudes:
            for frequency in frequencies:
                for phase in phases:
                    for vertical_shift in vertical_shifts:
                        X[itr,] = create_sinusoid(
                            amplitude,
                            frequency,
                            phase,
                            vertical_shift,
                            length
                        )
                        itr += 1
    return X

def perturb_color(rgb_color, magnitude):
    '''Modify the RGB color by adding or subtracting a random value.

    Args:
        rgb_color: Tuple of (R, G, B) values.
        magnitude: Maximum value to add or subtract from each color channel.

    Returns:
        Tuple of perturbed (R, G, B) values.
    '''
    assert magnitude < 100
    r = rgb_color[0]
    g = rgb_color[1]
    b = rgb_color[2]
    if r == 255:
        r -= randint(0, magnitude)
    if g == 255:
        g -= randint(0, magnitude)
    if b == 255:
        b -= randint(0, magnitude)
    if r == 0:
        r += randint(0, magnitude)
    if g == 0:
        g += randint(0, magnitude)
    if b == 0:
        b += randint(0, magnitude)
    return (r, g, b)

def perturb_shade(rgb_color, magnitude):
    '''Modify the RGB color by adding or subtracting a random value.

    Args:
        rgb_color: Tuple of (R, G, B) values.
        magnitude: Maximum value to add or subtract from each color channel.

    Returns:
        Tuple of perturbed (R, G, B) values.
    '''
    assert magnitude < 100
    assert (sum(rgb_color) == 0) or (sum(rgb_color) == 765)
    if sum(rgb_color) == 765:
        minus = True
    else:
        minus = False
    r = rgb_color[0]
    g = rgb_color[1]
    b = rgb_color[2]
    if minus:
        r -= randint(0, magnitude)
        g -= randint(0, magnitude)
        b -= randint(0, magnitude)
    else:
        r += randint(0, magnitude)
        g += randint(0, magnitude)
        b += randint(0, magnitude)
    return (r, g, b)

def simulate_shapes(num_examples: int,
                    shape_type: str,
                    image_height: int,
                    image_width: int, 
                    min_x: int, 
                    max_x: int, 
                    colors_dict: dict = {
                        "yellow": (255, 255, 0),
                        "blue": (0, 0, 255),
                        "red": (255, 0, 0),
                        "green": (0, 255, 0),
                        "cyan": (0, 255, 255),
                        "magenta": (255, 0, 255),
                    },
                    magnitude: int = 50,  
                    shades: bool =True, 
                    *args, 
                    **kwargs
                    ) -> tuple:
    '''
    Simulate images of shapes with perturbed colors and shades.

    Args:
        num_examples: Number of shape images to generate for each color and shade.
        shape_type: Type of shape to generate (e.g., 'circle', 'rectangle').
        image_height: Height of the generated images.
        image_width: Width of the generated images.
        min_x: Minimum size of the larger shape axis.
        max_x: Maximum size of the larger shape axis.
        colors_dict: Dictionary mapping color names to RGB tuples.
        color_magnitude: Magnitude of color perturbation.
        shape_magnitude: Magnitude of shade perturbation.
        shades: Whether to include shade perturbations.
        *args: Additional positional arguments for shape creation.
        **kwargs: Additional keyword arguments for shape creation.

    Returns:
        target_colors: List of target color names for each generated image.
        target_shapes: List of target shape names for each generated image.
        images: List of generated shape images as numpy arrays.
    '''

    n = num_examples
    if shades:
        num_examples *= 8
    else:
        num_examples *= 6
    target_color = np.empty(num_examples, dtype=StringDType)
    target_shape = np.empty(num_examples, dtype=StringDType)
    images = np.zeros(
        (num_examples,
            224,
            224,
            3,
        ),
        dtype = np.uint8
    )

    # write loop to create examples
    # identify the redundancies in ../examples/simulate.ipynb

    itr = 0

    shape_name = shape_type.lower()

    for _ in range(n):

        if shape_name == 'circle':

            # TO DO
            # circle

            pass

            # TO DO
            # add white and black as color labels

            pass

        elif shape_name == 'triangle':

            # TO DO
            # triangle

            pass

            # TO DO
            # add white and black as color labels

            pass

        else:

            # TO DO
            # not circle or triangle

            pass

            # TO DO
            # add white and black as colors

            pass

    return target_color, target_shape, images

