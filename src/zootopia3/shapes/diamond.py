import numpy as np
import numpy.typing as npt

def create_diamond_image(diagonal_h: int, diagonal_v: int, rgb_color: tuple) -> npt.NDArray[np.uint8]:
    """
    Create a diamond (diamond shape) with RGB color in a square pixel image.
    
    Args:
        diagonal_h: The length of the horizontal diagonal in pixels.
        diagonal_v: The length of the vertical diagonal in pixels.
        rgb_color: Tuple of (R, G, B) values (0-255)

    Returns:
        (max(diagonal_h, diagonal_v), max(diagonal_h, diagonal_v), 3) numpy array representing 
        the image with a colored diamond on black background.
    """
    # Create black background
    square_length = max(diagonal_h, diagonal_v)  
    image_array = np.random.randint(0, 256, (square_length, square_length, 3), dtype=np.uint8)
    
    # Create coordinate grids
    y, x = np.ogrid[:square_length, :square_length]
    center_x = square_length // 2
    center_y = square_length // 2
    
    # Create mask for diamond
    mask = (np.abs(x - center_x) * (diagonal_v / 2) + np.abs(y - center_y) * (diagonal_h / 2) <= (diagonal_h * diagonal_v) / 4)
    
    # Apply RGB color to diamond area
    image_array[mask] = rgb_color
    
    return image_array

class Diamond:
    '''Class representing a diamond shape with various properties and methods.'''
    def __init__(self, 
                    diagonal_h: int, 
                    diagonal_v: int, 
                    rgb_color: tuple, 
                    rgb_name: str,
                    ) -> None:
        '''Initialize a Diamond instance.

        Args:
            diagonal_h: The length of the horizontal diagonal in pixels.
            diagonal_v: The length of the vertical diagonal in pixels.
            rgb_color: Tuple of (R, G, B) values (0-255).
            rgb_name: Name of the RGB color.
        '''
        self.diagonal_h = diagonal_h
        self.diagonal_v = diagonal_v
        self.rgb_color = rgb_color
        self.rgb_name = rgb_name
        self.image = create_diamond_image(diagonal_h, diagonal_v, rgb_color)
    def __repr__(self) -> str:
        return f"Diamond(diagonal_h={self.diagonal_h}, diagonal_v={self.diagonal_v}, rgb_color={self.rgb_color})"
    def __str__(self) -> str:
        return f"Diamond of horizontal diagonal {self.diagonal_h}, vertical diagonal {self.diagonal_v} with color {self.rgb_name}"
    def get_image(self) -> npt.NDArray[np.uint8]:
        return self.image
    def get_diagonal_h(self) -> int:
        return self.diagonal_h
    def get_diagonal_v(self) -> int:
        return self.diagonal_v
    def get_rgb_color(self) -> tuple:
        return self.rgb_color
    def get_rgb_name(self) -> str:
        return self.rgb_name
    def get_area(self) -> float:
        """Calculate the area of the diamond."""
        return (self.diagonal_h * self.diagonal_v) / 2