import numpy as np
import numpy.typing as npt

def create_triangle_image(length: int, 
                            height: int, 
                            rgb_color: tuple,
                            upside_down: bool = False,
                            sideways: bool = False, 
                            ) -> npt.NDArray[np.uint8]:
    """
    Create a triangle with RGB color in a square pixel image.
    
    Args:
        length: The length of the triangle base in pixels.
        height: The height of the triangle in pixels.
        midpoint: The x-coordinate of the midpoint of the base (0 to 1).
        upside_down: Whether the triangle is upside down.
        sideways: Whether the triangle is sideways.
        rgb_color: Tuple of (R, G, B) values (0-255)

    Returns:
        (max(length, height), max(length, height), 3) numpy array representing 
        the image with a colored triangle on black background.
    """
    # Create black background
    square_length = max(length, height)  
    image_array = np.random.randint(0, 256, (square_length, square_length, 3), dtype=np.uint8)
    
    # Create coordinate grids
    midpoint = 0.5
    y, x = np.ogrid[:square_length, :square_length]
    center_x = int(square_length * midpoint)
    
    # Create mask for triangle
    mask = (x >= center_x - length // 2) & (x <= center_x + length // 2) & (y <= height) & (y >= (height / (length / 2)) * np.abs(x - center_x))
    if upside_down:
        mask = (x >= center_x - length // 2) & (x <= center_x + length // 2) & (y >= 0) & (y <= height - (height / (length / 2)) * np.abs(x - center_x))
    if sideways:
        mask = (y >= center_x - length // 2) & (y <= center_x + length // 2) & (x <= height) & (x >= (height / (length / 2)) * np.abs(y - center_x))
        if upside_down:
            mask = (y >= center_x - length // 2) & (y <= center_x + length // 2) & (x >= 0) & (x <= height - (height / (length / 2)) * np.abs(y - center_x))
    
    # Apply RGB color to triangle area
    image_array[mask] = rgb_color
    
    return image_array

class Triangle:
    '''Class representing a triangle shape with various properties and methods.'''
    def __init__(self, 
                    length: int, 
                    height: int, 
                    rgb_color: tuple, 
                    rgb_name: str,
                    upside_down: bool = False,
                    sideways: bool = False, 
                    ) -> None:
        '''Initialize a Triangle instance.
        
        Args:
            length: The length of the triangle base in pixels.
            height: The height of the triangle in pixels.
            midpoint: The x-coordinate of the midpoint of the base (0 to 1).
            rgb_color: Tuple of (R, G, B) values (0-255).
            rgb_name: Name of the RGB color.
            upside_down: Whether the triangle is upside down.
            sideways: Whether the triangle is sideways.
        '''
        self.length = length
        self.height = height
        self.rgb_color = rgb_color
        self.rgb_name = rgb_name
        self.image = create_triangle_image(length, height, rgb_color, upside_down, sideways)
    def __repr__(self) -> str:
        return f"Triangle(length={self.length}, height={self.height}, rgb_color={self.rgb_color})"
    def __str__(self) -> str:
        return f"Triangle of length {self.length}, height {self.height} with color {self.rgb_name}"
    def get_image(self) -> npt.NDArray[np.uint8]:
        return self.image
    def get_length(self) -> int:
        return self.length
    def get_height(self) -> int:
        return self.height
    def get_rgb_color(self) -> tuple:
        return self.rgb_color
    def get_rgb_name(self) -> str:
        return self.rgb_name
    def get_area(self) -> float:
        """Calculate the area of the triangle."""
        return 0.5 * self.length * self.height
    def is_equilateral(self) -> bool:
        """Check if the triangle is equilateral."""
        return self.length == self.height and self.midpoint == 0.5
    def is_isosceles(self) -> bool:
        """Check if the triangle is isosceles."""
        return self.midpoint == 0.5
    def is_scalene(self) -> bool:
        """Check if the triangle is scalene."""
        return not self.is_isosceles()