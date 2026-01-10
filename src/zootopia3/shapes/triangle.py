import numpy as np
import numpy.typing as npt

def create_triangle_image(length: int, height: int, midpoint: float, rgb_color: tuple) -> npt.NDArray[np.uint8]:
    """
    Create a triangle with RGB color in a square pixel image.
    
    Args:
        length: The length of the triangle base in pixels.
        height: The height of the triangle in pixels.
        midpoint: The x-coordinate of the midpoint of the base (0 to 1).
        rgb_color: Tuple of (R, G, B) values (0-255)

    Returns:
        (max(length, height), max(length, height), 3) numpy array representing 
        the image with a colored triangle on black background.
    """
    # Create black background
    square_length = max(length, height)  
    image_array = np.zeros((square_length, square_length, 3), dtype=np.uint8)
    
    # Create coordinate grids
    y, x = np.ogrid[:square_length, :square_length]
    center_x = int(square_length * midpoint)
    
    # Create mask for triangle
    mask = (x >= center_x - length // 2) & (x <= center_x + length // 2) & (y <= height) & (y >= (height / (length / 2)) * np.abs(x - center_x))
    
    # Apply RGB color to triangle area
    image_array[mask] = rgb_color
    
    return image_array

class Triangle:
    def __init__(self, length: int, height: int, midpoint: float, rgb_color: tuple, rgb_name: str):
        self.length = length
        self.height = height
        self.midpoint = midpoint
        self.rgb_color = rgb_color
        self.rgb_name = rgb_name
        self.image = create_triangle_image(length, height, midpoint, rgb_color)
    def __repr__(self) -> str:
        return f"Triangle(length={self.length}, height={self.height}, midpoint={self.midpoint}, rgb_color={self.rgb_color}, rgb_name='{self.rgb_name}')"
    def __str__(self) -> str:
        return f"Triangle of length {self.length}, height {self.height}, midpoint {self.midpoint} with color {self.rgb_name} ({self.rgb_color})"
    def get_image(self) -> npt.NDArray[np.uint8]:
        return self.image
    def get_length(self) -> int:
        return self.length
    def get_height(self) -> int:
        return self.height
    def get_midpoint(self) -> float:
        return self.midpoint
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