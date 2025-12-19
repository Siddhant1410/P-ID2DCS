import cv2
import numpy as np
from pdf2image import convert_from_path

def get_pdf_image_specs(pdf_path, dpi=300):
    """
    Converts the first page of a PDF to an image and returns its 
    pixel dimensions to calculate scaling factors.
    """
    print(f"Converting {pdf_path} at {dpi} DPI...")
    
    # Use the same DPI you use in your main extraction script
    pages = convert_from_path(pdf_path, dpi=dpi)
    
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    
    # Get dimensions
    height, width, channels = img.shape
    
    print("-" * 30)
    print(f"RESULTS FOR: {pdf_path}")
    print(f"Width:  {width} pixels")
    print(f"Height: {height} pixels")
    print("-" * 30)
    
    # Calculate recommended scaling factors for DCS (2151 x 984)
    target_w, target_h = 2151, 984
    scale_x = target_w / width
    scale_y = target_h / height
    
    print(f"Recommended Scale X: {scale_x:.6f}")
    print(f"Recommended Scale Y: {scale_y:.6f}")
    
    return width, height, scale_x, scale_y

# Execute for your specific file
if __name__ == "__main__":
    # Ensure 'subject2.pdf' is in your directory 
    get_pdf_image_specs('subject2.pdf', dpi=300)