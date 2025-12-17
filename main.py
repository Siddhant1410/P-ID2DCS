import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# 1. Configuration & Safety Limits
Image.MAX_IMAGE_PIXELS = None 

# 2. Refined Color Map (HSV)
# Adjust these slightly if your markers are a different shade
COLOR_MAP = {
    "FC":  {"lower": [0, 150, 50],   "upper": [10, 255, 255],  "color": "Red"},
    "PDT": {"lower": [20, 150, 50],  "upper": [35, 255, 255],  "color": "Yellow"},
    "PDI": {"lower": [40, 100, 50],  "upper": [80, 255, 255],  "color": "Green"},
    "ESD": {"lower": [100, 150, 50], "upper": [130, 255, 255], "color": "Blue"},
    "LIC": {"lower": [5, 150, 100],  "upper": [25, 255, 255],  "color": "Orange"}
}

def is_valid_rectangle(contour):
    """
    Applies geometric and dimensional filters to isolate POC markers.
    """
    # Approximate the shape
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Get bounding box dimensions
    x, y, w, h = cv2.boundingRect(contour)
    
    # --- GATE 1: Size Filter ---
    # At 300 DPI, your 5 marker boxes should be between 100 and 500 pixels.
    # This rejects the huge page frames (4000+ px) and tiny text fragments.
    if not (100 < w < 500 and 100 < h < 500):
        return False

    # --- GATE 2: Vertex Count ---
    # A rectangle must have exactly 4 corners after approximation.
    if len(approx) != 4:
        return False

    # --- GATE 3: Aspect Ratio ---
    # Markers are generally square-ish. This rejects long thin lines.
    aspect_ratio = float(w) / h
    if not (0.7 <= aspect_ratio <= 1.3):
        return False

    # --- GATE 4: Solidity ---
    # Compares the area of the contour to its bounding box to ensure it's "filled".
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False
    solidity = float(area) / hull_area
    if solidity < 0.8:
        return False

    return True

def extract_clean_pid_symbols(pdf_path):
    print(f"Opening {pdf_path} and converting to image...")
    # 300 DPI is required to match the pixel size logic in is_valid_rectangle
    pages = convert_from_path(pdf_path, dpi=300) 
    img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    detected_elements = []
    
    print(f"\n{'Symbol Type':<15} | {'Source X':<10} | {'Source Y':<10} | {'Box Size (WxH)'}")
    print("-" * 65)

    for symbol_name, bounds in COLOR_MAP.items():
        # Create mask for the specific color
        lower = np.array(bounds["lower"])
        upper = np.array(bounds["upper"])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphology: Close small gaps in the rectangle lines
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if is_valid_rectangle(cnt):
                x, y, w, h = cv2.boundingRect(cnt)
                
                print(f"{symbol_name:<15} | {x:<10} | {y:<10} | {w}x{h}")
                
                detected_elements.append({
                    "symbol": symbol_name,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })

    if len(detected_elements) == 0:
        print("No symbols detected. Check color thresholds or marker sizes.")
    else:
        print(f"\nFinal Count: {len(detected_elements)} symbols successfully extracted.")
        
    return detected_elements

# --- EXECUTION ---
symbols = extract_clean_pid_symbols('subject2.pdf')