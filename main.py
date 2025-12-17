import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Image.MAX_IMAGE_PIXELS = None 

def extract_multiple_red_rois(pdf_path):
    print("Converting PDF and searching for ALL red POC markers...")
    pages = convert_from_path(pdf_path, dpi=300) 
    img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    
    # 1. Detect Red Color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # P&IDs often have "dirty" whites; these bounds capture bright red strictly
    lower_red1 = np.array([0, 150, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 2. Find ALL contours (not just the max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Detected {len(contours)} potential marker areas.")
    
    results_list = []

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 500: # Filter out tiny red noise/dots
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        # Crop with a tiny margin to ensure we don't cut off the text edges
        margin = 5
        roi = img[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin]
        
        # 3. Enhanced Pre-processing for the crop
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Increase contrast for thin engineering fonts
        gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 4. OCR on the specific Box
        # --psm 11 tries to find as much sparse text as possible
        custom_config = r'--oem 3 --psm 11' 
        text_data = pytesseract.image_to_string(gray_roi, config=custom_config).strip()
        
        if text_data:
            clean_text = text_data.replace('\n', ' ')
            print(f"Box {i+1} at [X:{x}, Y:{y}] contains: {clean_text}")
            results_list.append({"tag": clean_text, "x": x, "y": y})

    return results_list

results = extract_multiple_red_rois('subject2.pdf')