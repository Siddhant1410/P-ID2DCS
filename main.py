import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import html

# 1. Configuration
Image.MAX_IMAGE_PIXELS = None

# 2. Refined Color Map (HSV)
COLOR_MAP = {
    "FC":  {"lower": [0, 180, 100],   "upper": [5, 255, 255]},   # Pure Red
    "PDT": {"lower": [22, 180, 100],  "upper": [32, 255, 255]},  # Pure Yellow
    "PDI": {"lower": [45, 120, 100],  "upper": [75, 255, 255]},  # Pure Green
    "ESD": {"lower": [105, 180, 100], "upper": [125, 255, 255]}, # Pure Blue
    "LIC": {"lower": [10, 180, 150],  "upper": [18, 255, 255]},  # Pure Orange
    "TANK": {"lower": [135, 100, 100], "upper": [155, 255, 255]}, # Pure Purple
}

def is_valid_poc_shape(contour, symbol_name):
    """Separates small instrument logic from large tank logic."""
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    if "TANK" in symbol_name:
        return h > 800 and aspect_ratio < 0.6
    else:
        if not (100 < w < 600 and 100 < h < 600):
            return False
        return 0.7 <= aspect_ratio <= 1.3

class CentumXamlWriter:
    """Uses Yokogawa-specific namespaces to avoid import errors."""
    def __init__(self, canvas_width=2151, canvas_height=984, start_id=500):
        self.canvas_width, self.canvas_height = canvas_width, canvas_height
        self.elements, self.next_id = [], start_id

    def new_id(self):
        val = self.next_id
        self.next_id += 1
        return val

    def header(self):
        return f"""<Canvas MaxHeight="10000" MaxWidth="10000" HorizontalAlignment="Center" VerticalAlignment="Center"
    Width="{self.canvas_width}" Height="{self.canvas_height}" Background="#FFC0C0C0" Tag="MaxId={self.next_id + 500}"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:yiapcspvgbdc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls"
    xmlns:yiapcspvgbdc0="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer">"""

    def add_element(self, x, y, w, h, fill, text):
        tagid = self.new_id()
        pts = f"0,0 {w},0 {w},{h} 0,{h}"
        el = f'''
    <yiapcspvgbdc:IPCSFillArea Tag="Id={tagid}" Fill="{fill}" Points="{pts}" Canvas.Left="{int(x)}" Canvas.Top="{int(y)}" 
        Stroke="#FF000000" StrokeThickness="1" yiapcspvgbdc0:ComponentProperties.Name="Sym_{tagid}" 
        yiapcspvgbdc0:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName><yiapcspvggn:GenericName /></yiapcspvggn:GenericNameComponent.GenericName>
    </yiapcspvgbdc:IPCSFillArea>'''
        txt_id = self.new_id()
        txt = f'''
    <yiapcspvgbdc:Text Tag="Id={txt_id}" Canvas.Left="{int(x+w/4)}" Canvas.Top="{int(y+h/2.5)}" FontSize="10" 
        Foreground="Black" yiapcspvgbdc0:ComponentProperties.LayerID="Normal Drawing Layer 1">{html.escape(text)}</yiapcspvgbdc:Text>'''
        self.elements.extend([el, txt])

    def add_line(self, x1, y1, x2, y2):
        tagid = self.new_id()
        el = f'''
    <yiapcspvgbdc:IPCSLine Tag="Id={tagid}" X1="0" Y1="0" X2="{int(x2-x1)}" Y2="{int(y2-y1)}" 
        Canvas.Left="{int(x1)}" Canvas.Top="{int(y1)}" Stroke="#FF000000" StrokeThickness="2" 
        yiapcspvgbdc0:ComponentProperties.Name="Line_{tagid}" yiapcspvgbdc0:ComponentProperties.LayerID="Normal Drawing Layer 1" />'''
        self.elements.append(el)

    def to_file(self, filename="Main.xaml"):
        with open(filename, "w") as f:
            f.write(self.header() + "\n".join(self.elements) + "</Canvas>")

# def detect_only_brown_lines(hsv, scale_x, scale_y):
#     """
#     Directly detects lines from the brown color mask.
#     This bypasses all black lines on the P&ID.
#     """
    # 1. Isolate Brown Reference Markers
    # brown_b = COLOR_MAP["LINE_REF"]
    # brown_mask = cv2.inRange(hsv, np.array(brown_b["lower"]), np.array(brown_b["upper"]))
    
    # 2. Apply Hough Line detection directly to the brown pixels
    # Parameters adjusted for strict detection of marker strokes
    # lines = cv2.HoughLinesP(brown_mask, 1, np.pi/180, threshold=50, minLineLength=80, maxLineGap=100)
    
    final_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Simple deduplication to ensure we don't draw multiple lines for one thick marker
            # if not any(np.abs(x1-fx1) < 20 and np.abs(y1-fy1) < 20 for fx1, fy1, fx2, fy2 in final_lines):
            #     final_lines.append((x1, y1, x2, y2))
    
    # return [(l[0]*scale_x, l[1]*scale_y, l[2]*scale_x, l[3]*scale_y) for l in final_lines]

def run_extraction_workflow(pdf_path):
    print(f"Opening {pdf_path} and converting to image...")
    pages = convert_from_path(pdf_path, dpi=300)
    img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    writer = CentumXamlWriter()
    scale_x, scale_y = 0.153, 0.108 
    
    # --- Brown Line Detection ---
    # print("-----------------------------------------")
    # print("Detecting brown marker lines...")
    # found_lines = detect_only_brown_lines(hsv, scale_x, scale_y)
    # for l in found_lines:
    #     writer.add_line(l[0], l[1], l[2], l[3])
    # print(f"Number of brown lines detected: {len(found_lines)}")
    # print("-----------------------------------------")

    # --- Symbol Detection (Unchanged) ---
    for symbol_name, bounds in COLOR_MAP.items():
        if symbol_name == "LINE_REF": continue
        mask = cv2.inRange(hsv, np.array(bounds["lower"]), np.array(bounds["upper"]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if is_valid_poc_shape(cnt, symbol_name):
                x, y, w, h = cv2.boundingRect(cnt)
                print(f"Detected Symbol: {symbol_name} at X:{x}, Y:{y}")
                fill = "#FFFFA500" if "LIC" in symbol_name else "#FFD3D3D3"
                if "TANK" in symbol_name: fill = "#FFB0C4DE"
                writer.add_element(x*scale_x, y*scale_y, w*scale_x, h*scale_y, fill, symbol_name)

    writer.to_file()
    print("-----------------------------------------")
    print("Main.xaml generated successfully.")

run_extraction_workflow('subject2.pdf')