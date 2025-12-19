import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from skimage.morphology import skeletonize
from skimage.util import img_as_bool
import html

# --------------------------------------------------
# Configuration
# --------------------------------------------------
Image.MAX_IMAGE_PIXELS = None


# Refined Color Map (HSV)
COLOR_MAP = {
    "FC":  {"lower": [0, 180, 100],   "upper": [5, 255, 255]},
    "PDT": {"lower": [22, 180, 100],  "upper": [32, 255, 255]},
    "PDI": {"lower": [45, 120, 100],  "upper": [75, 255, 255]},
    "ESD": {"lower": [105, 180, 100], "upper": [125, 255, 255]},
    "LIC": {"lower": [10, 180, 150],  "upper": [18, 255, 255]},
    "TANK": {"lower": [135, 100, 100], "upper": [155, 255, 255]},
}

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def is_valid_poc_shape(contour, symbol_name):
    x, y, w, h = cv2.boundingRect(contour)
    ar = w / float(h)
    if "TANK" in symbol_name:
        return h > 800 and ar < 0.6
    return 100 < w < 600 and 100 < h < 600 and 0.7 <= ar <= 1.3


def rdp(points, epsilon=8):
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = np.array(end) - np.array(start)
    line_len = np.linalg.norm(line)
    max_dist, idx = 0, 0

    for i in range(1, len(points) - 1):
        p = np.array(points[i])
        dist = np.linalg.norm(np.cross(line, p - np.array(start))) / (line_len or 1)
        if dist > max_dist:
            max_dist, idx = dist, i

    if max_dist > epsilon:
        left = rdp(points[:idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [start, end]

# --------------------------------------------------
# Centum XAML Writer
# --------------------------------------------------
class CentumXamlWriter:
    def __init__(self, canvas_width=2151, canvas_height=984, start_id=500):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.elements = []
        self.next_id = start_id

    def new_id(self):
        val = self.next_id
        self.next_id += 1
        return val

    def header(self):
        return f"""<Canvas Width="{self.canvas_width}" Height="{self.canvas_height}"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:rcsr="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Common.BrushExtension;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Common.iPCSBrushExtension"
        xmlns:yiapcspvgccbsc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls"
        xmlns:yiapcspvggn="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName"
        xmlns:yiapcspvgbdc0="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component">"""
        
    def add_element(self, x, y, w, h, fill, text):
        tagid = self.new_id()
        pts = f"0,0 {w},0 {w},{h} 0,{h}"
        self.elements.append(f'''
        <yiapcspvgccbsc:IPCSFillArea Tag="Id={tagid}" Fill="{fill}" Points="{pts}"
        Canvas.Left="{int(x)}" Canvas.Top="{int(y)}"
        Stroke="#FF000000" StrokeThickness="1" />''')

        txt_id = self.new_id()
        self.elements.append(f'''
        <yiapcspvgccbsc:Text Tag="Id={txt_id}" Canvas.Left="{int(x+w/4)}"
        Canvas.Top="{int(y+h/2.5)}" FontSize="10">{html.escape(text)}</yiapcspvgccbsc:Text>''')

    def add_polyline(self, points):
        tagid = self.new_id()

        # Absolute position
        min_x = min(x for x, y in points)
        min_y = min(y for x, y in points)

        # Relative geometry
        rel_pts = [(x - min_x, y - min_y) for x, y in points]
        pts_str = " ".join(f"{x},{y}" for x, y in rel_pts)

        self.elements.append(f'''
        <yiapcspvgccbsc:IPCSPolyLine
            StrokeThickness="1"
            Tag="Id={tagid}"
            Points="{pts_str}"
            Stroke="#FF000000"
            Canvas.Left="{min_x}"
            Canvas.Top="{min_y}"
            Panel.ZIndex="50">
            <yiapcspvggn:GenericNameComponent.GenericName>
                <yiapcspvggn:GenericName />
            </yiapcspvggn:GenericNameComponent.GenericName>
        </yiapcspvgccbsc:IPCSPolyLine>
        ''')

    def add_tank(self, x, y, w, h):
        tagid = self.new_id()

        # Draw a simple rectangle to represent the tank
        pts = f"0,0 {w},0 {w},{h} 0,{h}"

        self.elements.append(f'''
        <yiapcspvgccbsc:IPCSFillArea
            Tag="Id={tagid}"
            Fill="#FFB0C4DE"
            Points="{pts}"
            Canvas.Left="{int(x)}"
            Canvas.Top="{int(y)}"
            Stroke="#FF000000"
            StrokeThickness="2"
            Panel.ZIndex="30">
        </yiapcspvgccbsc:IPCSFillArea>
        ''')


    def to_file(self, filename="Main.xaml"):
        with open(filename, "w") as f:
            f.write(self.header())
            f.write("\n".join(self.elements))
            f.write("\n</Canvas>")

# --------------------------------------------------
# Main Workflow
# --------------------------------------------------
def run_extraction_workflow(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    writer = CentumXamlWriter()
    scale_x, scale_y = 0.153, 0.108
    symbol_count = {}
    polyline_count = 0

    # ---- Symbol Detection (unchanged)
    for name, bounds in COLOR_MAP.items():
        mask = cv2.inRange(hsv, np.array(bounds["lower"]), np.array(bounds["upper"]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if is_valid_poc_shape(cnt, name):
                x, y, w, h = cv2.boundingRect(cnt)

                sx, sy = x * scale_x, y * scale_y
                sw, sh = w * scale_x, h * scale_y

                if "TANK" in name:
                    writer.add_tank(sx, sy, sw, sh)
                else:
                    fill = "#FFD3D3D3"
                    if "LIC" in name:
                        fill = "#FFFFA500"

                    writer.add_element(sx, sy, sw, sh, fill, name)
                
                symbol_count[name] = symbol_count.get(name, 0) + 1
                print(f"[SYMBOL] {name} @ ({sx:.1f}, {sy:.1f}) size=({sw:.1f}x{sh:.1f})")



    # ---- Brown Annotation → IPCSPolyLine
    lower_brown = np.array([10, 120, 70])
    upper_brown = np.array([22, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # THEN merge strokes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # labels = cv2.connectedComponentsWithStats(mask, 8)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    (mask > 0).astype(np.uint8), connectivity=8
    )


    for label in range(1, labels.max() + 1):
        comp = (labels == label)
        print(f"[BROWN] Label {label}: area = {np.sum(comp)}")
        skel = skeletonize(comp)
        ys, xs = np.where(skel)
        pts = list(zip(xs, ys))
        print(f"[BROWN] Label {label}: skeleton points = {len(pts)}")
        if len(pts) < 5:
            continue

        pts.sort(key=lambda p: p[0])
        simplified = rdp(pts)
        print(f"[POLYLINE] Label {label}: simplified points = {len(simplified)}")

        scaled = [(int(x*scale_x), int(y*scale_y)) for x, y in simplified]
        writer.add_polyline(scaled)
        polyline_count += 1
        print(f"[POLYLINE] Written IPCSPolyLine #{polyline_count}")
    

    # Detailed Summary
    print("\n===== EXTRACTION SUMMARY =====")
    for k, v in symbol_count.items():
        print(f"{k}: {v}")
    print(f"Total IPCSPolyLines written: {polyline_count}")
    print("================================\n")

    writer.to_file()
    print("Main.xaml generated successfully.")

# --------------------------------------------------
run_extraction_workflow("subject3.pdf")
