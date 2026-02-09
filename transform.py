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
    def __init__(self, canvas_width=2151, canvas_height=984, start_id=100):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.elements = []
        self.next_id = start_id

    def new_id(self):
        val = self.next_id
        self.next_id += 1
        return val

    def footer (self):
        return f"""
        </Canvas>
    """

    def header(self):
        return f"""
    <!--PlatformRevisionProperties.PlatformFileRevision:0x0002-->
    <!--SolutionRevisionProperties.CENTUMFileRevision:0x0001-->
    <Canvas
    HorizontalAlignment="Center"
    VerticalAlignment="Center"
    Width="{self.canvas_width}"
    Height="{self.canvas_height}"
    Background="#FFC0C0C0"
    Tag="MaxId=99999"

    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:rcsr="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Common.BrushExtension;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Common.iPCSBrushExtension"
    xmlns:yiapcspvggn="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName"
    xmlns:yiapcspvgbdc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer"
    xmlns:yiapcspvgdl="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.DataLink;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.DataLink"
    xmlns:yiapcspvgflc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.FunctionLink.Control;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.FunctionLink.Control"
    xmlns:yiapcspvgbdc0="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component"
    xmlns:yiapcspvgccbsc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls"
    xmlns:yiapcspvgccdc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.DomainControls;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.DomainControls"
    xmlns:yiapcspvgdldm="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.DataLink.DataModel;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.DataLink.DataModel"
    xmlns:yiapcspvggnc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName.Common;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName"
    xmlns:yiapcspvgccd="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Common.Data;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Common"
    xmlns:yiapcspvgbmdg="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Module.Designer.Group;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Module"
    xmlns:s="clr-namespace:System;assembly=mscorlib,Version=4.0.0.0,Culture=neutral,PublicKeyToken=b77a5c561934e089"
    xmlns:yiapcspvgbd="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer"
    xmlns:yiapcspvgcp="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Common.Persistency;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Common.Persistency"
    xmlns:yiapcspvgbml="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Module.Layer;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Module.Core"
    xmlns:yiapcspvgflcwl="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.FunctionLink.Control.WindowLink;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.FunctionLink.Control">
  """
        
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
        line1_id = self.new_id()
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
            ArrowEndStyle="Triangle"
            ArrowSize="Small"
            Points="{pts_str}"
            Stroke="#FF000000"
            Canvas.Left="{min_x}"
            Canvas.Top="{min_y}"
            Panel.ZIndex="50"
            yiapcspvgbdc:ComponentProperties.Name="PolyLine"
            yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">
            <yiapcspvggn:GenericNameComponent.GenericName>
                <yiapcspvggn:GenericName />
            </yiapcspvggn:GenericNameComponent.GenericName>
        </yiapcspvgccbsc:IPCSPolyLine>
        ''')

    def add_esd(self, x, y, w=90.99, h=65.05):
        gid = self.new_id()
        rect_id = self.new_id()
        line_id = self.new_id()
        circ_id = self.new_id()
        poly_id = self.new_id()
        text_id = self.new_id()

        # ---- Relative ratios (derived from reference XAML)
        STEM_LEFT = 0.368
        STEM_TOP = 0.591
        STEM_W = 12 / 90.99
        STEM_H = 24 / 65.05

        LINE_LEFT = 0.50
        LINE_TOP = 0.345
        LINE_LEN = 9.5

        CIRCLE_LEFT = 0.393
        CIRCLE_TOP = 0.037
        CIRCLE_SIZE = 19.53 / 90.99

        POLY_LEFT = 0.324
        POLY_TOP = 0.391

        TEXT_LEFT = 0.58
        TEXT_TOP = 0.60

        self.elements.append(f'''
        <yiapcspvgbdc0:GroupComponent
            Tag="Id={gid}"
            Visibility="Visible"
            Width="{w}"
            Height="{h}"
            Canvas.Left="{x}"
            Canvas.Top="{y}"
            Panel.ZIndex="120"
            yiapcspvgbdc:ComponentProperties.Name="ESD_{gid}"
            yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">

            <!-- Stem -->
            <yiapcspvgccbsc:IPCSRectangle
                Rotation="-90"
                ShapeWidth="{w * STEM_W}"
                ShapeHeight="{h * STEM_H}"
                Fill="#FFC0C0C0"
                Stroke="{{x:Null}}"
                Canvas.Left="{w * STEM_LEFT}"
                Canvas.Top="{h * STEM_TOP}"
                Panel.ZIndex="121" />

            <!-- Stem line -->
            <yiapcspvgccbsc:IPCSLine
                StrokeThickness="2"
                Stroke="#FF666666"
                X1="0"
                X2="0"
                Y1="{LINE_LEN}"
                Y2="0"
                Canvas.Left="{w * LINE_LEFT}"
                Canvas.Top="{h * LINE_TOP}"
                Panel.ZIndex="122" />

            <!-- Motor circle -->
            <yiapcspvgccbsc:IPCSCircle
                ShapeWidth="{w * CIRCLE_SIZE}"
                ShapeHeight="{w * CIRCLE_SIZE}"
                Stroke="#FF000000"
                StrokeThickness="1"
                Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0011, Color1=DimGray, Color2=DarkGray}}"
                Canvas.Left="{w * CIRCLE_LEFT}"
                Canvas.Top="{h * CIRCLE_TOP}"
                Panel.ZIndex="123" />

            <!-- Valve body -->
            <yiapcspvgccbsc:IPCSFillArea
                Stroke="#FF000000"
                StrokeThickness="1"
                Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0004, Color1=DimGray, Color2=DarkGray}}"
                Points="0,0 0,14 32,0 32,14"
                Canvas.Left="{w * POLY_LEFT}"
                Canvas.Top="{h * POLY_TOP}"
                Panel.ZIndex="124" />

            <!-- % Text -->
            <yiapcspvgccbsc:Text
                FontSize="12"
                FontWeight="Bold"
                Foreground="#FF000000"
                Canvas.Left="{w * TEXT_LEFT}"
                Canvas.Top="{h * TEXT_TOP}"
                Panel.ZIndex="125">%</yiapcspvgccbsc:Text>

        </yiapcspvgbdc0:GroupComponent>
        ''')

    def clean_expr(expr_string):
        # Remove newlines and extra spaces
        return " ".join(expr_string.split())

    def add_tank(self, x, y, w, h):
        gid = self.new_id()
        rect_id = self.new_id()
        top_id = self.new_id()
        bot_id = self.new_id()

        # ---- Ratios from reference
        BODY_TOP_RATIO = 53.809057398134 / 478.999425128174
        BODY_HEIGHT_RATIO = 371.77270012602276 / 478.999425128174
        SEMI_HEIGHT_RATIO = 54.7875558080455 / 478.999425128174

        body_top = h * BODY_TOP_RATIO
        body_height = h * BODY_HEIGHT_RATIO

        semi_rx = w / 2
        semi_ry = h * SEMI_HEIGHT_RATIO

        # IMPORTANT: sector must be centered
        sector_left = 0

        self.elements.append(f'''
        <yiapcspvgbdc0:GroupComponent
            Tag="Id=191"
            Visibility="Visible"
            Width="{w}"
            Height="{h}"
            Canvas.Left="{x}"
            Canvas.Top="{y}"
            Panel.ZIndex="103"
            yiapcspvgbdc:ComponentProperties.Name="AutoTank_1234_{gid}"
            yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">

            <!-- Cylindrical body -->
            <yiapcspvgccbsc:IPCSRectangle
                Tag="Id=192"
                ShapeWidth="{w}"
                ShapeHeight="{body_height}"
                Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0003, Color1=#FF646464, Color2=Silver}}"
                Stroke="#00FFFFFF"
                Canvas.Left="0"
                Canvas.Top="{body_top}"
                Panel.ZIndex="112" />

            <!-- Top semicircle -->
            <yiapcspvgccbsc:IPCSSector
                Tag="Id=193;R50300"
                IsLargeArc="True"
                Size="{semi_rx},{semi_ry}"
                StartPoint="{w},{semi_ry}"
                EndPoint="0,{semi_ry}"
                SweepDirection="Counterclockwise"
                Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0003, Color1=#FF646464, Color2=Silver}}"
                Stroke="#00FFFFFF"
                Canvas.Left="{sector_left}"
                Canvas.Top="0"
                Panel.ZIndex="113" />

            <!-- Bottom semicircle -->
            <yiapcspvgccbsc:IPCSSector
            Tag="Id=194;R50300"
            IsLargeArc="True"
            Size="{semi_rx},{semi_ry}"
            StartPoint="0,0"
            EndPoint="{w},0"
            RenderTransform="1,0,0,-1,0,0"
            RenderTransformOrigin="0,0"
            SweepDirection="Clockwise"
            Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0003, Color1=#FF646464, Color2=Silver}}"
            Stroke="#00FFFFFF"
            Canvas.Left="0"
            Canvas.Top="{body_top + body_height}"
            Panel.ZIndex="114" />

        </yiapcspvgbdc0:GroupComponent>
    ''')

    def add_lic(self, cx, cy, ctr_tag, cv_tag):
        """
        Draws a Centum control valve (LIC / CV symbol)

        cx, cy  : center point of valve body (NOT top-left)
        ctr_tag : controller tag (e.g. 10PC0044)
        cv_tag  : valve tag (e.g. 10PV0044)
        """

        # -------------------------
        # Dimensions (from reference)
        # -------------------------
        BODY_W = 32
        BODY_H = 14

        STEM_W = 12
        STEM_H = 24

        ARROW_W = 8
        ARROW_H = 8

        FAN_RX = 16
        FAN_RY = 13.7033333333333

        # -------------------------
        # Derived positions
        # -------------------------
        body_left = cx - BODY_W / 2
        body_top  = cy - BODY_H / 2

        stem_left = cx - STEM_W / 2
        stem_top  = body_top + BODY_H

        arrow_left = cx - ARROW_W / 2
        arrow_top  = body_top - ARROW_H + 1

        fan_left = cx - FAN_RX
        fan_top  = body_top - FAN_RY - 2

        touch_left = cx - 20
        touch_top  = fan_top - 2

        z = 100

        self.elements.append(f"""
    <!-- ================= CONTROL VALVE ================= -->

    <!-- Stem -->
    <yiapcspvgccbsc:IPCSRectangle
        ShapeWidth="{STEM_W}"
        ShapeHeight="{STEM_H}"
        Fill="#FFC0C0C0"
        Stroke="{{x:Null}}"
        Canvas.Left="{stem_left}"
        Canvas.Top="{stem_top}"
        Panel.ZIndex="{z}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1" />

    <!-- Valve body -->
    <yiapcspvgccbsc:IPCSFillArea
        Points="0,0 0,{BODY_H} {BODY_W},0 {BODY_W},{BODY_H}"
        Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0004, Color1=DimGray, Color2=DarkGray}}"
        Stroke="#FF000000"
        Canvas.Left="{body_left}"
        Canvas.Top="{body_top}"
        Panel.ZIndex="{z+1}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName>
            <yiapcspvggn:GenericName GenericNames="">
                <yiapcspvggn:GenericName.BindingSets>
                    <yiapcspvggnc:BindingSetList>
                        <yiapcspvggn:BindingSet>
                            <yiapcspvggn:BindingSet.Bindings>
                                <yiapcspvggnc:GNBindingList>
                                    <yiapcspvggn:GNBinding GenericName="" />
                                    <yiapcspvggn:GNBinding GenericName="" />
                                </yiapcspvggnc:GNBindingList>
                            </yiapcspvggn:BindingSet.Bindings>
                        </yiapcspvggn:BindingSet>
                    </yiapcspvggnc:BindingSetList>
                </yiapcspvggn:GenericName.BindingSets>
            </yiapcspvggn:GenericName>
        </yiapcspvggn:GenericNameComponent.GenericName>
    </yiapcspvgccbsc:IPCSFillArea>

    <!-- Direction arrow -->
    <yiapcspvgccbsc:IPCSFillArea
        Points="4,8 0,0 8,0"
        Fill="#FF666666"
        Stroke="#00FFFFFF"
        Canvas.Left="{arrow_left}"
        Canvas.Top="{arrow_top}"
        Panel.ZIndex="{z+2}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1" />

    <!-- Actuator fan -->
    <yiapcspvgccbsc:IPCSSector
        StartPoint="{FAN_RX*2},{FAN_RY}"
        EndPoint="0,{FAN_RY}"
        Size="{FAN_RX},{FAN_RY}"
        SweepDirection="Counterclockwise"
        Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0004, Color1=DimGray, Color2=DarkGray}}"
        Stroke="#FF000000"
        Canvas.Left="{fan_left}"
        Canvas.Top="{fan_top}"
        Panel.ZIndex="{z+3}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1" />

    <!-- ================================================== -->
    """)

    def add_dvalve(self, x, y, w=90.99, h=50.1):
        gid = self.new_id()

        rect_id = self.new_id()
        vline_id = self.new_id()
        hline1_id = self.new_id()
        hline2_id = self.new_id()
        tri1_id = self.new_id()
        tri2_id = self.new_id()
        lock_id = self.new_id()

        # ---- Ratios derived from reference XAML
        STEM_LEFT = 35.495 / 90.99
        STEM_TOP  = 32.0 / 50.1
        STEM_W    = 12 / 90.99
        STEM_H    = 20 / 50.1

        VLINE_LEFT = 45.495 / 90.99
        VLINE_TOP  = 11.148 / 50.1
        VLINE_LEN  = 15.093

        HLINE_LEFT = 40.995 / 90.99
        HLINE_TOP1 = 19.242 / 50.1
        HLINE_TOP2 = 17.242 / 50.1
        HLINE_LEN  = 9

        TRI_LEFT = 41.495 / 90.99
        TRI_TOP  = 19.0 / 50.1

        LOCK_LEFT = 32.495 / 90.99
        LOCK_TOP  = 1.0 / 50.1
        LOCK_W    = 26 / 90.99
        LOCK_H    = 11 / 50.1

        self.elements.append(f'''
        <yiapcspvgbdc0:GroupComponent
            Tag="Id={gid}"
            Visibility="Visible"
            Width="{w}"
            Height="{h}"
            Canvas.Left="{x}"
            Canvas.Top="{y}"
            Panel.ZIndex="140"
            yiapcspvgbdc:ComponentProperties.Name="DVLV_{gid}"
            yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">

            <!-- Stem -->
            <yiapcspvgccbsc:IPCSRectangle
                Rotation="-90"
                ShapeWidth="{w * STEM_W}"
                ShapeHeight="{h * STEM_H}"
                Fill="#FFC0C0C0"
                Stroke="{{x:Null}}"
                Canvas.Left="{w * STEM_LEFT}"
                Canvas.Top="{h * STEM_TOP}"
                Panel.ZIndex="141" />

            <!-- Vertical line -->
            <yiapcspvgccbsc:IPCSLine
                StrokeThickness="2"
                Stroke="#FF666666"
                X1="0"
                X2="0"
                Y1="{VLINE_LEN}"
                Y2="0"
                Canvas.Left="{w * VLINE_LEFT}"
                Canvas.Top="{h * VLINE_TOP}"
                Panel.ZIndex="142" />

            <!-- Horizontal line (top) -->
            <yiapcspvgccbsc:IPCSLine
                StrokeThickness="1"
                Stroke="#FF666666"
                X1="0"
                X2="{HLINE_LEN}"
                Y1="0"
                Y2="0"
                Canvas.Left="{w * HLINE_LEFT}"
                Canvas.Top="{h * HLINE_TOP1}"
                Panel.ZIndex="143" />

            <!-- Horizontal line (bottom) -->
            <yiapcspvgccbsc:IPCSLine
                StrokeThickness="1"
                Stroke="#FF666666"
                X1="0"
                X2="{HLINE_LEN}"
                Y1="0"
                Y2="0"
                Canvas.Left="{w * HLINE_LEFT}"
                Canvas.Top="{h * HLINE_TOP2}"
                Panel.ZIndex="144" />

            <!-- Valve blade (triangle up) -->
            <yiapcspvgccbsc:IPCSFillArea
                Stroke="#00FFFFFF"
                Fill="#FF666666"
                Points="4,8 0,0 8,0"
                Canvas.Left="{w * TRI_LEFT}"
                Canvas.Top="{h * TRI_TOP}"
                Panel.ZIndex="145" />

            <!-- Valve blade (triangle down) -->
            <yiapcspvgccbsc:IPCSFillArea
                Stroke="#00FFFFFF"
                Fill="#FF666666"
                RenderTransform="1,0,0,-1,0,0"
                RenderTransformOrigin="0,0"
                Points="4,8 0,0 8,0"
                Canvas.Left="{w * TRI_LEFT}"
                Canvas.Top="{h * TRI_TOP}"
                Panel.ZIndex="146" />

            <!-- Lock / top block -->
            <yiapcspvgccbsc:IPCSFillArea
                Stroke="#FF000000"
                StrokeThickness="1"
                Fill="{{rcsr:iPCSBrushExtension Style=Gradient_0004, Color1=DimGray, Color2=DarkGray}}"
                Points="0,0 {w * LOCK_W},0 {w * LOCK_W},{h * LOCK_H} 0,{h * LOCK_H}"
                Canvas.Left="{w * LOCK_LEFT}"
                Canvas.Top="{h * LOCK_TOP}"
                Panel.ZIndex="147" />

        </yiapcspvgbdc0:GroupComponent>
        ''')

    def to_file(self, filename="Main.xaml"):
        with open(filename, "w") as f:
            f.write(self.header())
            f.write("\n".join(self.elements))
            f.write(self.footer())

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
                elif "ESD" in name:
                    writer.add_esd(sx, sy, sw, sh)
                elif "LIC" in name:
                    cx = sx + sw / 2
                    cy = sy + sh / 2
                    writer.add_lic(cx, cy, None, None)
                else:
                    writer.add_dvalve(sx, sy, sw, sh)
                    # fill = "#FFD3D3D3"
                    # if "LIC" in name:
                    #     fill = "#FFFFA500"

                    # writer.add_element(sx, sy, sw, sh, fill, name)
                
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
