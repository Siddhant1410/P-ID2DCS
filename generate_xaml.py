import html

def xaml_escape(text):
    return html.escape(text)

class CentumXamlWriter:
    def __init__(self, canvas_width: int, canvas_height: int, start_id: int = 500):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.elements = []
        self.next_id = start_id

    def new_id(self):
        val = self.next_id
        self.next_id += 1
        return val

    def header(self):
        # NOTE: yiapcspvgbdc points to BasicShapeControls (Visuals)
        # NOTE: yiapcspvgbdc0 points to Builder.Designer (Metadata/Properties)
        return f"""<Canvas
    MaxHeight="10000" MaxWidth="10000" HorizontalAlignment="Center" VerticalAlignment="Center"
    Width="{self.canvas_width}" Height="{self.canvas_height}" Background="#FFC0C0C0" Tag="MaxId={self.next_id + 100}"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:yiapcspvgbdc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls"
    xmlns:yiapcspvgbdc0="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer"
    xmlns:yiapcspvggn="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName"
    xmlns:yiapcspvgcp="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Common.Persistency;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Common.Persistency">"""

    def footer(self):
        return "</Canvas>"

    def add_text(self, x:int, y:int, text:str, font_size:int=14, color:str="#FF000000", bold:bool=False):
        tagid = self.new_id()
        esc = xaml_escape(text)
        weight = "Bold" if bold else "Normal"
        # We use yiapcspvgbdc for the Text control, but yiapcspvgbdc0 for the Properties
        el = f'''
    <yiapcspvgbdc:Text FontSize="{font_size}" FontWeight="{weight}" Visibility="Visible" Focusable="False" 
        FontFamily="Arial" Foreground="{color}" Tag="Id={tagid}" Canvas.Left="{int(x)}" Canvas.Top="{int(y)}" 
        yiapcspvgbdc0:ComponentProperties.Name="Text_{tagid}" 
        yiapcspvgbdc0:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName>
            <yiapcspvggn:GenericName />
        </yiapcspvggn:GenericNameComponent.GenericName>{esc}</yiapcspvgbdc:Text>'''
        self.elements.append(el)

    def add_rect(self, x:int, y:int, w:int, h:int, fill:str="#FFFFFFFF", stroke:str="#FF000000", name:str="Rect"):
        tagid = self.new_id()
        pts = f"0,0 {w},0 {w},{h} 0,{h}"
        # We use yiapcspvgbdc for the FillArea control, but yiapcspvgbdc0 for the Properties
        el = f'''
    <yiapcspvgbdc:IPCSFillArea Visibility="Visible" Focusable="False" Stroke="{stroke}" StrokeThickness="1" 
        Tag="Id={tagid}" Fill="{fill}" Points="{pts}" Canvas.Left="{int(x)}" Canvas.Top="{int(y)}" 
        yiapcspvgbdc0:ComponentProperties.Name="{name}_{tagid}" 
        yiapcspvgbdc0:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName>
            <yiapcspvggn:GenericName />
        </yiapcspvggn:GenericNameComponent.GenericName>
    </yiapcspvgbdc:IPCSFillArea>'''
        self.elements.append(el)

    def to_string(self):
        return "\n".join([self.header()] + self.elements + [self.footer()])

# --- POC COORDINATE DATA & GENERATION ---

def generate_main_xaml():
    # Canonical Canvas Dimensions for Yokogawa Graphics
    writer = CentumXamlWriter(canvas_width=2151, canvas_height=984)
    
    # Scaling logic (300DPI pixel coords -> 2151x984 Canvas space)
    scale_x, scale_y = 0.153, 0.108

    # Your verified detection results
    data = [
        {"type": "FC",  "x": 8161, "y": 1954, "color": "#FFFF0000"}, 
        {"type": "PDT", "x": 6164, "y": 1446, "color": "#FFFFFFFF"}, 
        {"type": "PDI", "x": 6137, "y": 1081, "color": "#FFFFFF00"}, 
        {"type": "ESD", "x": 8193, "y": 1577, "color": "#FFFF0000"}, 
        {"type": "LIC", "x": 6164, "y": 1446, "color": "#FFFFA500"}, 
    ]

    for item in data:
        cx = item["x"] * scale_x
        cy = item["y"] * scale_y
        label = item["type"]

        if label == "FC":
            writer.add_text(cx, cy, "FC", font_size=16, color="#FFFF0000", bold=True)
        
        elif label == "ESD":
            writer.add_rect(cx, cy, 60, 40, fill="#FFFF0000", stroke="#FF000000", name="ESD_Box")
            writer.add_text(cx+10, cy+10, "ESD", font_size=12, color="#FFFFFFFF")
            
        elif label == "LIC":
            # Offset slightly from PDT if they share coordinates
            writer.add_rect(cx+20, cy+20, 70, 50, fill="#FFFFA500", name="LIC_Controller")
            writer.add_text(cx+30, cy+30, "LIC-3682", font_size=10)

        elif label == "PDI":
            writer.add_rect(cx, cy, 70, 50, fill="#FFFFFF00", name="PDI_Indicator")
            writer.add_text(cx+10, cy+10, "PDI-3682", font_size=10)

        elif label == "PDT":
            writer.add_rect(cx, cy, 50, 50, fill="#FFFFFFFF", stroke="#FF000000", name="PDT_Transmitter")
            writer.add_text(cx+5, cy+15, "PDT-3682", font_size=9)

    with open("Main.xaml", "w", encoding="utf-8") as f:
        f.write(writer.to_string())
    print("Main.xaml successfully generated with fixed namespace-property mapping.")

if __name__ == "__main__":
    generate_main_xaml()