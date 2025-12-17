#!/usr/bin/env python3
"""
pid_to_xaml_centum_final_v2.py

Final, single-file CENTUM / iPCS-style XAML generator with enhanced preprocessing:
 - Multi-page PDF -> images (PyMuPDF)
 - Enhanced preprocessing for OCR and detection (upscale, denoise, CLAHE, sharpen)
 - Graphic-line removal from OCR layer (Hough + morphological)
 - Line detection (HoughP) with thickness/style estimation
 - Curve detection (approx)
 - Arrowhead detection (triangle)
 - Symbol candidate detection (contours) with sub-element exclusion
 - OCR (optional, pytesseract) on cleaned text layer (multi-angle)
 - Build merged graph (snap endpoints, split at intersections)
 - Attach symbols and OCR tags to graph nodes
 - Generate Yokogawa CENTUM / iPCS-style XAML with proper tags
 - Outputs debug images, summary, graph JSONs, and CENTUM XAML

Usage:
    python pid_to_xaml_centum_final_v2.py --pdf input.pdf --out_dir output --ocr --denoise --deskew

Dependencies (pip):
    pip install pymupdf opencv-python numpy pillow networkx pytesseract

Author: generated for your workflow
"""
import os
import math
import json
import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# third-party imports
try:
    import fitz  # pymupdf
except Exception:
    raise SystemExit("Missing dependency 'pymupdf'. Install: pip install pymupdf")
try:
    import cv2
except Exception:
    raise SystemExit("Missing dependency 'opencv-python'. Install: pip install opencv-python")
import numpy as np
from PIL import Image

try:
    import networkx as nx
except Exception:
    raise SystemExit("Missing dependency 'networkx'. Install: pip install networkx")

# optional OCR
try:
    import pytesseract
    PYSTESSERACT = True
except Exception:
    PYSTESSERACT = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------ Data classes ------------------
@dataclass
class LineSegment:
    x1: int; y1: int; x2: int; y2: int; page: int
    thickness: float = 1.0; style: str = 'solid'; color: Tuple[int,int,int] = (0,0,0)
    is_curve: bool = False; id: Optional[str] = None

@dataclass
class SymbolCandidate:
    bbox: Tuple[int,int,int,int]
    centroid: Tuple[int,int]
    page: int
    area: float
    cls: Optional[str] = None
    score: float = 0.0

@dataclass
class Arrow:
    x: int; y: int; dx: int; dy: int; page: int; direction: str

# ------------------ Utilities ------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def debug_write_image(img: np.ndarray, path: str):
    try:
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(path)
    except Exception:
        cv2.imwrite(path, img)

def color_to_centum_hex(bgr: Tuple[int,int,int]) -> str:
    # Convert BGR -> #FFRRGGBB (Centum uses #FFRRGGBB)
    b,g,r = bgr
    return f"#FF{int(r):02X}{int(g):02X}{int(b):02X}"

def xaml_escape(s):
    if s is None:
        return ""
    return (str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))

# ------------------ PDF -> Images ------------------
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Tuple[int, np.ndarray]]:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        if pix.n >= 3:
            img = arr.reshape((pix.height, pix.width, pix.n))[:, :, :3]
        else:
            img = arr.reshape((pix.height, pix.width))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pages.append((i+1, img))
    return pages

# ------------------ ENHANCED PREPROCESS FOR OCR & DETECTION ------------------
def enhance_page_for_ocr(img: np.ndarray,
                         scale_up: float = 1.4,
                         denoise_strength: int = 7,
                         clahe_clip: float = 3.0,
                         binarize_patch: bool = True,
                         remove_graphic_lines: bool = True) -> np.ndarray:
    """
    Improve image for both detection and OCR:
     - optional upscale (improves OCR on small text)
     - denoise (fastNlMeans or bilateral fallback)
     - CLAHE (contrast) + unsharp mask
     - adaptive binarization to generate mask
     - morphological cleanup and small-object removal
     - optional removal of long straight graphic lines (Hough) to produce a text-friendly image
     - inpaint removed regions to allow OCR continuity
    Returns enhanced (BGR) image suitable for both visual processing and OCR.
    """
    img0 = img.copy()

    # 1) upscale slightly (avoid too large to keep performance)
    if scale_up != 1.0:
        h, w = img0.shape[:2]
        new_w = max(1, int(w * scale_up))
        new_h = max(1, int(h * scale_up))
        img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2) denoise: try fastNlMeansColored, fallback bilateral filter
    try:
        img0 = cv2.fastNlMeansDenoisingColored(img0, None, denoise_strength, denoise_strength, 7, 21)
    except Exception:
        img0 = cv2.bilateralFilter(img0, 5, 75, 75)

    # 3) convert to gray and CLAHE
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # 4) Unsharp mask (sharpen)
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # 5) Adaptive threshold (binarize) to get mask for small object removal
    if binarize_patch:
        th = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 9)
    else:
        _, th = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6) Morphological open to remove small speckles, then close to join broken letters
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_small, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # 7) Remove very small connected components (noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - th, connectivity=8)
    min_area_keep = 20  # small glyphs below this are likely noise; tune as needed
    mask_keep = np.zeros_like(th)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_keep:
            mask_keep[labels == i] = 255
    # mask_keep is the foreground (text-like) mask

    # 8) Optionally remove long graphic lines that interfere with OCR:
    if remove_graphic_lines:
        edges = cv2.Canny(sharpen, 50, 150, apertureSize=3)
        kernel = np.ones((3,3), np.uint8)
        edges_d = cv2.dilate(edges, kernel, iterations=1)
        # Probabilistic Hough: threshold tuned for drawings; may need adjustment
        lines = cv2.HoughLinesP(edges_d, rho=1, theta=np.pi/180, threshold=120, minLineLength=100, maxLineGap=10)
        line_mask = np.zeros_like(gray)
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                # draw slightly thick lines into mask to remove from text layer
                thickness = max(2, int(round((abs(x2-x1)+abs(y2-y1))/300)))
                cv2.line(line_mask, (x1,y1), (x2,y2), 255, thickness)
        # remove detected line pixels from mask_keep
        mask_keep[line_mask > 0] = 0

    # 9) Inpaint the removed regions on the grayscale image to make characters contiguous
    inpaint_mask = (255 - mask_keep).astype(np.uint8)
    inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    sharpen_bgr = cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR)
    try:
        inpainted = cv2.inpaint(sharpen_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        inpainted = sharpen_bgr

    # 10) Final contrast stretch and return BGR
    final_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    p2, p98 = np.percentile(final_gray, (2,98))
    if p98 - p2 > 0:
        final_gray = np.clip((final_gray - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    final_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
    return final_bgr

# ------------------ Thinning (fallback skeletonization) ------------------
def thin_edges(edges: np.ndarray) -> np.ndarray:
    # simple morphological skeletonization fallback
    size = np.size(edges)
    skel = np.zeros(edges.shape, np.uint8)
    ret, img = cv2.threshold(edges, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

# ------------------ Line detection ------------------
def detect_line_segments(img: np.ndarray, min_length=30, max_gap=8, hough_threshold=None, page_idx=1) -> List[LineSegment]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    try:
        gray = clahe.apply(gray)
    except Exception:
        pass
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = thin_edges(edges)
    if hough_threshold is None:
        hough_threshold = max(40, int(np.clip(gray.mean(), 40, 200)))
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=hough_threshold, minLineLength=min_length, maxLineGap=max_gap)
    segments = []
    if lines is None:
        return segments
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(255 - bw, distanceType=cv2.DIST_L2, maskSize=3)
    for i, l in enumerate(lines):
        x1,y1,x2,y2 = map(int, l[0])
        sample_points = 7
        ts = []
        for s in range(sample_points):
            t = s/(sample_points-1) if sample_points>1 else 0.5
            sx = int(round(x1*(1-t) + x2*t)); sy = int(round(y1*(1-t) + y2*t))
            if 0 <= sy < dist.shape[0] and 0 <= sx < dist.shape[1]:
                ts.append(dist[sy, sx])
        thickness = float(max(1.0, np.median(ts) if ts else 1.0))
        mx = max(1, min(img.shape[1]-2, int((x1+x2)/2)))
        my = max(1, min(img.shape[0]-2, int((y1+y2)/2)))
        bgr = img[my-1:my+2, mx-1:mx+2].reshape(-1,3).mean(axis=0)
        color = tuple(int(x) for x in bgr.tolist())
        dotted_score = sample_gap_score(edges, (x1,y1,x2,y2))
        if dotted_score > 0.45:
            style = 'dotted'
        elif dotted_score > 0.18:
            style = 'dashed'
        else:
            style = 'solid'
        seg = LineSegment(x1,y1,x2,y2,page=page_idx,thickness=thickness,style=style,color=color,is_curve=False,id=f'p{page_idx}_l{i}')
        segments.append(seg)
    return segments

def sample_gap_score(edges, seg):
    x1,y1,x2,y2 = seg
    length = int(math.hypot(x2-x1, y2-y1))
    if length < 2:
        return 0.0
    N = max(50, min(400, length))
    pts = [(int(round(x1 + (x2-x1)*t)), int(round(y1 + (y2-y1)*t))) for t in np.linspace(0,1,N)]
    hits = []
    h,w = edges.shape
    for (x,y) in pts:
        if x<0 or x>=w or y<0 or y>=h:
            hits.append(0)
        else:
            hits.append(1 if edges[y,x]>0 else 0)
    hits = np.array(hits)
    frac_zero = float((hits==0).sum()) / len(hits)
    transitions = np.count_nonzero(hits[1:] != hits[:-1]) / len(hits)
    score = transitions * frac_zero
    return score

# ------------------ Curve detection ------------------
def detect_curves(img: np.ndarray, min_area=120) -> List[List[Tuple[int,int]]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    curves = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        approx = cv2.approxPolyDP(c, epsilon=2.0, closed=False)
        if len(approx) > 10:
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            curves.append(pts)
    return curves

# ------------------ Arrow detection ------------------
def detect_arrowheads(img: np.ndarray, min_area=20, max_area=2000, page_idx=1) -> List[Arrow]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2,2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrows = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            pts = [tuple(p[0]) for p in approx]
            cx = sum(p[0] for p in pts)/3
            cy = sum(p[1] for p in pts)/3
            dists = [math.hypot(p[0]-cx, p[1]-cy) for p in pts]
            tip_idx = int(np.argmax(dists))
            tip = pts[tip_idx]
            dx = tip[0] - cx
            dy = tip[1] - cy
            dirn = 'right' if dx > 0 else 'left'
            arrows.append(Arrow(int(cx), int(cy), int(round(dx)), int(round(dy)), page=page_idx, direction=dirn))
    return arrows

# ------------------ Symbol detection (with sub-element exclusion) ------------------
def detect_symbol_candidates(img: np.ndarray, min_area=200, max_area=50000, page_idx=1) -> List[SymbolCandidate]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    syms: List[SymbolCandidate] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cx = x + w//2; cy = y + h//2
        syms.append(SymbolCandidate((x,y,w,h),(cx,cy),page=page_idx,area=area))

    # Remove candidates fully contained in another larger bbox (sub-elements)
    filtered: List[SymbolCandidate] = []
    for s in syms:
        x,y,w,h = s.bbox
        keep = True
        for t in syms:
            if s is t:
                continue
            x2,y2,w2,h2 = t.bbox
            if (x2 <= x) and (y2 <= y) and ((x + w) <= (x2 + w2)) and ((y + h) <= (y2 + h2)):
                if t.area > s.area:
                    keep = False
                    break
        if keep:
            filtered.append(s)
    return filtered

# ------------------ Template classifier (optional) ------------------
def load_templates(template_dir: Optional[str]) -> Dict[str, np.ndarray]:
    templates = {}
    if not template_dir:
        return templates
    if not os.path.isdir(template_dir):
        return templates
    for fname in os.listdir(template_dir):
        if not fname.lower().endswith(('.png','.jpg','.bmp')):
            continue
        key = os.path.splitext(fname)[0]
        img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[key] = img
    return templates

def classify_symbols_by_template(img: np.ndarray, syms: List[SymbolCandidate], templates: Dict[str,np.ndarray]) -> List[SymbolCandidate]:
    if not templates:
        return syms
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for s in syms:
        x,y,w,h = s.bbox
        roi = gray[y:y+h, x:x+w] if h>0 and w>0 else None
        best_cls = None; best_score = 0.0
        if roi is not None and roi.size>0:
            for name, tmpl in templates.items():
                try:
                    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, maxval, _, _ = cv2.minMaxLoc(res)
                except Exception:
                    maxval = 0.0
                if maxval > best_score:
                    best_score = float(maxval); best_cls = name
        s.cls = best_cls; s.score = best_score
        out.append(s)
    return out

# ------------------ OCR (text-layer) ------------------
def ocr_text_layer(img: np.ndarray, page_idx: int = 1, angles=(0, -90, 90), psm=6) -> List[Dict]:
    """
    Given a cleaned/enhanced BGR image, produce OCR tags:
    - create a text-focused binary using morphology to exclude long lines
    - optional deskew via minAreaRect on text pixels
    - multi-angle pytesseract passes
    Returns list of {"text","bbox":[x,y,w,h],"confidence":float,"page":page_idx}
    """
    tags: List[Dict] = []
    if not PYSTESSERACT:
        return tags

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 9)
    # detect long horizontal and vertical strokes
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    hlines = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vlines = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    lines_mask = cv2.bitwise_or(hlines, vlines)
    mask_no_lines = cv2.bitwise_and(th, cv2.bitwise_not(lines_mask))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_no_lines = cv2.morphologyEx(mask_no_lines, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask_no_lines = cv2.morphologyEx(mask_no_lines, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    coords = cv2.findNonZero(255 - mask_no_lines) if np.count_nonzero(255 - mask_no_lines) > 0 else None
    if coords is not None:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = angle + 90
        if abs(angle) > 0.3:
            h_img,w_img = gray.shape
            M = cv2.getRotationMatrix2D((w_img/2,h_img/2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w_img,h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mask_no_lines = cv2.warpAffine(mask_no_lines, M, (w_img,h_img), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    text_focused = cv2.bitwise_and(gray, gray, mask=255 - mask_no_lines)
    _, ocr_bin = cv2.threshold(text_focused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    seen = set()
    for angle in angles:
        if angle != 0:
            h_img,w_img = ocr_bin.shape
            M = cv2.getRotationMatrix2D((w_img/2,h_img/2), angle, 1.0)
            img_rot = cv2.warpAffine(ocr_bin, M, (w_img,h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            img_rot = ocr_bin
            M = None
        cfg = f'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:/.() --psm {psm}'
        try:
            data = pytesseract.image_to_data(img_rot, config=cfg, output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        n = len(data.get('text', []))
        for i in range(n):
            txt = str(data['text'][i]).strip()
            if txt == '':
                continue
            conf_raw = data.get('conf', [None]*n)[i]
            try:
                conf = float(conf_raw)
            except Exception:
                try:
                    conf = float(str(conf_raw))
                except Exception:
                    conf = 0.0
            x = int(data.get('left', [0]*n)[i]); y = int(data.get('top', [0]*n)[i]); w = int(data.get('width', [0]*n)[i]); h = int(data.get('height', [0]*n)[i])
            if M is not None:
                invM = cv2.invertAffineTransform(M)
                cx = x + w/2; cy = y + h/2
                px,py = invM.dot([cx,cy,1])
                x = int(px - w/2); y = int(py - h/2)
            key = (txt, x, y, w, h)
            if key in seen:
                continue
            seen.add(key)
            tags.append({"text": txt, "bbox": [x, y, w, h], "confidence": conf, "page": page_idx})
    return tags

# ------------------ Graph builder ------------------
def segment_intersection(seg1, seg2):
    x1,y1,x2,y2 = seg1
    x3,y3,x4,y4 = seg2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/denom
    eps = 1e-6
    if (min(x1,x2)-eps <= px <= max(x1,x2)+eps and min(y1,y2)-eps <= py <= max(y1,y2)+eps and
        min(x3,x4)-eps <= px <= max(x3,x4)+eps and min(y3,y4)-eps <= py <= max(y3,y4)+eps):
        return (px,py)
    return None

def build_merged_graph(all_segments: List[List[LineSegment]], page_images: List[Tuple[int,np.ndarray]], page_order_positions: List[Tuple[int,int]]=None, snap_tol: int=10):
    n_pages = len(page_images)
    widths = [img.shape[1] for (_,img) in page_images]
    heights = [img.shape[0] for (_,img) in page_images]
    offsets = []
    if page_order_positions:
        offsets = page_order_positions
    else:
        if n_pages == 3:
            offsets = [(0,0),(widths[0],0),(widths[0]+widths[1],0)]
        else:
            x = 0
            for w in widths:
                offsets.append((x,0)); x += w
    global_segments: List[LineSegment] = []
    for p_idx,segs in enumerate(all_segments):
        ox,oy = offsets[p_idx]
        for s in segs:
            gs = LineSegment(s.x1+ox, s.y1+oy, s.x2+ox, s.y2+oy, page=s.page, thickness=s.thickness, style=s.style, color=s.color, is_curve=s.is_curve, id=s.id)
            global_segments.append(gs)
    G = nx.Graph()
    nodes_positions: List[Tuple[int,int]] = []
    def snap_or_add_node(x,y):
        for i,(nxp,nyp) in enumerate(nodes_positions):
            if abs(nxp-x) <= snap_tol and abs(nyp-y) <= snap_tol:
                return f"N{i}"
        i = len(nodes_positions)
        nodes_positions.append((x,y))
        G.add_node(f"N{i}", xy=(x,y))
        return f"N{i}"
    # add edges from endpoints
    for seg in global_segments:
        n1 = snap_or_add_node(seg.x1, seg.y1)
        n2 = snap_or_add_node(seg.x2, seg.y2)
        if G.has_edge(n1,n2):
            existing = G[n1][n2].get('seg_ids', [])
            existing.append(seg.id)
            G[n1][n2]['seg_ids'] = existing
        else:
            G.add_edge(n1,n2, seg_ids=[seg.id], thickness=seg.thickness, style=seg.style, color=seg.color)
    # naive intersection splitting
    for i in range(len(global_segments)):
        a = global_segments[i]
        for j in range(i+1, len(global_segments)):
            b = global_segments[j]
            inter = segment_intersection((a.x1,a.y1,a.x2,a.y2),(b.x1,b.y1,b.x2,b.y2))
            if inter is not None:
                ix,iy = inter
                n_int = snap_or_add_node(int(round(ix)), int(round(iy)))
                for seg_ in (a,b):
                    n_a = snap_or_add_node(seg_.x1, seg_.y1)
                    n_b = snap_or_add_node(seg_.x2, seg_.y2)
                    if not G.has_edge(n_int, n_a):
                        G.add_edge(n_int, n_a, seg_ids=[seg_.id], thickness=seg_.thickness, style=seg_.style, color=seg_.color)
                    if not G.has_edge(n_int, n_b):
                        G.add_edge(n_int, n_b, seg_ids=[seg_.id], thickness=seg_.thickness, style=seg_.style, color=seg_.color)
    return G, offsets

# ------------------ Attach symbols & tags ------------------
def attach_symbols_and_tags_to_graph(G, symbols_per_page: List[List[SymbolCandidate]], tags_per_page: List[List[Dict]], offsets):
    attachments = defaultdict(lambda: {"symbols": [], "tags": []})
    nodes = list(G.nodes(data=True))
    node_xy = [(n, data['xy'][0], data['xy'][1]) for n,data in nodes]
    # symbols
    for page_syms in symbols_per_page:
        for s in page_syms:
            ox,oy = offsets[s.page-1]
            cx = s.centroid[0] + ox; cy = s.centroid[1] + oy
            best = None; bestd = None
            for n,x,y in node_xy:
                d = (x-cx)**2 + (y-cy)**2
                if bestd is None or d < bestd:
                    best = n; bestd = d
            if best:
                attachments[best]["symbols"].append({"bbox": s.bbox, "centroid": (cx,cy), "area": s.area, "page": s.page, "cls": s.cls, "score": s.score})
    # tags
    for page_tags in tags_per_page:
        for t in page_tags:
            ox,oy = offsets[t.get('page',1)-1] if 'page' in t else (0,0)
            tb = t['bbox']
            cx = tb[0] + tb[2]//2 + ox; cy = tb[1] + tb[3]//2 + oy
            best = None; bestd = None
            for n,x,y in node_xy:
                d = (x-cx)**2 + (y-cy)**2
                if bestd is None or d < bestd:
                    best = n; bestd = d
            if best:
                attachments[best]["tags"].append({"text": t['text'], "bbox": [tb[0]+ox,tb[1]+oy,tb[2],tb[3]], "confidence": t.get('confidence',0.0), "page": t.get('page',1)})
    return attachments

# ------------------ CENTUM XAML Writer ------------------
class CentumXamlWriter:
    def __init__(self, canvas_width: int, canvas_height: int, start_id: int = 500):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.elements: List[str] = []
        self.next_id = start_id

    def new_id(self):
        val = self.next_id
        self.next_id += 1
        return val

    def header(self):
        hdr = f"""<!--Generated by pid_to_xaml_centum_final_v2.py-->
<!--PlatformRevisionProperties.PlatformFileRevision:0x0002-->
<!--SolutionRevisionProperties.CENTUMFileRevision:0x0001-->
<Canvas
    MaxHeight="{self.canvas_height}"
    MaxWidth="{self.canvas_width}"
    HorizontalAlignment="Center"
    VerticalAlignment="Center"
    Width="{self.canvas_width}"
    Height="{self.canvas_height}"
    Background="#FFC0C0C0"
    Tag="MaxId={self.next_id}"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:yiapcspvgbdc="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Components.Controls.BasicShapeControls"
    xmlns:yiapcspvgbdc0="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Builder.Designer.Component"
    xmlns:yiapcspvggn="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.GenericName"
    xmlns:yiapcspvgcp="clr-namespace:Yokogawa.IA.iPCS.Platform.View.Graphic.Common.Persistency;assembly=Yokogawa.IA.iPCS.Platform.View.Graphic.Common.Persistency">"""
        return hdr

    def footer(self):
        return "</Canvas>"

    def add_ipcs_polyline(self, points: List[Tuple[int,int]], stroke: str = "#FF000000", stroke_thickness: int = 4,
                         dashed: bool = False, arrow_start: bool = False, arrow_end: bool = False,
                         name: Optional[str] = None, disjoint: bool = True, zindex: int = 0):
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        minx = min(xs); miny = min(ys)
        pts_rel = [f"{int(x-minx)},{int(y-miny)}" for x,y in points]
        pts_str = " ".join(pts_rel)
        dash_attr = ' StrokeDashArray="5 3"' if dashed else ''
        arrow_start_attr = ' ArrowStartStyle="Triangle"' if arrow_start else ''
        arrow_end_attr = ' ArrowEndStyle="Triangle"' if arrow_end else ''
        disj = "True" if disjoint else "False"
        tagid = self.new_id()
        nm = name or f"PolyLine{tagid}"
        el = f'''
    <yiapcspvgbdc:IPCSPolyLine
        Visibility="Visible"
        Focusable="False"
        StrokeThickness="{stroke_thickness}"
        Tag="Id={tagid}"
        {arrow_start_attr}
        {arrow_end_attr}
        ArrowSize="Small"
        Points="{pts_str}"
        Stroke="{stroke}"{dash_attr}
        Disjoint="{disj}"
        Canvas.Left="{int(minx)}"
        Canvas.Top="{int(miny)}"
        Panel.ZIndex="{zindex}"
        yiapcspvgbdc:ComponentProperties.Name="{nm}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName>
            <yiapcspvggn:GenericName />
        </yiapcspvggn:GenericNameComponent.GenericName>
    </yiapcspvgbdc:IPCSPolyLine>'''
        self.elements.append(el)

    def add_ipcs_line(self, x1:int,y1:int,x2:int,y2:int, stroke:str="#FF000000", stroke_thickness:int=1, dashed:bool=False, arrow_start:bool=False, arrow_end:bool=False, name:Optional[str]=None, zindex:int=0):
        minx = min(x1,x2); miny = min(y1,y2)
        X1 = int(x1 - minx); X2 = int(x2 - minx); Y1 = int(y1 - miny); Y2 = int(y2 - miny)
        dash_attr = ' StrokeDashArray="5 3" LineStyle="Dot"' if dashed else ''
        arrow_start_attr = ' ArrowStartStyle="Triangle"' if arrow_start else ''
        arrow_end_attr = ' ArrowEndStyle="Triangle"' if arrow_end else ''
        tagid = self.new_id()
        nm = name or f"Line{tagid}"
        el = f'''
    <yiapcspvgbdc:IPCSLine
        Visibility="Visible"
        Focusable="False"
        StrokeThickness="{stroke_thickness}"
        Tag="Id={tagid}"
        {arrow_start_attr}
        {arrow_end_attr}
        X1="{X1}"
        X2="{X2}"
        Y1="{Y1}"
        Y2="{Y2}"
        Stroke="{stroke}"{dash_attr}
        Canvas.Left="{int(minx)}"
        Canvas.Top="{int(miny)}"
        Panel.ZIndex="{zindex}"
        yiapcspvgbdc:ComponentProperties.Name="{nm}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName>
            <yiapcspvggn:GenericName />
        </yiapcspvggn:GenericNameComponent.GenericName>
    </yiapcspvgbdc:IPCSLine>'''
        self.elements.append(el)

    def add_fill_area(self, points: List[Tuple[int,int]], fill_color: str = "#FFAAAAAA", stroke: str = "#00FFFFFF", stroke_thickness: int = 1, name: Optional[str] = None, zindex:int=0):
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        minx = min(xs); miny = min(ys)
        pts_rel = [f"{int(x-minx)},{int(y-miny)}" for x,y in points]
        pts_str = " ".join(pts_rel)
        tagid = self.new_id()
        nm = name or f"FillArea{tagid}"
        el = f'''
    <yiapcspvgbdc:IPCSFillArea
        Visibility="Visible"
        Focusable="False"
        Stroke="{stroke}"
        StrokeThickness="{stroke_thickness}"
        Tag="Id={tagid}"
        Fill="{fill_color}"
        Points="{pts_str}"
        Canvas.Left="{int(minx)}"
        Canvas.Top="{int(miny)}"
        Panel.ZIndex="{zindex}"
        yiapcspvgbdc:ComponentProperties.Name="{nm}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName>
            <yiapcspvggn:GenericName />
        </yiapcspvggn:GenericNameComponent.GenericName>
    </yiapcspvgbdc:IPCSFillArea>'''
        self.elements.append(el)

    def add_text(self, x:int, y:int, text:str, font_size:int=14, zindex:int=0):
        tagid = self.new_id()
        esc = xaml_escape(text)
        el = f'''
    <yiapcspvgbdc:Text
        FontSize="{font_size}"
        Visibility="Visible"
        Focusable="False"
        FontFamily="Arial"
        Foreground="#FF000000"
        Tag="Id={tagid}"
        Canvas.Left="{int(x)}"
        Canvas.Top="{int(y)}"
        Panel.ZIndex="{zindex}"
        yiapcspvgbdc:ComponentProperties.Name="Text_{tagid}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">
        <yiapcspvggn:GenericNameComponent.GenericName>
            <yiapcspvggn:GenericName />
        </yiapcspvggn:GenericNameComponent.GenericName>{esc}</yiapcspvgbdc:Text>'''
        self.elements.append(el)

    def add_group_symbol(self, x:int, y:int, width:int, height:int, shape_points:Optional[List[Tuple[int,int]]] = None, fill_color:str="#FFA5A5A5", name:Optional[str]=None, zindex:int=0):
        tagid = self.new_id()
        nm = name or f"LinkedPart_{tagid}"
        el_header = f'''
    <yiapcspvgbdc0:GroupComponent
        Tag="Id={tagid}"
        Visibility="Visible"
        Width="{int(width)}"
        Height="{int(height)}"
        Canvas.Left="{int(x)}"
        Canvas.Top="{int(y)}"
        Panel.ZIndex="{zindex}"
        yiapcspvgbdc:ComponentProperties.Name="{nm}"
        yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">'''
        self.elements.append(el_header)
        if shape_points:
            pts_rel = " ".join([f"{int(px)},{int(py)}" for px,py in shape_points])
            fill_el = f'''
        <yiapcspvgbdc:IPCSFillArea
            Rotation="0"
            Visibility="Visible"
            Focusable="False"
            Stroke="#FFA5A5A5"
            StrokeThickness="1"
            Tag="Id={self.new_id()}"
            Fill="{fill_color}"
            Points="{pts_rel}"
            Canvas.Left="0"
            Canvas.Top="0"
            Panel.ZIndex="{zindex+1}"
            yiapcspvgbdc:ComponentProperties.Name="Polygon_{tagid}"
            yiapcspvgbdc:ComponentProperties.LayerID="Normal Drawing Layer 1">
            <yiapcspvggn:GenericNameComponent.GenericName>
                <yiapcspvggn:GenericName />
            </yiapcspvggn:GenericNameComponent.GenericName>
        </yiapcspvgbdc:IPCSFillArea>'''
            self.elements.append(fill_el)
        self.elements.append("    </yiapcspvgbdc0:GroupComponent>")

    def to_string(self):
        return "\n".join([self.header()] + self.elements + [self.footer()])

# ------------------ Generate CENTUM XAML from graph ------------------
def generate_centum_xaml(G, attachments, out_path: str, canvas_size=(2000,1200)):
    width, height = canvas_size
    writer = CentumXamlWriter(width, height, start_id=500)
    z = 50
    for u,v,data in G.edges(data=True):
        x1,y1 = G.nodes[u]['xy']; x2,y2 = G.nodes[v]['xy']
        pts = [(x1,y1),(x2,y2)]
        color = data.get('color',(0,0,0))
        stroke = color_to_centum_hex(color)
        thickness = int(round(max(1.0, float(data.get('thickness',1.0)))))
        style = data.get('style','solid')
        dashed = style in ('dashed','dotted')
        arrow_start = False; arrow_end = False
        if 'nav' in data.get('seg_ids',[]):
            arrow_end = True
        dx = abs(x2-x1); dy = abs(y2-y1)
        if dx == 0 or dy == 0:
            writer.add_ipcs_line(x1,y1,x2,y2, stroke=stroke, stroke_thickness=thickness, dashed=dashed, arrow_start=arrow_start, arrow_end=arrow_end, zindex=z)
        else:
            writer.add_ipcs_polyline(points=pts, stroke=stroke, stroke_thickness=thickness, dashed=dashed, arrow_start=arrow_start, arrow_end=arrow_end, zindex=z)
        z += 1

    # symbols and tags
    for n,data in G.nodes(data=True):
        x,y = data['xy']
        at = attachments.get(n, {})
        syms = at.get('symbols', [])
        tags = at.get('tags', [])
        for s in syms:
            bbox = s['bbox']
            sx, sy0, sw, sh = bbox
            shape_pts = [(0,0),(sw,0),(sw,sh),(0,sh)]
            writer.add_group_symbol(x=int(s['centroid'][0]-sw/2), y=int(s['centroid'][1]-sh/2), width=sw, height=sh, shape_points=shape_pts, fill_color="#FFA5A5A5", name=s.get('cls') or "Symbol", zindex=200)
        for t in tags:
            tb = t['bbox']
            tx = tb[0]; ty = tb[1]
            text = t.get('text','')
            writer.add_text(tx, ty, text, font_size=14, zindex=250)

    xaml_str = writer.to_string()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xaml_str)
    logging.info("Wrote CENTUM XAML to %s", out_path)

# ------------------ Main orchestrator ------------------
def main():
    parser = argparse.ArgumentParser(description="PID -> CENTUM XAML generator (final v2)")
    parser.add_argument("--pdf", required=True, help="input PDF path")
    parser.add_argument("--out_dir", default="output", help="output folder")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min_line_len", type=int, default=30)
    parser.add_argument("--max_line_gap", type=int, default=8)
    parser.add_argument("--hough_thresh", type=int, default=None)
    parser.add_argument("--snap_tol", type=int, default=12)
    parser.add_argument("--ocr", action="store_true", help="enable pytesseract OCR")
    parser.add_argument("--template_dir", default=None, help="directory with symbol templates")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--out_xaml", default="Main.xaml")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    logging.info("Loading PDF pages...")
    pages = pdf_to_images(args.pdf, dpi=args.dpi)
    logging.info("Loaded %d pages", len(pages))

    templates = load_templates(args.template_dir) if args.template_dir else {}

    all_segments: List[List[LineSegment]] = []
    all_symbols: List[List[SymbolCandidate]] = []
    all_arrows: List[List[Arrow]] = []
    all_tags: List[List[Dict]] = []

    for (page_num, img) in pages:
        logging.info("Processing page %d", page_num)
        # Use enhanced preprocessing to produce image for both detection and OCR layer
        pre_for_ocr = enhance_page_for_ocr(img, scale_up=1.4, denoise_strength=7, clahe_clip=3.0, binarize_patch=True, remove_graphic_lines=True)
        # Also create a detection-friendly version (slightly different — keep lines)
        # For detection we may prefer a less line-removed version; use denoised+clahe without line removal
        pre_for_detect = preprocess_image_for_detection(img, denoise=args.denoise, binarize=args.binarize, deskew=args.deskew)

        # Save pre images for debugging
        debug_write_image(pre_for_ocr, os.path.join(args.out_dir, f"page_{page_num:02d}_pre_ocr.png"))
        debug_write_image(pre_for_detect, os.path.join(args.out_dir, f"page_{page_num:02d}_pre_detect.png"))

        # Detect lines (on detection-friendly image)
        segs = detect_line_segments(pre_for_detect, min_length=args.min_line_len, max_gap=args.max_line_gap, hough_threshold=args.hough_thresh, page_idx=page_num)
        syms = detect_symbol_candidates(pre_for_detect, page_idx=page_num)
        if templates:
            syms = classify_symbols_by_template(pre_for_detect, syms, templates)
        arrows = detect_arrowheads(pre_for_detect, page_idx=page_num)
        tags = ocr_text_layer(pre_for_ocr, page_idx=page_num, angles=(0,-90,90)) if (args.ocr and PYSTESSERACT) else []

        all_segments.append(segs)
        all_symbols.append(syms)
        all_arrows.append(arrows)
        all_tags.append(tags)

        # debug overlay - show detection results on original-sized copy
        vis = img.copy()
        for s in segs:
            cv2.line(vis, (s.x1,s.y1), (s.x2,s.y2), (0,255,0), 1)
        for c in syms:
            x,y,w,h = c.bbox
            color = (255,0,0) if c.cls is None else (0,128,255)
            cv2.rectangle(vis, (x,y), (x+w,y+h), color, 1)
            if c.cls:
                cv2.putText(vis, f'{c.cls}:{c.score:.2f}', (x, max(0,y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        for t in tags:
            x,y,w,h = t['bbox']
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 1)
            try:
                cv2.putText(vis, t['text'], (x, max(0,y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            except Exception:
                pass
        for a in arrows:
            cv2.circle(vis, (a.x,a.y), 6, (0,0,255), 1)
            cv2.putText(vis, a.direction[0].upper(), (a.x+6,a.y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        debug_write_image(vis, os.path.join(args.out_dir, f"page_{page_num:02d}_debug.png"))

    # compute page offsets (tile horizontally by default)
    n_pages = len(pages)
    widths = [img.shape[1] for (_,img) in pages]
    heights = [img.shape[0] for (_,img) in pages]
    if n_pages == 3:
        offsets = [(0,0),(widths[0],0),(widths[0]+widths[1],0)]
    else:
        offsets = []
        x = 0
        for w in widths:
            offsets.append((x,0)); x += w

    logging.info("Building merged graph across pages...")
    G, page_offsets = build_merged_graph(all_segments, pages, page_order_positions=offsets, snap_tol=args.snap_tol)
    logging.info("Total lines detected: %d", sum(len(s) for s in all_segments))
    logging.info("Total graph nodes: %d", len(G.nodes))
    logging.info("Total graph edges: %d", len(G.edges))

    attachments = attach_symbols_and_tags_to_graph(G, all_symbols, all_tags, page_offsets)

    # navigation linking via arrows
    page_widths = widths
    for p_idx,(page_num,img) in enumerate(pages):
        arrows = all_arrows[p_idx]
        ox,oy = page_offsets[p_idx]
        for a in arrows:
            gx = a.x + ox; gy = a.y + oy
            try:
                nearest = min(G.nodes(data=True), key=lambda nd: (nd[1]['xy'][0]-gx)**2 + (nd[1]['xy'][1]-gy)**2)
            except ValueError:
                continue
            if a.direction == 'right' and p_idx+1 < len(pages):
                next_off = page_offsets[p_idx+1]
                candidates = [(n,data) for n,data in G.nodes(data=True) if (next_off[0]-5 <= data['xy'][0] <= next_off[0]+60)]
                if candidates:
                    target = min(candidates, key=lambda nd: abs(nd[1]['xy'][1]-gy))
                    G.add_edge(nearest[0], target[0], seg_ids=['nav'], thickness=1.0, style='navigation', color=(0,0,255))
            if a.direction == 'left' and p_idx-1 >= 0:
                prev_off = page_offsets[p_idx-1]
                limit_left = prev_off[0] + (page_widths[p_idx-1] - 60)
                candidates = [(n,data) for n,data in G.nodes(data=True) if (limit_left <= data['xy'][0] <= prev_off[0] + page_widths[p_idx-1] + 5)]
                if candidates:
                    target = min(candidates, key=lambda nd: abs(nd[1]['xy'][1]-gy))
                    G.add_edge(nearest[0], target[0], seg_ids=['nav'], thickness=1.0, style='navigation', color=(0,0,255))

    combined_width = sum(widths)
    max_h = max(heights) if heights else 1200
    out_xaml = os.path.join(args.out_dir, args.out_xaml)
    generate_centum_xaml(G, attachments, out_xaml, canvas_size=(combined_width, max_h))

    # save JSON artifacts
    summary = {"pdf": args.pdf, "pages": len(pages), "total_segments": sum(len(s) for s in all_segments), "graph_nodes": len(G.nodes), "graph_edges": len(G.edges)}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    nodes_export = {n:{"xy":data['xy']} for n,data in G.nodes(data=True)}
    edges_export = []
    for u,v,data in G.edges(data=True):
        edges_export.append({"u":u,"v":v,"data": {"seg_ids": data.get("seg_ids",[]), "thickness": data.get("thickness"), "style": data.get("style")}})
    with open(os.path.join(args.out_dir, "graph_nodes.json"), "w", encoding="utf-8") as f:
        json.dump(nodes_export, f, indent=2)
    with open(os.path.join(args.out_dir, "graph_edges.json"), "w", encoding="utf-8") as f:
        json.dump(edges_export, f, indent=2)

    logging.info("POC finished. Results in %s", args.out_dir)

# ------------------ Small helper: detection-preprocess wrapper ------------------
def preprocess_image_for_detection(img: np.ndarray, denoise=True, binarize=False, deskew=True) -> np.ndarray:
    """
    Simple preprocess variant for detection pipeline — less aggressive line removal than OCR enhancement.
    Uses denoising, optional deskew, optional binarization.
    """
    out = img.copy()
    if denoise:
        try:
            out = cv2.fastNlMeansDenoisingColored(out, None, 8, 8, 7, 21)
        except Exception:
            out = cv2.blur(out, (3,3))
    if deskew:
        out = deskew_image_for_detection(out)
    if binarize:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 9)
        out = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return out

def deskew_image_for_detection(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    if lines is None:
        return img
    angles = []
    for l in lines[:100]:
        rho,theta = l[0]
        angle = (theta - np.pi/2) * (180/np.pi)
        angles.append(angle)
    if len(angles)==0:
        return img
    median = np.median(angles)
    if abs(median) < 0.1:
        return img
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), median, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

if __name__ == "__main__":
    main()
