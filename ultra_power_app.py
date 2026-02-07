"""
⚡ ULTRA POWER GEOMETRY QA v2.0 - WITH VISION & VECTOR VALIDATION
Gemini + Oxlo.ai Dual AI • Advanced ML • Vision Analysis • Vector QA
Image Anomaly Detection • Geometry Validation • Complete QA Suite
ENTERPRISE GRADE - MAXIMUM CAPABILITY
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from pathlib import Path
from shapely import wkt
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import explain_validity
import json
import warnings
import requests
from datetime import datetime
from io import StringIO
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from PIL import Image
import io

warnings.filterwarnings('ignore')

# ============================================================================
# ULTRA POWER CONFIG - DUAL AI ENGINES + VISION
# ============================================================================

st.set_page_config(
    page_title="⚡ ULTRA POWER GEOMETRY QA v2.0 + VISION",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GEMINI_API_KEY = "AIzaSyD6dIZXQOwNCuAJ6dikiFOklinfd9C06YI"
OXLO_API_KEY = "sk_zxi3JzCfEZ4o7TchUMH180We_JgJc-Hs2yX3uP3II9E"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ============================================================================
# ULTRA POWER DARK THEME
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&display=swap');

    * { font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }
    body { 
        background: radial-gradient(1200px 600px at 10% 0%, #fff7ed 0%, transparent 60%),
                    radial-gradient(900px 500px at 90% 10%, #e0f2fe 0%, transparent 55%),
                    linear-gradient(140deg, #f7f4ef 0%, #f2f8ff 45%, #f8fff6 100%);
        color: #1f2937;
    }

    .stApp { background: transparent; }

    .power-header {
        background: linear-gradient(120deg, #ffffff 0%, #f6fffb 100%);
        border: 3px solid #0ea5a5;
        border-radius: 18px;
        padding: 30px;
        margin: 25px 0;
        box-shadow: 0 18px 40px rgba(14, 165, 165, 0.15), inset 0 0 30px rgba(255, 107, 53, 0.08);
    }

    .power-header h1 {
        color: #0f172a;
        font-size: 44px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
    }

    .power-header p {
        color: #475569;
        font-size: 12px;
        margin: 10px 0 0 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-title {
        color: #0f766e;
        font-size: 24px;
        font-weight: 800;
        margin: 35px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 4px solid #ff6b35;
        letter-spacing: 2px;
    }

    .metric-ultra {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #0ea5a5;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 12px 25px rgba(15, 118, 110, 0.12);
        margin: 12px 0;
    }

    .metric-ultra-label {
        color: #0f766e;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.9;
        font-weight: 700;
    }

    .metric-ultra-value {
        color: #ff6b35;
        font-size: 38px;
        font-weight: 800;
        margin: 10px 0 0 0;
    }

    .error-box {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.12) 0%, rgba(255, 107, 53, 0.04) 100%);
        border-left: 6px solid #ff6b35;
        border-right: 2px solid #ff6b35;
        padding: 16px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 10px 20px rgba(255, 107, 53, 0.15);
        color: #9a3412;
        font-weight: 700;
    }

    .success-box {
        background: linear-gradient(135deg, rgba(14, 165, 165, 0.12) 0%, rgba(14, 165, 165, 0.04) 100%);
        border-left: 6px solid #0ea5a5;
        border-right: 2px solid #0ea5a5;
        padding: 16px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 10px 20px rgba(14, 165, 165, 0.15);
        color: #0f766e;
        font-weight: 700;
    }

    .anomaly-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.04) 100%);
        border-left: 6px solid #f59e0b;
        border-right: 2px solid #f59e0b;
        padding: 16px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 10px 20px rgba(245, 158, 11, 0.15);
        color: #92400e;
        font-weight: 700;
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff6b35, #ff9f1c) !important;
        color: #111827 !important;
        border: 2px solid #ff6b35 !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        padding: 12px 22px !important;
        box-shadow: 0 10px 22px rgba(255, 107, 53, 0.2) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 12px;
    }

    .stButton > button:hover {
        box-shadow: 0 16px 30px rgba(255, 107, 53, 0.35) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API FUNCTIONS
# ============================================================================

def call_oxlo_gemini_pro(prompt: str) -> str:
    """Call Oxlo.ai GPT-4"""
    try:
        headers = {
            "Authorization": f"Bearer {OXLO_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        response = requests.post(
            "https://api.oxlo.ai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=15
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return "Oxlo.ai API error"
    except Exception as e:
        return f"Error: {str(e)[:100]}"

def call_gemini_vision(prompt: str) -> str:
    """Call Gemini 2.0 Flash"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text if response else "No response"
    except Exception as e:
        return f"Gemini error: {str(e)[:100]}"

def analyze_image_with_vision(image_path: str, analysis_type: str = "general") -> dict:
    """Analyze geometry image with Gemini Vision"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        image = Image.open(image_path)
        
        prompts = {
            "general": "Analyze this geometry/map image. What anomalies or errors do you see09 (Missing features, distorted shapes, incorrect styling, gaps, overlaps, coordinate issues)",
            "missing_features": "Check this map image for missing line features or incomplete geometries. List what's missing.",
            "distortion": "Analyze this geometry image for distorted shapes, irregular polygons, or coordinate errors.",
            "topology": "Check this geometry visualization for topology errors (gaps between features, overlaps, self-intersections)."
        }
        
        prompt = prompts.get(analysis_type, prompts["general"])
        
        response = model.generate_content([prompt, image])
        result = response.text if response else "No anomalies detected"
        
        return {
            'status': 'success',
            'analysis': result,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)[:200],
            'analysis_type': analysis_type
        }

# ============================================================================
# HACKATHON SINGLE-ERROR MODE (SELF-INTERSECTION)
# ============================================================================

HACK_TRAIN_GOOD = Path(__file__).parent / "training" / "good_examples.wkt"
HACK_ERROR_EXAMPLES = Path(__file__).parent / "errors" / "self_intersection_examples.wkt"

HACK_DEFAULT_GOOD = """
POLYGON ((0 0, 6 0, 6 4, 0 4, 0 0))
POLYGON ((10 0, 14 0, 14 5, 10 5, 10 0))
POLYGON ((0 10, 3 10, 4 14, 1 16, 0 10))
POLYGON ((20 10, 25 10, 25 14, 22 16, 20 14, 20 10))
POLYGON ((30 0, 35 0, 36 4, 33 6, 30 4, 30 0))
POLYGON ((40 10, 45 10, 45 15, 40 15, 40 10))
""".strip()

HACK_DEFAULT_BAD = """
POLYGON ((0 0, 4 4, 0 4, 4 0, 0 0))
POLYGON ((10 0, 14 4, 10 4, 14 0, 10 0))
POLYGON ((20 0, 24 4, 20 4, 24 0, 20 0))
""".strip()


def hack_write_default_examples():
    HACK_TRAIN_GOOD.parent.mkdir(parents=True, exist_ok=True)
    HACK_ERROR_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    HACK_TRAIN_GOOD.write_text(HACK_DEFAULT_GOOD + "\n", encoding="utf-8")
    HACK_ERROR_EXAMPLES.write_text(HACK_DEFAULT_BAD + "\n", encoding="utf-8")
    return len(HACK_DEFAULT_GOOD.splitlines()), len(HACK_DEFAULT_BAD.splitlines())


def hack_load_wkts_from_text(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []
    parse_errors = []
    for idx, line in enumerate(lines, 1):
        try:
            geom = wkt.loads(line)
            rows.append({"id": idx, "wkt": line, "geom": geom})
        except Exception as exc:
            parse_errors.append({"id": idx, "wkt": line, "error": str(exc)})
    return rows, parse_errors


def hack_geom_features(geom):
    if geom.is_empty:
        return None
    area = geom.area
    perimeter = geom.length
    minx, miny, maxx, maxy = geom.bounds
    width = max(maxx - minx, 0.0)
    height = max(maxy - miny, 0.0)
    aspect = width / height if height > 0 else 0.0
    if isinstance(geom, Polygon):
        vertex_count = len(list(geom.exterior.coords))
    elif isinstance(geom, LineString):
        vertex_count = len(list(geom.coords))
    else:
        vertex_count = 1
    compactness = 0.0
    if area > 0 and perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter ** 2)
    return [area, perimeter, vertex_count, aspect, compactness]


def hack_train_model(good_rows):
    feats = []
    for row in good_rows:
        f = hack_geom_features(row["geom"])
        if f is not None:
            feats.append(f)
    if len(feats) < 3:
        return None, None
    X = np.array(feats, dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(Xs)
    return model, scaler


def hack_self_intersection_info(geom):
    msg = explain_validity(geom)
    if msg is None:
        return False, ""
    if "Self-intersection" in msg or "self-intersection" in msg:
        loc = ""
        m = re.search(r"\\[(.+)\\]", msg)
        if m:
            loc = m.group(1)
        return True, loc
    return False, ""

# ============================================================================
# VECTOR QA VALIDATION - RULE-BASED GEOMETRY CHECKING
# ============================================================================

class VectorQAValidator:
    """Rule-based validation for vector geometry data"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.rules = {
            'no_gaps': True,
            'no_overlaps': True,
            'valid_coordinates': True,
            'closed_rings': True,
            'no_self_intersections': True
        }
    
    def validate_geometry(self, geom: any, geom_id: int = 0) -> dict:
        """Complete geometry validation"""
        results = {
            'geom_id': geom_id,
            'errors': [],
            'warnings': [],
            'valid': True,
            'checks_passed': 0,
            'checks_failed': 0
        }
        
        try:
            # Check 1: Is valid
            if not geom.is_valid:
                results['errors'].append('Invalid topology detected')
                results['valid'] = False
                results['checks_failed'] += 1
            else:
                results['checks_passed'] += 1
            
            # Check 2: Has coordinates
            if not geom.bounds:
                results['errors'].append('No coordinates found')
                results['valid'] = False
                results['checks_failed'] += 1
            else:
                results['checks_passed'] += 1
            
            # Check 3: Coordinate range
            bounds = geom.bounds
            if len(bounds) == 4:
                minx, miny, maxx, maxy = bounds
                if minx == maxx or miny == maxy:
                    results['warnings'].append('Degenerate geometry (zero area)')
                    results['checks_failed'] += 1
                else:
                    results['checks_passed'] += 1
            
            # Check 4: Polygon specifics
            if geom.geom_type == 'Polygon':
                if not geom.exterior.is_ring:
                    results['errors'].append('Exterior ring not closed')
                    results['valid'] = False
                    results['checks_failed'] += 1
                else:
                    results['checks_passed'] += 1
                
                # Check interior rings
                for idx, interior in enumerate(geom.interiors):
                    if not interior.is_ring:
                        results['errors'].append(f'Interior ring {idx} not closed')
                        results['valid'] = False
                        results['checks_failed'] += 1
            
            # Check 5: Vertex count
            vertices = 0
            if hasattr(geom, 'exterior'):
                vertices = len(list(geom.exterior.coords))
            elif hasattr(geom, 'coords'):
                vertices = len(list(geom.coords))
            
            if vertices < 3:
                results['warnings'].append(f'Very few vertices ({vertices})')
                results['checks_failed'] += 1
            else:
                results['checks_passed'] += 1
            
            results['checks_passed'] = max(0, results['checks_passed'] - results['checks_failed'])
            
        except Exception as e:
            results['errors'].append(f'Validation error: {str(e)[:100]}')
            results['valid'] = False
        
        return results
    
    def check_gaps_overlaps(self, geometries: list) -> dict:
        """Check for gaps and overlaps between geometries"""
        results = {
            'gaps': [],
            'overlaps': [],
            'adjacent_pairs': [],
            'total_checked': len(geometries)
        }
        
        try:
            for i in range(len(geometries)):
                for j in range(i + 1, len(geometries)):
                    geom1 = geometries[i]
                    geom2 = geometries[j]
                    
                    # Check intersection
                    if geom1.intersects(geom2):
                        intersection = geom1.intersection(geom2)
                        if intersection.area > 0:
                            results['overlaps'].append({
                                'geom1_id': i,
                                'geom2_id': j,
                                'overlap_area': float(intersection.area)
                            })
                        else:
                            results['adjacent_pairs'].append((i, j))
        except:
            pass
        
        return results

# ============================================================================
# 12-FEATURE ML DETECTOR (ENHANCED)
# ============================================================================

class UltraPowerMLDetector:
    """12-Feature ML with vision integration"""
    
    def __init__(self, contamination: float = 0.15):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, n_estimators=300, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = [
            'area', 'perimeter', 'vertices', 'compactness', 'aspect_ratio',
            'circularity', 'solidity', 'fractal_dim', 'centroid_dist', 
            'coord_variance', 'bounds_ratio', 'density'
        ]
    
    def extract_features(self, geometries: list) -> np.ndarray:
        """Extract 12 features"""
        features = []
        for geom in geometries:
            try:
                if isinstance(geom, str):
                    geom = wkt.loads(geom)
                
                area = float(geom.area) if hasattr(geom, 'area') else 0
                perimeter = float(geom.length) if hasattr(geom, 'length') else 0
                
                vertices = 0
                if hasattr(geom, 'exterior'):
                    vertices = len(list(geom.exterior.coords))
                elif hasattr(geom, 'coords'):
                    vertices = len(list(geom.coords))
                
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                bounds = geom.bounds if hasattr(geom, 'bounds') else (0, 0, 0, 0)
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                aspect_ratio = width / height if height > 0 else 1
                
                radius = perimeter / (2 * np.pi) if perimeter > 0 else 1
                circularity = area / (np.pi * radius ** 2) if radius > 0 else 0
                
                try:
                    convex_hull = geom.convex_hull
                    solidity = area / convex_hull.area if convex_hull.area > 0 else 1
                except:
                    solidity = 1
                
                fractal_dim = np.log(vertices) / np.log(perimeter) if vertices > 0 and perimeter > 0 else 0
                
                try:
                    centroid = geom.centroid
                    bound_center_x = (bounds[0] + bounds[2]) / 2
                    bound_center_y = (bounds[1] + bounds[3]) / 2
                    centroid_dist = np.sqrt((centroid.x - bound_center_x) ** 2 + (centroid.y - bound_center_y) ** 2)
                except:
                    centroid_dist = 0
                
                try:
                    coords = list(geom.exterior.coords) if hasattr(geom, 'exterior') else list(geom.coords)
                    if len(coords) > 1:
                        distances = [np.sqrt((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2) 
                                   for i in range(len(coords)-1)]
                        coord_variance = np.std(distances) if distances else 0
                    else:
                        coord_variance = 0
                except:
                    coord_variance = 0
                
                bounds_ratio = (width + height) / (width * height) if width * height > 0 else 0
                density = vertices / (area + 1)
                
                features.append([area, perimeter, vertices, compactness, aspect_ratio,
                               circularity, solidity, fractal_dim, centroid_dist,
                               coord_variance, bounds_ratio, density])
            except:
                features.append([0] * 12)
        
        return np.array(features)
    
    def train(self, geometries: list) -> bool:
        """Train model"""
        if len(geometries) < 5:
            return False
        
        features = self.extract_features(geometries)
        self.scaler.fit(features)
        scaled = self.scaler.transform(features)
        self.model.fit(scaled)
        self.trained = True
        return True
    
    def predict(self, geometries: list) -> dict:
        """Detect anomalies"""
        if not self.trained:
            return {'error': 'Model not trained'}
        
        features = self.extract_features(geometries)
        scaled = self.scaler.transform(features)
        predictions = self.model.predict(scaled)
        scores = self.model.score_samples(scaled)
        
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:
                anomalies.append({
                    'id': i,
                    'score': float(score),
                    'severity': 'CRITICAL' if score < -1.0 else 'HIGH' if score < -0.5 else 'MEDIUM',
                    'features': dict(zip(self.feature_names, features[i]))
                })
        
        return {
            'total': len(geometries),
            'anomalies': len(anomalies),
            'rate': len(anomalies) / len(geometries) * 100,
            'list': anomalies,
            'min_score': float(scores.min()),
            'max_score': float(scores.max()),
            'avg_score': float(scores.mean())
        }

# ============================================================================
# SESSION STATE
# ============================================================================

if 'geometries' not in st.session_state:
    st.session_state.geometries = []
if 'ml_detector' not in st.session_state:
    st.session_state.ml_detector = UltraPowerMLDetector()
if 'vector_qa' not in st.session_state:
    st.session_state.vector_qa = VectorQAValidator()
if 'errors' not in st.session_state:
    st.session_state.errors = []
if 'fixes' not in st.session_state:
    st.session_state.fixes = []
if 'vision_results' not in st.session_state:
    st.session_state.vision_results = []

# ============================================================================
# POWER HEADER
# ============================================================================

st.markdown("""
<div class="power-header">
    <h1>⚡ ULTRA POWER GEOMETRY QA v2.0 + VISION & VECTOR QA</h1>
    <p>Gemini 2.0 + Oxlo.ai • 12-Feature ML • Vision Analysis • Vector Validation • Complete QA Suite</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 1: LOAD & VALIDATE
# ============================================================================

st.markdown('<div class="section-title">01 LOAD & VALIDATE GEOMETRIES</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2.5, 1, 1])

with col1:
    uploaded_file = st.file_uploader("📁 Upload WKT File", type=['wkt', 'txt'])

with col2:
    if st.button("⚡ SCAN", use_container_width=True):
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                geometries = []
                errors = []
                
                for idx, line in enumerate(content.strip().split('\n')):
                    if not line.strip() or line.startswith('#'):
                        continue
                    try:
                        geom = wkt.loads(line.strip())
                        geometries.append(geom)
                        
                        # Validate with Vector QA
                        qa_result = st.session_state.vector_qa.validate_geometry(geom, idx)
                        
                        if not qa_result['valid']:
                            for err in qa_result['errors']:
                                errors.append({
                                    'id': idx,
                                    'type': 'VALIDATION_ERROR',
                                    'message': err,
                                    'severity': 'CRITICAL'
                                })
                    except Exception as e:
                        errors.append({
                            'id': idx,
                            'type': 'PARSE_ERROR',
                            'message': str(e)[:100],
                            'severity': 'ERROR'
                        })
                
                st.session_state.geometries = geometries
                st.session_state.errors = errors
                
                st.success(f"✅ Loaded {len(geometries)} geometries | Errors: {len(errors)}")
            except Exception as e:
                st.error(f"❌ {str(e)[:200]}")

with col3:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.geometries = []
        st.session_state.errors = []
        st.session_state.vision_results = []
        st.rerun()

# Display metrics
if st.session_state.geometries:
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f"""
        <div class="metric-ultra">
            <div class="metric-ultra-label">📊 Total</div>
            <div class="metric-ultra-value">{len(st.session_state.geometries)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        valid = len([g for g in st.session_state.geometries if g.is_valid])
        st.markdown(f"""
        <div class="metric-ultra">
            <div class="metric-ultra-label">✅ Valid</div>
            <div class="metric-ultra-value">{valid}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        invalid = len([g for g in st.session_state.geometries if not g.is_valid])
        st.markdown(f"""
        <div class="metric-ultra">
            <div class="metric-ultra-label">❌ Invalid</div>
            <div class="metric-ultra-value">{invalid}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        errors = len(st.session_state.errors)
        st.markdown(f"""
        <div class="metric-ultra">
            <div class="metric-ultra-label">⚠️ Errors</div>
            <div class="metric-ultra-value">{errors}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m5:
        total_vertices = sum(len(list(g.exterior.coords)) if hasattr(g, 'exterior') else len(list(g.coords)) for g in st.session_state.geometries)
        st.markdown(f"""
        <div class="metric-ultra">
            <div class="metric-ultra-label">🔢 Vertices</div>
            <div class="metric-ultra-value">{total_vertices}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# SECTION 2: VISION IMAGE ANALYSIS ⭐ NEW
# ============================================================================

st.markdown('<div class="section-title">02 VISION IMAGE ANOMALY DETECTION</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2.5, 1, 1])

with col1:
    image_file = st.file_uploader("📸 Upload Map/Geometry Image", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])

with col2:
    analysis_type = st.selectbox(
        "Analysis Type",
        ["general", "missing_features", "distortion", "topology"],
        key="vision_analysis_type"
    )

with col3:
    if st.button("👁️ ANALYZE", use_container_width=True):
        if image_file:
            try:
                # Save image temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(image_file.read())
                    tmp_path = tmp.name
                
                # Analyze with Vision
                result = analyze_image_with_vision(tmp_path, analysis_type)
                
                if result['status'] == 'success':
                    st.session_state.vision_results.append(result)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ VISION ANALYSIS COMPLETE<br>
                        Type: {result['analysis_type'].upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Analysis Results:**")
                    st.info(result['analysis'])
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        ❌ ANALYSIS ERROR<br>
                        {result.get('error', 'Unknown error')}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cleanup
                import os
                try:
                    os.remove(tmp_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"Vision analysis error: {str(e)[:200]}")

# Display vision history
if st.session_state.vision_results:
    with st.expander("📜 Vision Analysis History", expanded=False):
        for idx, result in enumerate(st.session_state.vision_results[-5:]):
            st.markdown(f"""
            <div class="anomaly-box">
                <strong>Analysis {idx + 1}</strong> | {result['analysis_type'].upper()}<br>
                Time: {result['timestamp']}<br>
                {result['analysis'][:200]}...
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# SECTION 3: VECTOR QA VALIDATION
# ============================================================================

st.markdown('<div class="section-title">03 VECTOR QA VALIDATION</div>', unsafe_allow_html=True)

if st.session_state.geometries:
    if st.button("🔍 RUN VECTOR QA CHECKS", use_container_width=True):
        try:
            qa_results = []
            gap_overlap_results = st.session_state.vector_qa.check_gaps_overlaps(st.session_state.geometries)
            
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.markdown(f"""
                <div class="metric-ultra">
                    <div class="metric-ultra-label">✅ Checked</div>
                    <div class="metric-ultra-value">{gap_overlap_results['total_checked']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m2:
                st.markdown(f"""
                <div class="metric-ultra">
                    <div class="metric-ultra-label">🔗 Adjacent</div>
                    <div class="metric-ultra-value">{len(gap_overlap_results['adjacent_pairs'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m3:
                st.markdown(f"""
                <div class="metric-ultra">
                    <div class="metric-ultra-label">⚠️ Overlaps</div>
                    <div class="metric-ultra-value">{len(gap_overlap_results['overlaps'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m4:
                gap_count = sum(1 for g1 in st.session_state.geometries for g2 in st.session_state.geometries 
                               if g1 != g2 and not g1.touches(g2) and not g1.intersects(g2))
                st.markdown(f"""
                <div class="metric-ultra">
                    <div class="metric-ultra-label">💔 Gaps</div>
                    <div class="metric-ultra-value">{gap_count // 2}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show overlaps
            if gap_overlap_results['overlaps']:
                st.markdown("**⚠️ Overlapping Geometries:**")
                for overlap in gap_overlap_results['overlaps'][:5]:
                    st.markdown(f"""
                    <div class="error-box">
                        Geom {overlap['geom1_id']} ∩ Geom {overlap['geom2_id']}<br>
                        Overlap Area: {overlap['overlap_area']:.4f}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show adjacent
            if gap_overlap_results['adjacent_pairs']:
                st.markdown("**🔗 Adjacent Geometries (Touching):**")
                for pair in gap_overlap_results['adjacent_pairs'][:5]:
                    st.markdown(f"✅ Geom {pair[0]} touches Geom {pair[1]}")
        
        except Exception as e:
            st.error(f"QA validation error: {str(e)[:150]}")

# ============================================================================
# SECTION 4: Y/X BOX VISUALIZATION
# ============================================================================

st.markdown('<div class="section-title">04 Y/X COORDINATE BOX</div>', unsafe_allow_html=True)

if st.session_state.geometries:
    if st.button("📊 PLOT COORDINATES", use_container_width=True):
        try:
            all_coords = []
            for geom in st.session_state.geometries:
                if hasattr(geom, 'exterior'):
                    coords = list(geom.exterior.coords)
                elif hasattr(geom, 'coords'):
                    coords = list(geom.coords)
                else:
                    coords = []
                all_coords.extend(coords)
            
            if all_coords:
                x_vals = [c[0] for c in all_coords]
                y_vals = [c[1] for c in all_coords]
                
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.scatter(x_vals, y_vals, c='#00d4ff', s=100, alpha=0.7, edgecolors='#ff3333', linewidth=2)
                ax.set_xlabel('X Coordinate', fontsize=14, color='#00d4ff', fontweight='bold')
                ax.set_ylabel('Y Coordinate', fontsize=14, color='#00d4ff', fontweight='bold')
                ax.set_title('🔬 Geometry Coordinate Distribution (Y/X Box)', fontsize=16, color='#ff3333', fontweight='bold')
                ax.grid(True, alpha=0.2, color='#00d4ff')
                ax.set_facecolor('#0a0a0a')
                fig.patch.set_facecolor('#1a1a2a')
                
                st.pyplot(fig)
                st.info(f"📊 Total points: {len(all_coords)} | X range: [{min(x_vals):.2f}, {max(x_vals):.2f}] | Y range: [{min(y_vals):.2f}, {max(y_vals):.2f}]")
        except Exception as e:
            st.error(f"Plot error: {str(e)[:100]}")

# ============================================================================
# SECTION 5: GEOMETRY VISUALIZATION
# ============================================================================

st.markdown('<div class="section-title">05 GEOMETRY GRAPH VISUALIZATION</div>', unsafe_allow_html=True)

if st.session_state.geometries:
    if st.button("📈 DRAW GEOMETRIES", use_container_width=True):
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            colors = plt.cm.Spectral(np.linspace(0, 1, len(st.session_state.geometries)))
            
            for idx, geom in enumerate(st.session_state.geometries):
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax.fill(x, y, alpha=0.3, color=colors[idx], edgecolor=colors[idx], linewidth=2)
                    ax.plot(x, y, color=colors[idx], linewidth=3)
                elif geom.geom_type == 'LineString':
                    x, y = geom.xy
                    ax.plot(x, y, color=colors[idx], linewidth=3, label=f'Line {idx}')
            
            ax.set_xlabel('X', fontsize=12, color='#00d4ff', fontweight='bold')
            ax.set_ylabel('Y', fontsize=12, color='#00d4ff', fontweight='bold')
            ax.set_title('⚡ ULTRA POWER GEOMETRY VISUALIZATION', fontsize=16, color='#ff3333', fontweight='bold')
            ax.grid(True, alpha=0.2, color='#00d4ff')
            ax.set_facecolor('#0a0a0a')
            fig.patch.set_facecolor('#1a1a2a')
            ax.legend(loc='upper right', fontsize=10)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Visualization error: {str(e)[:100]}")

# ============================================================================
# SECTION 6: 12-FEATURE ML DETECTION
# ============================================================================

st.markdown('<div class="section-title">06 12-FEATURE ML ANOMALY DETECTION</div>', unsafe_allow_html=True)

if st.session_state.geometries:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        threshold = st.slider("Anomaly Threshold (%)", 1, 50, 15)
    
    with col2:
        if st.button("🔍 DETECT", use_container_width=True):
            try:
                detector = UltraPowerMLDetector(contamination=threshold/100)
                
                if detector.train(st.session_state.geometries):
                    results = detector.predict(st.session_state.geometries)
                    
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with m1:
                        st.markdown(f"""
                        <div class="metric-ultra">
                            <div class="metric-ultra-label">Analyzed</div>
                            <div class="metric-ultra-value">{results['total']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with m2:
                        st.markdown(f"""
                        <div class="metric-ultra">
                            <div class="metric-ultra-label">🚨 Anomalies</div>
                            <div class="metric-ultra-value">{results['anomalies']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with m3:
                        st.markdown(f"""
                        <div class="metric-ultra">
                            <div class="metric-ultra-label">Rate %</div>
                            <div class="metric-ultra-value">{results['rate']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with m4:
                        st.markdown(f"""
                        <div class="metric-ultra">
                            <div class="metric-ultra-label">Avg Score</div>
                            <div class="metric-ultra-value">{results['avg_score']:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if results['list']:
                        st.markdown("**Anomalies Detected:**")
                        for anom in results['list'][:5]:
                            st.markdown(f"""
                            <div class="anomaly-box">
                                <strong>Geometry {anom['id']}</strong> | {anom['severity']}<br>
                                Score: {anom['score']:.4f}
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"ML error: {str(e)[:200]}")

# ============================================================================
# SECTION 7: DUAL AI AUTO-FIX
# ============================================================================

st.markdown('<div class="section-title">07 DUAL AI AUTO-FIX</div>', unsafe_allow_html=True)

if st.session_state.geometries and st.session_state.errors:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✨ GEMINI AUTO-FIX", use_container_width=True):
            try:
                fix_count = 0
                for err in st.session_state.errors[:3]:
                    if err['id'] < len(st.session_state.geometries):
                        geom_wkt = st.session_state.geometries[err['id']].wkt
                        prompt = f"Fix: {geom_wkt}\nError: {err['message']}\nProvide ONLY valid corrected WKT."
                        fixed_wkt = call_gemini_vision(prompt)
                        
                        if 'POLYGON' in fixed_wkt or 'LINESTRING' in fixed_wkt:
                            st.session_state.fixes.append({
                                'geom_id': err['id'],
                                'original': geom_wkt,
                                'fixed': fixed_wkt,
                                'engine': 'GEMINI',
                                'timestamp': datetime.now().isoformat()
                            })
                            fix_count += 1
                
                if fix_count > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ GEMINI Fixed {fix_count} geometries
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gemini fix error: {str(e)[:150]}")
    
    with col2:
        if st.button("🚀 OXLO AUTO-FIX", use_container_width=True):
            try:
                fix_count = 0
                for err in st.session_state.errors[:3]:
                    if err['id'] < len(st.session_state.geometries):
                        geom_wkt = st.session_state.geometries[err['id']].wkt
                        prompt = f"Fix: {geom_wkt}\nError: {err['message']}\nReturn ONLY valid WKT."
                        fixed_wkt = call_oxlo_gemini_pro(prompt)
                        
                        if 'POLYGON' in fixed_wkt or 'LINESTRING' in fixed_wkt:
                            st.session_state.fixes.append({
                                'geom_id': err['id'],
                                'original': geom_wkt,
                                'fixed': fixed_wkt,
                                'engine': 'OXLO',
                                'timestamp': datetime.now().isoformat()
                            })
                            fix_count += 1
                
                if fix_count > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        🚀 OXLO Fixed {fix_count} geometries
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Oxlo fix error: {str(e)[:150]}")

# ============================================================================
# ============================================================================
# SECTION 8: HACKATHON SINGLE ERROR DETECTION
# ============================================================================

st.markdown('<div class="section-title">08 HACKATHON: SINGLE ERROR (SELF-INTERSECTION)</div>', unsafe_allow_html=True)

st.markdown("Focus on one error type only: **self-intersecting polygons**. Uses training data from `training\\good_examples.wkt`.")

hack_col1, hack_col2 = st.columns([2, 1])

with hack_col1:
    hack_source = st.radio("Input source", ["Use provided error examples", "Upload WKT file"], index=0, key="hack_source")
    hack_using_default = False
    if hack_source == "Upload WKT file":
        hack_upload = st.file_uploader("Upload WKT (.wkt/.txt)", type=["wkt", "txt"], key="hack_upload")
        hack_input_text = hack_upload.read().decode("utf-8") if hack_upload else ""
    else:
        if HACK_ERROR_EXAMPLES.exists():
            hack_input_text = HACK_ERROR_EXAMPLES.read_text(encoding="utf-8")
            if not hack_input_text.strip():
                hack_input_text = HACK_DEFAULT_BAD
                hack_using_default = True
        else:
            hack_input_text = HACK_DEFAULT_BAD
            hack_using_default = True

with hack_col2:
    if "hack_auto_added" not in st.session_state:
        st.session_state.hack_auto_added = None
    if st.button("Auto Add Examples (Good + Bad)", key="hack_auto_add"):
        good_count, bad_count = hack_write_default_examples()
        st.session_state.hack_auto_added = (good_count, bad_count)
    if st.session_state.hack_auto_added:
        good_count, bad_count = st.session_state.hack_auto_added
        st.success(f"Added {good_count} good and {bad_count} bad examples.")

    st.markdown("**Training file**")
    st.code(str(HACK_TRAIN_GOOD), language="text")
    if HACK_TRAIN_GOOD.exists():
        good_rows, good_errors = hack_load_wkts_from_text(HACK_TRAIN_GOOD.read_text(encoding="utf-8"))
        st.write(f"Good examples: {len(good_rows)}")
        if good_errors:
            st.warning("Training file has parse errors; they are ignored.")
    else:
        st.warning("Training file missing.")

if not hack_input_text:
    if hack_source == "Upload WKT file":
        st.info("Provide a WKT file or use the provided error examples to run this check.")
        st.stop()
    st.warning("Provided error examples are empty. Click Auto Add Examples (Good + Bad).")
    st.stop()
if hack_source == "Use provided error examples" and hack_using_default:
    st.info("Using built-in bad examples because the provided file was empty or missing.")
else:
    hack_rows, hack_parse_errors = hack_load_wkts_from_text(hack_input_text)
    if hack_parse_errors:
        st.warning("Some lines failed to parse. They were skipped.")
        st.json(hack_parse_errors)

    if hack_rows:
        hack_model, hack_scaler = (None, None)
        if HACK_TRAIN_GOOD.exists():
            good_rows, _ = hack_load_wkts_from_text(HACK_TRAIN_GOOD.read_text(encoding="utf-8"))
            hack_model, hack_scaler = hack_train_model(good_rows)

        hack_report = []
        for row in hack_rows:
            geom = row["geom"]
            if not isinstance(geom, Polygon):
                continue
            is_err, loc = hack_self_intersection_info(geom)
            score = None
            if hack_model is not None:
                f = hack_geom_features(geom)
                if f is not None:
                    X = hack_scaler.transform([f])
                    score = float(hack_model.decision_function(X)[0])
            if is_err:
                hack_report.append({"id": row["id"], "error_type": "Self-Intersection", "location": loc or "unknown", "ml_score": score, "wkt": row["wkt"]})

        if not hack_report:
            st.success("No self-intersection errors detected in polygon geometries.")
        else:
            hack_df = pd.DataFrame(hack_report)
            st.error(f"Detected {len(hack_df)} self-intersection error(s)")
            st.dataframe(hack_df, use_container_width=True)

            st.markdown("**Simple Error Report**")
            hack_json_report = json.dumps(hack_report, indent=2)
            st.download_button("Download JSON Report", data=hack_json_report, file_name="self_intersection_report.json", mime="application/json")

# ============================================================================
# SECTION 9: EXPORTS
# ============================================================================

st.markdown('<div class="section-title">09 EXPORT REPORTS & DATA</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.geometries:
        wkt_data = "\n".join([g.wkt for g in st.session_state.geometries])
        st.download_button(
            label="📄 WKT",
            data=wkt_data,
            file_name=f"geometries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wkt",
            mime="text/plain",
            use_container_width=True
        )

with col2:
    if st.session_state.errors:
        json_data = json.dumps(st.session_state.errors, indent=2)
        st.download_button(
            label="🔴 ERRORS",
            data=json_data,
            file_name=f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

with col3:
    if st.session_state.fixes:
        json_fixes = json.dumps(st.session_state.fixes, indent=2)
        st.download_button(
            label="✅ FIXES",
            data=json_fixes,
            file_name=f"fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

with col4:
    if st.session_state.geometries:
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_geometries': len(st.session_state.geometries),
            'total_errors': len(st.session_state.errors),
            'total_fixes': len(st.session_state.fixes),
            'vision_analyses': len(st.session_state.vision_results),
            'geometries': [g.wkt for g in st.session_state.geometries],
            'errors': st.session_state.errors,
            'fixes': st.session_state.fixes,
            'vision_results': st.session_state.vision_results
        }
        json_report = json.dumps(report, indent=2)
        st.download_button(
            label="📊 FULL",
            data=json_report,
            file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# ============================================================================
# POWER FOOTER
# ============================================================================

st.markdown("""
<div style="text-align: center; margin-top: 60px; padding: 30px; border-top: 3px solid #ff3333; color: #888;">
    <p style="font-size: 11px; letter-spacing: 2px; text-transform: uppercase; margin: 0;">
    ⚡ ULTRA POWER GEOMETRY QA v2.0 • Gemini + Oxlo • ML + Vision + Vector QA • Enterprise Grade ⚡
    </p>
</div>
""", unsafe_allow_html=True)
