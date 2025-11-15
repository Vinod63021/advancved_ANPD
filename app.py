# ==============================================================================
# ALPR Advanced - Flask + Templates + Multiple Vehicle Detection
# ==============================================================================

import os
import time
import uuid
import traceback
import re  # Added for regex
import io  # Added for in-memory file
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import flask
from ultralytics import YOLO
import easyocr
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------------------
# FOLDER STRUCTURE
# ------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
VEHICLE_IMAGES = os.path.join(UPLOAD_FOLDER, 'vehicles')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

for folder in [UPLOAD_FOLDER, DATA_FOLDER, VEHICLE_IMAGES, TEMPLATE_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

LOCAL_VIDEO_PATH = os.path.join(BASE_DIR, 'demo.mp4')
CSV_PATH = os.path.join(DATA_FOLDER, 'vehicles.csv')

AI_FRAME_SKIP = 5
RECENT_TTL_SECONDS = 10
REQUIRED_PLATE_CHARS = 10  # Your constraint

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# ------------------------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------------------------
model_status = {"vehicle_yolo": False, "plate_yolo": False, "ocr": False}

try:
    vehicle_model = YOLO("yolov8n.pt")
    model_status["vehicle_yolo"] = True
    print("✔ Loaded YOLOv8n (vehicle model)")
except Exception as e:
    print(f"❌ Failed to load vehicle model: {e}")
    vehicle_model = None

try:
    plate_model = YOLO("plate_model.pt")
    model_status["plate_yolo"] = True
    print("✔ Loaded plate_model.pt")
except Exception as e:
    print(f"❌ Failed to load plate YOLO: {e}")
    plate_model = None

try:
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    model_status["ocr"] = True
    print("✔ EasyOCR Loaded")
except Exception as e:
    print(f"❌ Failed to load EasyOCR: {e}")
    ocr_reader = None

ALL_MODELS_OPERATIONAL = all(model_status.values())

# ------------------------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------------------------
app = flask.Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024  # 1GB Max Upload

current_state = {
    "fps": 0,
    "recent_records": [],
    "models_loaded": model_status
}

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

# BGR Colors
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

COCO_NAME_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# --- NEW: Regex for Indian Number Plates ---
# Format: MH12AB1234
INDIAN_PLATE_REGEX_1 = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$")
# Format: 24BH1234AA (Bharat Series)
INDIAN_PLATE_REGEX_2 = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    """Cleans text by keeping only alphanumeric characters."""
    return "".join(c for c in text if c.isalnum()).upper()


def validate_plate(text):
    """
    Checks if the plate text is a valid 10-char Indian plate.
    Returns: (isValid, isIndian, series)
    """
    # Basic data integrity: must be 10 chars
    isValid = (len(text) == REQUIRED_PLATE_CHARS)
    isIndian = False
    series = "---"

    if not isValid:
        return isValid, isIndian, series

    # Check against Indian plate formats
    if INDIAN_PLATE_REGEX_1.match(text):
        isIndian = True
        series = text[:2]  # State code (e.g., "MH")
    elif INDIAN_PLATE_REGEX_2.match(text):
        isIndian = True
        series = "BH"  # BH series
    
    # If it's 10 chars but doesn't match, it's valid by length but not recognized as Indian
    return isValid, isIndian, series


def dominant_color(image):
    """Finds the dominant color of a vehicle crop."""
    try:
        img = cv2.resize(image, (80, 80))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        if len(pixels) > 2000:
            px = pixels[np.random.choice(len(pixels), 2000, replace=False)]
        else:
            px = pixels

        kmeans = KMeans(n_clusters=3).fit(px)
        h, s, v = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

        if v < 50: return "black"
        if s < 35 and v > 200: return "white"
        if h < 10 or h >= 160: return "red"
        if 10 <= h < 25: return "orange"
        if 25 <= h < 35: return "yellow"
        if 35 <= h < 85: return "green"
        if 85 <= h < 125: return "blue"
        if 125 <= h < 160: return "purple"
        return "other"
    except:
        return "unknown"


def save_record(timestamp, vtype, color, plate, valid, is_indian, series, imgpath):
    """Saves the detected vehicle record to a CSV file."""
    # --- UPDATED: Added new columns ---
    new_columns = ["timestamp", "vehicle_type", "color", "plate", 
                   "plate_valid_len", "is_indian", "series", "image_path"]
    df = pd.DataFrame(
        [[timestamp, vtype, color, plate, valid, is_indian, series, imgpath]],
        columns=new_columns
    )
    
    file_exists = os.path.exists(CSV_PATH)
    if file_exists:
        # Check if columns match, if not, force rewrite header
        try:
            existing_cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
            if existing_cols != new_columns:
                file_exists = False # Force header write
        except:
             file_exists = False # File is corrupt or empty
             
    df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)


# ------------------------------------------------------------------------------
# FRAME PROCESSOR
# ------------------------------------------------------------------------------
def process_frames():
    if not ALL_MODELS_OPERATIONAL:
        print("❌ Missing models. Streaming blank frames.")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", blank)
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n"

    cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        frame_count += 1
        fps_counter += 1

        run_ai = frame_count % AI_FRAME_SKIP == 0
        detections_display = []  # List to hold all data for drawing

        if run_ai:
            try:
                results = vehicle_model.track(frame, persist=True, verbose=False, conf=0.35)

                if results and results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        vtype = COCO_NAME_MAP.get(cls, "vehicle")

                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        color = dominant_color(crop)

                        # ---- Plate detection ----
                        plate_text = "---"
                        plate_valid = False
                        is_indian = False
                        series = "---"
                        plate_box_global = None
                        box_color = COLOR_RED  # Default to red

                        plate_res = plate_model.predict(crop, conf=0.4)
                        if plate_res and plate_res[0].boxes:
                            px1, py1, px2, py2 = map(int, plate_res[0].boxes.xyxy[0])
                            plate_box_global = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                            plate_crop = crop[py1:py2, px1:px2]

                            if plate_crop.size > 0:
                                ocr_res = ocr_reader.readtext(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY))
                                if ocr_res:
                                    text = clean_text(" ".join(r[1] for r in ocr_res))
                                    plate_text = text if text else "---"
                                    
                                    # --- UPDATED: Get all 3 values ---
                                    plate_valid, is_indian, series = validate_plate(text)
                                    
                                    if plate_valid:
                                        box_color = COLOR_GREEN

                        # Save snapshot
                        uid = str(uuid.uuid4())[:8]
                        imgname = f"veh_{uid}.jpg"
                        imgpath = os.path.join("uploads", "vehicles", imgname)
                        cv2.imwrite(os.path.join(BASE_DIR, imgpath), crop)

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # --- UPDATED: Pass new data to save_record ---
                        save_record(timestamp, vtype, color, plate_text, plate_valid, is_indian, series, imgpath)

                        # --- UPDATED: Add new data to recent_records ---
                        current_state["recent_records"].insert(0, {
                            "timestamp": timestamp,
                            "vehicle_type": vtype,
                            "color": color,
                            "plate": plate_text,
                            "plate_valid": plate_valid,
                            "is_indian": is_indian,
                            "series": series,
                            "image_path": imgpath
                        })
                        
                        # --- UPDATED: Enhance label ---
                        label = f"{vtype} | {color} | {plate_text} ({series})"
                        detections_display.append({
                            "vehicle_box": (x1, y1, x2, y2),
                            "plate_box": plate_box_global,
                            "label": label,
                            "box_color": box_color
                        })

            except Exception as e:
                print(f"Error in detection: {e}")
                traceback.print_exc()

        # ---- Draw ----
        out = frame.copy()
        for det in detections_display:
            x1, y1, x2, y2 = det["vehicle_box"]
            label = det["label"]
            v_color = det["box_color"]
            p_box = det["plate_box"]

            cv2.rectangle(out, (x1, y1), (x2, y2), v_color, 2)
            if p_box:
                cv2.rectangle(out, (p_box[0], p_box[1]), (p_box[2], p_box[3]), v_color, 2)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - h - 10), (x1 + w, y1 - 5), v_color, -1)
            cv2.putText(out, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 2)

        # FPS
        if time.time() - fps_start >= 1.0:
            current_state["fps"] = round(fps_counter / (time.time() - fps_start), 1)
            fps_counter = 0
            fps_start = time.time()

        _, encoded = cv2.imencode(".jpg", out)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n"

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return flask.Response(
        process_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/get_status")
def get_status():
    current_state["recent_records"] = current_state["recent_records"][:50]
    return flask.jsonify(current_state)


@app.route("/download_csv")
def download_csv():
    """
    --- NEW LOGIC ---
    Performs data integrity checks (deduplication)
    and serves an Excel file.
    """
    if not os.path.exists(CSV_PATH):
        return "File not found. No vehicles detected yet.", 404
    
    try:
        # 1. Read the raw data
        # --- THIS IS THE FIX ---
        # Added 'on_bad_lines='skip'' to ignore rows with the wrong
        # number of columns, which is causing your ParserError.
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
        
        # 2. Data Integrity Check: Remove Duplicates
        #    We keep the *last* (most recent) entry for each plate
        #    Make sure the 'plate' column exists before dropping duplicates
        if 'plate' in df.columns:
            df_deduplicated = df.drop_duplicates(subset=['plate'], keep='last')
        else:
            # If 'plate' column is not found (maybe due to bad CSV read)
            # just return the skipped-line data.
            print("Warning: 'plate' column not found, skipping deduplication.")
            df_deduplicated = df

        
        # 3. Create an in-memory Excel file
        output = io.BytesIO()
        output_filename = "vehicle_data_deduplicated.xlsx"
        
        # Note: 'openpyxl' is required, make sure it's in requirements.txt
        df_deduplicated.to_excel(output, index=False, sheet_name="Vehicles", engine='openpyxl')
        output.seek(0)
        
        # 4. Send the file for download
        return flask.send_file(
            output,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        print(f"Error generating download: {e}")
        traceback.print_exc()
        return "Error processing file", 500


@app.route("/upload", methods=["GET", "POST"])
def upload():
    global LOCAL_VIDEO_PATH

    if flask.request.method == "POST":
        file = flask.request.files.get("file")
        if not file or file.filename == "":
            return flask.jsonify({"error": "No file"}), 400
        if not allowed_file(file.filename):
            return flask.jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        LOCAL_VIDEO_PATH = save_path
        print(f"🔄 New video file set: {save_path}")
        # Note: A server restart is required for the generator to pick up the new path
        
        return flask.jsonify({"uploaded": True, "path": save_path})

    return flask.render_template("upload.html")


@app.route("/uploads/vehicles/<path:filename>")
def serve_vehicle_image(filename):
    return flask.send_from_directory(VEHICLE_IMAGES, filename)

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting ALPR Advanced on http://127.0.0.1:5000")
    print("ℹ️ NOTE: After uploading a new video, you must restart the server.")
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    except:
        app.run(host="0.0.0.0", port=5000, threaded=True)