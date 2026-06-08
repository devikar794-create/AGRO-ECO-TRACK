import socket
import threading
import json
import base64
import cv2
import numpy as np
import struct
import sqlite3
import os
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO

HOST_IP = '0.0.0.0'
PORT = 65432
DB_NAME = 'waste_carbon_footprint.db'

current_state = {
    "status": "Waiting for connection...",
    "object_detected": False,
    "classification": "None",
    "carbon_data": [],
    "image_data": None,
    "locked": False,
    "belt_paused": False # NEW STATE
}

active_pi_connection = None # Tracks the socket connection to send commands

app = Flask(__name__)
CORS(app)

print("⏳ Loading AI Models...")
try:
    primary_model = YOLO('best.pt')       
    organic_model = YOLO('2nd model.pt')  
    print("✅ Models Loaded!")
except Exception as e:
    exit()

def get_all_disposal_methods(waste_name):
    if not os.path.exists(DB_NAME): return []
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT disposal_method, avg_factor FROM waste_emissions WHERE waste_name = ? COLLATE NOCASE", (waste_name,))
        rows = cursor.fetchall()
        conn.close()
        return [{"method": r["disposal_method"], "factor": r["avg_factor"]} for r in rows]
    except: return []

def run_socket_server():
    global current_state, active_pi_connection
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST_IP, PORT))
        s.listen()
        print(f"📡 Socket Server listening...")
        
        while True:
            try:
                conn, addr = s.accept()
                active_pi_connection = conn
                with conn:
                    current_state["status"] = "Pi Connected"
                    
                    while True:
                        meta_len_data = conn.recv(4)
                        if not meta_len_data: break
                        meta_len = struct.unpack('>I', meta_len_data)[0]
                        
                        metadata_bytes = b""
                        while len(metadata_bytes) < meta_len:
                            packet = conn.recv(meta_len - len(metadata_bytes))
                            metadata_bytes += packet
                        metadata = json.loads(metadata_bytes.decode('utf-8'))
                        
                        img_len_data = conn.recv(4)
                        img_len = struct.unpack('>I', img_len_data)[0]
                        
                        img_bytes = b""
                        while len(img_bytes) < img_len:
                            packet = conn.recv(min(4096, img_len - len(img_bytes)))
                            img_bytes += packet

                        if current_state["locked"]:
                            continue 

                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        results = primary_model(img, conf=0.25)
                        primary_result = results[0]
                        
                        annotated_frame = primary_result.plot()
                        
                        carbon_data = []
                        detected_labels = set()
                        organic_detected = False

                        if len(primary_result.boxes) > 0:
                            for box in primary_result.boxes:
                                class_id = int(box.cls[0])
                                label = primary_result.names[class_id]
                                detected_labels.add(label)
                                if label == "organic":
                                    organic_detected = True

                            if organic_detected:
                                sec_results = organic_model(img, conf=0.25)
                                sec_result = sec_results[0]
                                
                                if len(sec_result.boxes) > 0:
                                    best_box = max(sec_result.boxes, key=lambda b: b.conf[0])
                                    best_raw_label = sec_result.names[int(best_box.cls[0])]
                                    best_specific_label = best_raw_label.strip().title()
                                    
                                    new_carbon_data = get_all_disposal_methods(best_specific_label)
                                    if new_carbon_data: carbon_data.extend(new_carbon_data)

                                    specific_items = set()
                                    for sec_box in sec_result.boxes:
                                        specific_label = sec_result.names[int(sec_box.cls[0])].strip().title()
                                        specific_items.add(specific_label)
                                        x1, y1, x2, y2 = map(int, sec_box.xyxy[0])
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        (w, h), _ = cv2.getTextSize(specific_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                                        cv2.putText(annotated_frame, specific_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                                    if "organic" in detected_labels:
                                        detected_labels.remove("organic")
                                    detected_labels.update(specific_items)
                                    
                            # --- NEW: INORGANIC SERVO TRIGGER ---
                            elif len(detected_labels) > 0:
                                # We saw something, but NO organic was detected.
                                print("🦾 Inorganic waste identified. Sending Knock command...")
                                try:
                                    cmd = json.dumps({"command": "servo_knock"}) + "\n"
                                    active_pi_connection.sendall(cmd.encode('utf-8'))
                                except Exception as e:
                                    print("Failed to send servo command:", e)

                            final_label = ", ".join(list(detected_labels))
                        else:
                            final_label = "No object detected"

                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        img_base64 = base64.b64encode(buffer).decode('utf-8')

                        current_state.update({
                            "object_detected": True,
                            "classification": final_label,
                            "carbon_data": carbon_data,
                            "image_data": img_base64,
                            "status": "Processed",
                            "locked": True 
                        })
                        
            except Exception as e:
                active_pi_connection = None
                current_state["status"] = "Waiting for connection..."
                current_state["locked"] = False 

threading.Thread(target=run_socket_server, daemon=True).start()

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(current_state)

@app.route('/reset', methods=['POST'])
def reset():
    global current_state
    current_state.update({"object_detected": False, "classification": "None", "carbon_data": [], "image_data": None, "locked": False})
    return jsonify({"status": "reset"})

# --- NEW: PAUSE/RESUME ROUTE ---
@app.route('/toggle_belt', methods=['POST'])
def toggle_belt():
    global current_state, active_pi_connection
    current_state["belt_paused"] = not current_state["belt_paused"]
    
    if active_pi_connection:
        try:
            cmd = json.dumps({"command": "pause_toggle", "state": current_state["belt_paused"]}) + "\n"
            active_pi_connection.sendall(cmd.encode('utf-8'))
        except Exception as e:
            print("Failed to send pause command:", e)
            
    return jsonify({"paused": current_state["belt_paused"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)