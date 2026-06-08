import socket
import cv2
import json
import struct
import time
import os
import subprocess
import threading

# --- CONFIGURATION ---
SERVER_IP = 'DESKTOP-BGITREO.local' 
SERVER_PORT = 65432

# --- PI 5 HARDWARE SETUP ---
os.environ['GPIOZERO_PIN_FACTORY'] = 'lgpio'
from gpiozero import DigitalInputDevice, Motor, Servo
from gpiozero.pins.lgpio import LGPIOFactory

factory = LGPIOFactory()

# --- SENSOR & MOTOR PINS ---
ir_sensor = DigitalInputDevice(22, pin_factory=factory)        
moisture_sensor = DigitalInputDevice(21, pin_factory=factory)  
conveyor = Motor(forward=17, backward=27, pin_factory=factory)

# --- NEW: SERVO MOTOR ---
# GPIO 18 for Servo PWM
servo = Servo(18, pin_factory=factory)
servo.min() # Set to starting position

# Global State
belt_paused = False

def listen_to_server(sock):
    """Listens for commands from the PC (Pause belt, move servo)"""
    global belt_paused
    buffer = ""
    while True:
        try:
            data = sock.recv(1024).decode('utf-8')
            if not data: break
            buffer += data
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                cmd = json.loads(line)
                
                if cmd.get("command") == "pause_toggle":
                    belt_paused = cmd.get("state")
                    if belt_paused:
                        print("⏸️ Belt Paused by Web UI")
                        conveyor.stop()
                    else:
                        print("▶️ Belt Resumed by Web UI")
                        conveyor.forward()
                        time.sleep(5)    
                elif cmd.get("command") == "servo_knock":
                    print("🦾 Inorganic Detected! Knocking waste to the right...")
                    time.sleep(5)
                    servo.max() # Move arm to right
                    time.sleep(1.5) # Wait for it to push
                    servo.min() # Bring arm back
        except:
            break

def take_photo():
    try:
        cmd = [
            "rpicam-still", "-o", "temp_capture.jpg",
            "-t", "1", "--width", "640", "--height", "480", "-n"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return cv2.imread("temp_capture.jpg")
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return None

def send_data(sock, image, moisture_status):
    if image is None: return
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    
    metadata = {"moisture": moisture_status}
    meta_json = json.dumps(metadata).encode('utf-8')
    
    sock.sendall(struct.pack('>I', len(meta_json)))
    sock.sendall(meta_json)
    sock.sendall(struct.pack('>I', len(img_bytes)))
    sock.sendall(img_bytes)

def main():
    global belt_paused
    print(f"🔌 Connecting to PC at {SERVER_IP}...")
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((SERVER_IP, SERVER_PORT))
                print("✅ Connected to Server! Starting conveyor belt...")
                
                # Start listener thread
                threading.Thread(target=listen_to_server, args=(s,), daemon=True).start()
                
                conveyor.forward()
                belt_paused = False
                last_trigger_time = 0
                
                while True:
                    if ir_sensor.value == 0 and not belt_paused:
                        if time.time() - last_trigger_time > 5.0:
                            print("🚨 Object Detected! Stopping conveyor...")
                            conveyor.stop()
                            time.sleep(0.5) 
                            
                            moist_val = "Wet" if moisture_sensor.value == 0 else "Dry"
                            frame = take_photo()
                            
                            if frame is not None:
                                print(f"📤 Sending Data...")
                                send_data(s, frame, moist_val)
                                last_trigger_time = time.time()
                            
                            # Give PC time to process and trigger servo if needed
                            time.sleep(2) 
                            
                            if not belt_paused:
                                print("▶️ Resuming conveyor belt...")
                                conveyor.forward()
                                
                    time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Connection failed: {e}. Retrying in 5s...")
            conveyor.stop() 
            time.sleep(5)

if __name__ == "__main__":
    main()