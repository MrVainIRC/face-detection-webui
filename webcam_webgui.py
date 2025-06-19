from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import os
import numpy as np
from datetime import datetime
import face_recognition

app = Flask(__name__)
camera = cv2.VideoCapture(0)

KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []

# Default settings (can be changed via GUI)
frame_resize_width = 320
frame_resize_height = 240
detection_interval = 5
frame_count = 0

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def apply_camera_settings():
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_resize_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_resize_height)

def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        return

    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)

def generate_frames():
    global frame_count
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % detection_interval == 0:
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
        else:
            face_locations = []
            face_encodings = []

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                name = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', width=frame_resize_width, height=frame_resize_height, interval=detection_interval)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    name = request.form.get('name', 'unknown')
    if not name.strip():
        name = "unknown"

    success, frame = camera.read()
    if not success:
        return "Error capturing image"

    folder = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = f"{folder}/{name}_{timestamp}.jpg"
    cv2.imwrite(path, frame)

    load_known_faces()
    return redirect(url_for('index'))

@app.route('/settings', methods=['POST'])
def settings():
    global frame_resize_width, frame_resize_height, detection_interval

    try:
        frame_resize_width = int(request.form.get('width', frame_resize_width))
        frame_resize_height = int(request.form.get('height', frame_resize_height))
        detection_interval = max(1, int(request.form.get('interval', detection_interval)))
    except ValueError:
        pass

    apply_camera_settings()
    return redirect(url_for('index'))

if __name__ == '__main__':
    load_known_faces()
    apply_camera_settings()
    app.run(host='0.0.0.0', port=5000)