# face-detection-webui

**A minimal face detection system with live webcam stream and web-based configuration.**  
Powered by Python, OpenCV and face_recognition, this tool runs locally and requires no cloud or internet connection.

## Features

- Live webcam stream via browser (MJPEG)
- Local face recognition using `face_recognition`
- Add new persons via Web UI (photo capture + training)
- Configure frame resolution and detection interval in real time
- Works fully offline on Linux / Raspberry Pi

## Requirements

- Python 3.7+
- A USB webcam supported by OpenCV
- `pip install` dependencies:

```bash
pip install flask opencv-python face_recognition numpy
```

> On Raspberry Pi, additional system packages may be required to build `dlib` (used by `face_recognition`).

## Usage

```bash
python3 webcam_webgui.py
```

Then open [http://localhost:5000](http://localhost:5000) or use your Pi's IP in a browser.

## File structure

```
face-detection-webui/
├── webcam_webgui.py
├── known_faces/            # Photos and encodings per person
└── templates/
    └── index.html          # Web UI layout
```

## License

This project is licensed under the [MIT License](LICENSE).