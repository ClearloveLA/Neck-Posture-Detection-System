from flask import Flask, Response, render_template, jsonify
import cv2
from demo import myDetect
import threading

app = Flask(__name__)

class VideoStream:
    def __init__(self):
        self.frame = None
        self.status = "未开始"
        self.angle = 0
        self.calibrating = False
        self.detection = None

video_stream = VideoStream()

def gen_frames():
    while True:
        if video_stream and video_stream.detection and video_stream.detection.frame is not None:
            ret, buffer = cv2.imencode('.jpg', video_stream.detection.frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global video_stream
    if not video_stream.detection:
        video_stream.detection = myDetect("detection", None)
        video_stream.detection.start()
    return "检测已启动"

@app.route('/get_status')
def get_status():
    if video_stream.detection:
        return jsonify({
            'status': video_stream.detection.status,
            'angle': video_stream.detection.angle,
            'calibrating': video_stream.detection.calibrating
        })
    return jsonify({
        'status': "未开始",
        'angle': 0,
        'calibrating': False
    })
@app.route('/set_remind_interval', methods=['POST'])
def set_remind_interval():
    interval = request.json.get('interval', 30)
    if video_stream and video_stream.detection:
        video_stream.detection.set_remind_interval(interval)
    return jsonify({'success': True})

@app.route('/toggle_sound', methods=['POST'])
def toggle_sound():
    enabled = request.json.get('enabled', True)
    if video_stream and video_stream.detection:
        video_stream.detection.toggle_sound(enabled)
    return jsonify({'success': True})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    if video_stream and video_stream.detection:
        video_stream.detection.stop()
    return jsonify({'success': True})
if __name__ == '__main__':
    app.run(debug=True)