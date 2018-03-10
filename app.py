#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, Response, request
from flask_bootstrap import Bootstrap
import camera_opencv
from faceAnalyse import FaceDetector
import cv2
from flask_socketio import SocketIO, emit
from threading import Lock

# global variable
Camera = camera_opencv.Camera
detector = FaceDetector()
detector.load_image_from_folder('../../../img/img_people')

bAnalyze = False
bVideo = True
faceData = {}
thread = None
thread_lock = Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
bootstrap = Bootstrap(app)
socketio = SocketIO(app, async_mode=None)



# カメラのON/OFF関数
def camera_handle(bool):
    pass


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.route("/", methods=['GET', 'POST'])
def index():
    global bAnalyze, bVideo, faceData
    if request.method == 'POST':
        print(request.form.get('analyze'))
        if request.form.get('analyze') == 'on':
            bAnalyze = True
            print('analyze start')
        elif request.form.get('analyze') == 'off':
            bAnalyze = False
            print('analyze stop')

        if request.form.get('video') == 'start':
            camera_handle(True)
            print('video start')
        elif request.form.get('video') == 'stop':
            camera_handle(False)
            print('video stop')

    elif request.method == 'GET':
        bAnalyze = False
        bVideo = True

    return render_template('index.html',
                           title='VideoStreaming',
                           analyze_running=bAnalyze,
                           video_running=bVideo,
                           async_mode=socketio.async_mode)


@socketio.on('connect', namespace='/face')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=face_send)
    emit('face_response', {'data': 'Connected'})


def face_send():
    info_iterator = gen_data()
    for data in info_iterator:
        print('socketio emit')
        socketio.emit('face_response',
                      {'data': data},
                      namespace='/face')

def gen_data():
    print('yield face data')
    if len(faceData) != 0:
        yield faceData['face']


def gen_camera(camera):
    """Video streaming generator function."""
    while True:
        # get every frame
        frame = camera.get_frame()

        # decide whether anlyze face or not
        global bAnalyze, faceData
        if bAnalyze:
            faceData = detector.analyze_faces_in_image(frame)
            frame = detector.draw_rect(frame, faceData)

        # resize and encode image
        image = cv2.resize(frame, None, fx=0.5, fy=0.5)
        image = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + image + b'\r\n')


@app.route('/video-feed')
def video_feed():
    return Response(gen_camera(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, debug=True)
