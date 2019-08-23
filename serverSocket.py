#!/usr/bin/env python
from utils import (
    load_model, get_face,
    get_faces_live,
    forward_pass,
    save_embedding,
    load_embeddings,
    identify_face,
    allowed_file,
    remove_file_extension,
    save_image
)
from threading import Event, Thread, Lock

from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect

from flask_cors import CORS
import os
import cv2
import tensorflow as tf
import detect_face  # for MTCNN face detection
import requests
# import grequests
import json
import base64
#from skimage import io
from scipy.misc import imread
import logging
import time
from gevent import monkey
monkey.patch_all()

d = 0

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.

# async_mode = None
# async_mode = 'eventlet'
async_mode = 'gevent'
no_of_client = 0
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


CORS(app, supports_credentials=True)



params = {
	'ping_timeout': 10,
	'ping_interval': 5
}

socketio = SocketIO(app, async_mode=async_mode, **params)
thread = None
thread_lock = Lock()
stop_threads = False


token = None

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
embeddings_path = os.path.join(APP_ROOT, 'embeddings')
allowed_set = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload

cancel_call_repeatedly = None


def call_repeatedly(interval, func, *args):
    stopped = Event()

    def loop():
        while not stopped.wait(interval):  # the first call is in `interval` secs
            func(*args)
    Thread(target=loop).start()
    return stopped.set


def processFrame():
    global token
    # try:
    #     print("in-----processFrame------")
    if token == None:
        token = get_ms_token()

    frame = get_video_frame()

    if frame["data"] == None:
        print("VIDEO-FRAME-ERROR", frame)
        return None

    file = os.path.join(APP_ROOT, 'video_frames/imageToSave.png')
    imgdata = base64.b64decode(frame["data"]["getLastFrame"])
    with open(file, "wb") as fh:
        fh.write(imgdata)

    # file = os.path.join(APP_ROOT, 'uploads/Mahmoud Abdel Wahab.jpg')
    result = predict_image(file)
    print("predict_image", result)
    emit('got_face', {'data': result}, namespace='/test')

    # except ConnectionError:
    #     print("***********************An exception occurred************************")
    # except:
    #     print("***********************An exception occurred************************")

def background_stuff():
    global token, stop_threads
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        try:
            
            if stop_threads: 
                break
            time.sleep(1)
            app.logger.info('-----START=====>background_stuff------%s', str(time.clock()))
            count += 1
            if token == None:
                token = get_ms_token()

            frame = get_video_frame()

            if frame["data"] == None:
                print("VIDEO-FRAME-ERROR", frame)
                continue
            file = os.path.join(APP_ROOT, 'video_frames/imageToSave.png')
            imgdata = base64.b64decode(frame["data"]["getLastFrame"])
            with open(file, "wb") as fh:
                fh.write(imgdata)

            # file = os.path.join(APP_ROOT, 'uploads/Mahmoud Abdel Wahab.jpg')
            result = predict_image(file)
            app.logger.info('-----predict_image------%s', str(result))
            if result != None:
                socketio.emit('got_face', {'data': result, 'count': count}, namespace='/test')
            app.logger.info('-----END=====>background_stuff------%s', str(time.clock()))

        except requests.exceptions.ConnectionError as e:
            handle_http_errors("*******REQUESTS-ConnectionError*******=>",e)
        except requests.exceptions.Timeout as e:
            handle_http_errors("*******REQUESTS-Timeout*******=>",e)
        except requests.exceptions.TooManyRedirects as e:
            handle_http_errors("*******REQUESTS-TooManyRedirects*******=>",e)
        except requests.exceptions.RequestException as e:
            handle_http_errors("*******REQUESTS-RequestException*******=>",e)
        except Exception as e:
            handle_http_errors("***********************UNKNOWN exception occurred************************",e)

def stop_background_stuff():
    global thread, stop_threads
    stop_threads = True
    # thread.join() 
    print('thread killed')
    app.logger.info('===&&====THREAD KILLED===&&====')

def handle_http_errors(msg, err=None):
    socketio.emit('got_face',
        {'error': [msg]},
        namespace='/test')
    print(msg, err)
    app.logger.info('=======HTTP_ERROR=======>%s', str(msg))

def background_thread():
    global token
    # global socketio
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        try:
            socketio.sleep(3)
            print("in-----background_thread------")
            logging.info("logging.info::-in-----background_thread------")

            count += 1
            if token == None:
                token = get_ms_token()

            frame = get_video_frame()

            if frame["data"] == None:
                print("VIDEO-FRAME-ERROR", frame)
                continue
            file = os.path.join(APP_ROOT, 'video_frames/imageToSave.png')
            imgdata = base64.b64decode(frame["data"]["getLastFrame"])
            with open(file, "wb") as fh:
                fh.write(imgdata)

            # file = os.path.join(APP_ROOT, 'uploads/Mahmoud Abdel Wahab.jpg')
            result = predict_image(file)
            print("predict_image", result)
            logging.info("logging.info::--------------predict_image------")
            # print(result)
            socketio.emit('got_face',
                          {'data': result, 'count': count},
                          namespace='/test')

        except:
            socketio.emit('got_face',
                          {'data': ["ERROR"], 'count': count},
                          namespace='/test')
            logging.info(
                "logging.info::--------------An exception occurred------")
            print("***********************An exception occurred************************")

def get_video_frame():
    global token
    global proxyDict

    url = "https://gateway.eu1.mindsphere.io/api/streamsvc-riyaddev/v1/video"
    # url = "http://192.168.2.117:8081/video"

    payload = "{\"operationName\":null,\"variables\":{},\"query\":\"{getLastFrame(camId: \\\"3\\\")}\"}"
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer " + token['access_token'],
        'cache-control': "no-cache"
    }

    response = requests.request("POST",
                                url,
                                data=payload,
                                headers=headers,
                                # proxies=proxyDict,
                                timeout=15,
                                verify=False)

    # print("response.text",response.text)
    if response.status_code == 401:
        token = get_ms_token()
        return get_video_frame()
    return response.json()


def get_video_frame1():
    global token
    print("token['access_token']", token['access_token'])
    # For this agent we have already created the agent in MindSphere and will be just registring it.
    resp = requests.post('https://gateway.eu1.mindsphere.io/api/streamsvc-riyaddev/v1/video',
                         data='{"operationName":null,"variables":{},"query":"{getLastFrame(camId: \"1\")}"}',
                         #  data=json.dumps({"operationName":None,"variables":{},"query":"{getLastFrame(camId: \"1\")}"}),
                         headers={
                             'Content-Type': 'application/json',
                             'Accept': 'application/json',
                             'Authorization': "Bearer " + token['access_token']},
                         verify=False)
    # 'Host': 'https://southgate.eu1.mindsphere.io'})
    if resp.status_code != 201:
        print('POST /tasks/ {}'.format(resp.status_code))
        # print(resp.json())
    if resp.status_code == 401:
        print("-------------------token-expired-------------------")
        token = get_ms_token()
        return get_video_frame()
        # print(resp.json())

    # print('raw Jason response, Step 1:')
    # print(str(resp.json()))
    # print("-------------------------------")
    # print('Created task. ID: {}'.format(resp.json()["id"]))

    return resp.json()


def get_ms_token():
    # For this agent we have already created the agent in MindSphere and will be just registring it.
    proxyDict = {
        "http": 'http://194.138.0.25:9400',
        "https": 'http://194.138.0.25:9400',
                # "ftp"   : 'http://194.138.0.25:9400'
    }

    resp = requests.get('https://riyaddev.piam.eu1.mindsphere.io/oauth/token?grant_type=client_credentials',
                        headers={
                            # 'Content-Type': 'application/json',
                            # 'Accept': 'application/json',
                            'Authorization': 'Basic bm90aWZpY2F0aW9uU2VydmljZUFkbWluM3JkUGFydHlUZWNoVXNlcjoxOTAwMDEzYy1jMTdmLTRiZTYtOGQ5MC05ODI0NWYwODZlODc='
                        },
                        # proxies=proxyDict,
                        verify=False)
    if resp.status_code != 200:
        # print('POST /tasks/ {}'.format(resp.status_code))
        print(resp.json())

    # print('raw Jason response, Step 1:')
    # print(resp.json())
    # print(resp)
    # print("-------------------------------")
    # print('Created task. ID: {}'.format(resp.json()["id"]))

    return resp.json()


def predict_image(file):
    # file = request.files['file']
    # file = os.path.join(APP_ROOT, 'uploads/Abdulrahman Safh.png')
    # Read image file as numpy array of RGB dimension
    #img = io.imread(fname=file)
    img = imread(name=file, mode='RGB')
    # Detect and crop a 160 x 160 image containing a human face in the image file
    faces, rects = get_faces_live(img=img, pnet=pnet, rnet=rnet,
                           onet=onet, image_size=image_size)
    #global d
    # If there are human faces detected
    if faces:
        embedding_dict = load_embeddings()
        if embedding_dict:
            people_found = []
            for i in range(len(faces)):
                face_img = faces[i]
                rect = rects[i]

                face_embedding = forward_pass(
                    img=face_img, session=facenet_persistent_session,
                    images_placeholder=images_placeholder, embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )

                # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
                identity = identify_face(
                    embedding=face_embedding, embedding_dict=embedding_dict)
                people_found.append(identity)

                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 3)

                W = int(rect[2] - rect[0]) // 2
                H = int(rect[3] - rect[1]) // 2

                cv2.putText(img, identity, (rect[0] + W - (W // 2), rect[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

            # code for saving the output images
            # cv2.imwrite("SavedImgesFull/file_%d.jpg" % d, img)
            #d += 1
            return people_found

        else:
            # return ["No Face"]
            return None
            # return render_template(
            #     'predict_result.html',
            #     identity="No embedding files detected! Please upload image files for embedding!"
            # )
    else:
        # return ["No Image"]
        return None
        # return render_template(
        #     'predict_result.html',
        #     identity="Operation was unsuccessful! No human face was detected."
        # )


@app.route('/')
def index():
    return render_template('socket-test-client.html', async_mode=socketio.async_mode)


@app.route('/add-face')
def add_face():
    return render_template("index.html")


@app.route("/home")
def index_page():
    return render_template("index.html")


@app.route('/push-video-frame', methods=['POST', 'GET'])
def push_video_frame():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No selected file"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            filePath = os.path.join(APP_ROOT, 'video_frames/pushedVideoFrame.png')
            file.save(filePath)
            result = predict_image(filePath)
            app.logger.info('-----predict_image------%s', str(result))
            if result != None:
                socketio.emit('got_face', {'data': result}, namespace='/test')
            app.logger.info('-----END=====>background_stuff------%s', str(time.clock()))
            return "Success"

    else:
        return "POST HTTP method required!"

@app.route('/upload', methods=['POST', 'GET'])
def get_image():

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No selected file"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            # Read image file as numpy array of RGB dimension
            #img = io.imread(fname=file, mod='RGB')
            img = imread(name=file, mode='RGB')
            # Detect and crop a 160 x 160 image containing a human face in the image file
            img = get_face(img=img, pnet=pnet, rnet=rnet,
                           onet=onet, image_size=image_size)

            # If a human face is detected
            if img is not None:

                embedding = forward_pass(
                    img=img, session=facenet_persistent_session,
                    images_placeholder=images_placeholder, embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )
                # Save cropped face image to 'uploads/' folder
                save_image(img=img, filename=filename,
                           uploads_path=uploads_path)
                # Remove file extension from image filename for numpy file storage being based on image filename
                filename = remove_file_extension(filename=filename)
                # Save embedding to 'embeddings/' folder
                save_embedding(embedding=embedding, filename=filename,
                               embeddings_path=embeddings_path)

                return render_template("upload_result.html",
                                       status="Image uploaded and embedded successfully!")

            else:
                return render_template("upload_result.html",
                                       status="Image upload was unsuccessful! No human face was detected.")

    else:
        return "POST HTTP method required!"


@socketio.on('my_event', namespace='/test')
def test_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})


@socketio.on('my_broadcast_event', namespace='/test')
def test_broadcast_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         broadcast=True)


@socketio.on('join', namespace='/test')
def join(message):
    join_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'count': session['receive_count']})


@socketio.on('leave', namespace='/test')
def leave(message):
    leave_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'count': session['receive_count']})


@socketio.on('close_room', namespace='/test')
def close(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response', {'data': 'Room ' + message['room'] + ' is closing.',
                         'count': session['receive_count']},
         room=message['room'])
    close_room(message['room'])


@socketio.on('my_room_event', namespace='/test')
def send_room_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         room=message['room'])


@socketio.on('disconnect_request', namespace='/test')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)


@socketio.on('my_ping', namespace='/test')
def ping_pong():
    emit('my_pong')


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread, stop_threads, no_of_client
    no_of_client += 1
    # if thread is None:
    if no_of_client == 1:
        stop_threads = False
        thread = Thread(target=background_stuff)
        thread.start()
    emit('my_response', {'data': 'Connected', 'count': 0})
    app.logger.info('===CONNECTED====NO_OF_CLIENT=======>%s', str(no_of_client))

# @socketio.on('connect', namespace='/test')
# def test_connect():
#     global thread
#     # global socketio
#     with thread_lock:
#         if thread is None:
#             thread = socketio.start_background_task(background_thread)
#     emit('my_response', {'data': 'Connected', 'count': 0})


# @socketio.on('connect', namespace='/test')
# def test_connect():
#     global no_of_client
#     global cancel_call_repeatedly
#     no_of_client += 1
#     if no_of_client == 1:
#         cancel_call_repeatedly = call_repeatedly(3, processFrame)

#     emit('my_response', {'data': 'NO OF CLIENT', 'count': no_of_client})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    global no_of_client
    # global cancel_call_repeatedly
    no_of_client -= 1
    if no_of_client == 0:
        stop_background_stuff()
        # cancel_call_repeatedly()
    # print('Client disconnected', request.sid)
    app.logger.info('===CLIENT DISCONNECTED==%s==NO_OF_CLIENT=======>%s', request.sid, str(no_of_client))


@socketio.on_error_default  # handles all namespaces without an explicit error handler
def default_error_handler(e):
    print(e)
    pass


if __name__ == '__main__':
    # Load FaceNet model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
    model_path = 'model/MTCNN.pb'
    facenet_model = load_model(model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    image_size = 160
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Initiate persistent FaceNet model in memory
    facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

    # Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
    pnet, rnet, onet = detect_face.create_mtcnn(
        sess=facenet_persistent_session, model_path=None)

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    # token = get_ms_token()
