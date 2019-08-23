import cv2
import tensorflow as tf
import os
import pyttsx3
import random
from skimage import io
import detect_face  # for MTCNN face detection
from flask import Flask, request, render_template
from utils import (
    load_model, get_face, get_faces_live, forward_pass, save_embedding, load_embeddings,
    identify_face, allowed_file, remove_file_extension, save_image
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
embeddings_path = os.path.join(APP_ROOT, 'embeddings')
allowed_set = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload


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
            img = io.imread(fname=file, mode='RGB')
            # Detect and crop a 160 x 160 image containing a human face in the image file
            img = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)

            # If a human face is detected
            if img is not None:

                embedding = forward_pass(
                    img=img, session=facenet_persistent_session,
                    images_placeholder=images_placeholder, embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )
                # Save cropped face image to 'uploads/' folder
                save_image(img=img, filename=filename, uploads_path=uploads_path)
                # Remove file extension from image filename for numpy file storage being based on image filename
                filename = remove_file_extension(filename=filename)
                # Save embedding to 'embeddings/' folder
                save_embedding(embedding=embedding, filename=filename, embeddings_path=embeddings_path)

                return render_template("upload_result.html",
                                       status="Image uploaded and embedded successfully!")

            else:
                return render_template("upload_result.html",
                                       status="Image upload was unsuccessful! No human face was detected.")

    else:
        return "POST HTTP method required!"


@app.route('/predictImage', methods=['POST', 'GET'])
def predict_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No selected file"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            # Read image file as numpy array of RGB dimension
            img = imread(name=file, mode='RGB')
            # Detect and crop a 160 x 160 image containing a human face in the image file
            img = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)

            # If a human face is detected
            if img is not None:

                embedding = forward_pass(
                    img=img, session=facenet_persistent_session,
                    images_placeholder=images_placeholder, embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )

                embedding_dict = load_embeddings()
                if embedding_dict:
                    # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
                    identity = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                    return render_template('predict_result.html', identity=identity)

                else:
                    return render_template(
                        'predict_result.html',
                        identity="No embedding files detected! Please upload image files for embedding!"
                    )

            else:
                return render_template(
                    'predict_result.html',
                    identity="Operation was unsuccessful! No human face was detected."
                )
    else:
        return "POST HTTP method required!"


@app.route("/live", methods=['GET', 'POST'])
def face_detect_live():
    # Load text reading engine
    #engine = pyttsx3.init()
    spoken_face_names = []
    greetings = ['How do you do', 'Hello', 'Hi', 'Hai', 'Hey', 'How have you been', 'How are you',
                 'How is it going', 'Salam alikom ', 'Esh loonak ya', 'Ahlaaaan']

    embedding_dict = load_embeddings()
    if embedding_dict:
        try:
            cap = cv2.VideoCapture(0)

            while True:
                return_code, frame = cap.read()  # RGB frame

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                faces, rects = get_faces_live(img=frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                    # If there are human faces detected
                if faces:
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
                        identity = identify_face(embedding=face_embedding, embedding_dict=embedding_dict)

                        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 215, 0), 2)

                        W = int(rect[2] - rect[0]) // 2
                        H = int(rect[3] - rect[1]) // 2

                        cv2.putText(frame, identity, (rect[0]+W-(W//2), rect[1]-7),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1, cv2.LINE_AA)

                        if identity == "Unknown":
                            continue
                        elif identity in spoken_face_names:
                            continue
                        else:
                            print(random.choice(greetings) + " " + identity)
                            #engine.say(random.choice(greetings) + name)
                            #engine.runAndWait()
                            spoken_face_names.append(identity)
                            continue

                    cv2.imshow('Video', frame)
                else:
                    continue

            cap.release()
            cv2.destroyAllWindows()
            return render_template('index.html')
        except Exception as e:
            print(e)
    else:
        return "No loaded faces detected! Please upload image files for embedding!"


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


if __name__ == '__main__':
    """Server and FaceNet Tensorflow configuration."""

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
    pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

    # Start flask application
    app.run()
