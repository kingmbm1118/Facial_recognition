import tensorflow as tf
import numpy as np
import glob
import os
from tensorflow.python.platform import gfile
from facenet import get_model_filenames
from detect_face import detect_face  # face detection
from facenet import load_img
from scipy.misc import imresize, imsave
from collections import defaultdict
from flask import flash
#from skimage import io
#from skimage.transform import resize


def allowed_file(filename, allowed_set):
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set
    return check


def remove_file_extension(filename):
    filename = os.path.splitext(filename)[0]
    return filename


def save_image(img, filename, uploads_path):
    try:
        #io.imsave(os.path.join(uploads_path, filename), arr=np.squeeze(img))
        imsave(os.path.join(uploads_path, filename), arr=np.squeeze(img))
        flash("Image saved!")
    except Exception as e:
        print(str(e))
        return str(e)


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        #print('Model filename: %s' % model_exp)
        with tf.gfile.GFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            return graph
    else:
        #print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        #print('Metagraph file: %s' % meta_file)
        #print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        graph = saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        return graph


def get_face(img, pnet, rnet, onet, image_size):
    # Default constants from the FaceNet repository implementation of the MTCNN
    minsize = 20
    threshold = [0.9, 0.9, 0.9]
    factor = 0.709
    margin = 44
    input_image_size = image_size

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face(
        img=img, minsize=minsize, pnet=pnet, rnet=rnet,
        onet=onet, threshold=threshold, factor=factor
    )

    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            det = np.squeeze(face[0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]: bb[3], bb[0]:bb[2], :]
            #face_img = resize(image=cropped, output_shape=(input_image_size, input_image_size))
            face_img = imresize(arr=cropped, size=(input_image_size, input_image_size), mode='RGB')
            return face_img
    else:
        return None


def get_faces_live(img, pnet, rnet, onet, image_size):
    # Default constants from the FaceNet repository implementation of the MTCNN
    minsize = 20
    threshold = [0.9, 0.9, 0.9]
    factor = 0.709
    margin = 44
    input_image_size = image_size

    faces = []
    rects = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face(
        img=img, minsize=minsize, pnet=pnet, rnet=rnet,
        onet=onet, threshold=threshold, factor=factor
    )
    # If human face(s) is/are detected:
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                #resized = resize(image=cropped, output_shape=(input_image_size, input_image_size))
                resized = imresize(arr=cropped, size=(input_image_size, input_image_size), mode='RGB')
                faces.append(resized)
                rects.append([bb[0], bb[1], bb[2], bb[3]])

    return faces, rects


def forward_pass(img, session, images_placeholder, phase_train_placeholder, embeddings, image_size):
    # If there is a human face
    if img is not None:
        # Normalize the pixel values of the image for noise reduction for better accuracy and resize to desired size
        image = load_img(
            img=img, do_random_crop=False, do_random_flip=False,
            do_prewhiten=True, image_size=image_size
        )
        # Run forward pass on FaceNet model to calculate embedding
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        embedding = session.run(embeddings, feed_dict=feed_dict)
        return embedding

    else:
        return None


def save_embedding(embedding, filename, embeddings_path):
    # Save embedding of image using filename
    path = os.path.join(embeddings_path, str(filename))
    try:
        np.save(path, embedding)

    except Exception as e:
        print(str(e))


def load_embeddings():
    embedding_dict = defaultdict()

    for embedding in glob.iglob(pathname='embeddings/*.npy'):
        name = remove_file_extension(embedding)
        dict_embedding = np.load(embedding)
        embedding_dict[name] = dict_embedding

    return embedding_dict


def identify_face(embedding, embedding_dict):
    min_distance = 1.1
    try:
        for (name, dict_embedding) in embedding_dict.items():
            # Compute euclidean distance between current embedding and the embeddings from the 'embeddings' folder
            distance = np.linalg.norm(embedding - dict_embedding)

            if distance < min_distance:
                min_distance = distance
                identity = name

        if min_distance <= 0.9:
            identity = identity[11:]
            result = str(identity)
            return result

        elif min_distance <= 1:
            result = "Unknown"
            return result

    except Exception as e:
        print(str(e))
        return str(e)
