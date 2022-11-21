import cv2
import os
import sys
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as T

from PIL import Image
from model import siamese_model
from facenet_pytorch import MTCNN, InceptionResnetV1

torch.cuda.empty_cache()

def cvDnnDetectFaces(image, min_confidence=0.2, display = True):

    opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                            caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")


    image_height, image_width, _ = image.shape
    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)
   
    results = opencv_dnn_model.forward()   
    boxes = []

    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > min_confidence:

            bbox = face[3:]

            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)

            boxes.append([x1, y1, x2, y2])
    
    return boxes



def main():
    db_path = "database/"
    siamese_model_path = "saved_models/siamese_model"
    load_from_file = False
    yolov5_type = "yolov5m"
    
    frame = cv2.imread("./group.jpg")

    # Initializing all the models and reference images
    device, classes, loader, reference_cropped_img, yolov5, resnet, mtcnn, model = init(
        load_from_file=load_from_file,
        db_path=db_path,
        siamese_model_path=siamese_model_path,
        yolov5_type=yolov5_type,
    )

    face_boxes = []  # selecting person class alone
    face_name = []  # selecting person class alone


    # boxes, probs, points = mtcnn.detect(frame[:, :, ::-1], landmarks=True)
    boxes = cvDnnDetectFaces(frame)
    print("boxes", boxes)

    if boxes is not None:
        for box in boxes:  # classifying predicted boxes
            predicted_class, similarity = classify(
                box,
                frame,
                loader,
                resnet,
                model,
                reference_cropped_img,
                classes,
                device,
            )
            face_boxes.append(box)
            if predicted_class == -1:
                face_name.append("Stranger")
            else:
                face_name.append(predicted_class)

    for i, j in zip(face_boxes, face_name):
        x_min, y_min, x_max, y_max = i
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        # color coding boxes
        color = (0, 255, 0)  # Green
        if j == "Stranger":
            color = (0, 0, 255)  # Red

        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (color), 2)
        cv2.putText(
            frame,
            f"{j}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    print("here")
    scale_percent = 30# percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    # cv2.imshow("frame",frame)
    cv2.imwrite("output.jpeg", frame)


def classify(box, frame, loader, resnet, model, reference_cropped_img, classes, device):

    input_img = frame[:, :, ::-1]  # converting BGR ---> RGB
    box = (np.array(box)).astype(int)
    input_img = np.array(input_img)[box[1] : box[3] + 1, box[0] : box[2] + 1].copy()
    input_img = cv2.resize(input_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    input_img = loader((input_img - 127.5) / 128.0).type(
        torch.FloatTensor
    )  # Normalizing and converting to tensor

    THRESHOLD = 0.6# Minimum similairty required to be classified among classes

    similarity = []
    target_embeddings = resnet(input_img.unsqueeze(0).to(device)).reshape((1, 1, 512))

    for j in reference_cropped_img:
        j_embeddings = resnet(j.unsqueeze(0).to(device)).reshape((1, 1, 512))
        similarity.append(
            model(target_embeddings, j_embeddings).item()
        )  # feeding embeddings into siamese model

    max_similarity = max(similarity)
    if max_similarity >= THRESHOLD:
        predicted_class = classes[similarity.index(max_similarity)]
        return predicted_class, max_similarity

    return -1, -1


def IOU(box1, box2, screen_size=(1080, 1080)):  # calculating IOU
    boolean_box1 = np.zeros(screen_size, dtype=bool)
    boolean_box2 = np.zeros(screen_size, dtype=bool)

    x_min, y_min, x_max, y_max = box1
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            boolean_box1[y][x] = True

    x_min, y_min, x_max, y_max = box2
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max)-1, int(y_max)-1

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            boolean_box2[x][y] = True

    overlap = boolean_box1 * boolean_box2  # Logical AND
    union = boolean_box1 + boolean_box2  # Logical OR

    return overlap.sum() / float(union.sum())


def init(
    load_from_file=False, db_path=None, siamese_model_path=None, yolov5_type="yolov5m"
):
    margin = 0
    dirname = os.path.dirname(__file__)

    database_embeddings_path = os.path.join(db_path, "database_embeddings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    classes = []
    reference_img = []
    reference_cropped_img = []

    # Loading weights
    model = siamese_model()
    model.load_state_dict(torch.load(siamese_model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    # Initializing models
    yolov5 = torch.hub.load("ultralytics/yolov5", yolov5_type)
    mtcnn = MTCNN(image_size=128, margin=margin).eval()
    resnet = InceptionResnetV1(pretrained="vggface2").to(device).eval()

    loader = T.Compose([T.ToTensor()])

    if load_from_file == True:
        if os.path.exists(database_embeddings_path):
            reference_cropped_img = torch.load(database_embeddings_path)["reference"]

        else:
            print("It seems there isn't any previous reference embeddings saved !")
            load_from_file = False

    if load_from_file == False:

        classes.append("Friend1")
        classes.append("Friend2")

        print(db_path + "Friend1/" + os.listdir(db_path + "Friend1")[0])
        print(db_path + "Friend2/" + os.listdir(db_path + "Friend2")[0])

        reference_img.append(
                cv2.imread(db_path + "Friend1/" + os.listdir(db_path + "Friend1")[0])
            )
        reference_img.append(
                cv2.imread(db_path + "Friend2/" + os.listdir(db_path + "Friend2")[0])
            )

        # print("Creating new embeddings for the reference images.....")
        for i in range(len(reference_img)):
            boxes, probs, points = mtcnn.detect(reference_img[i], landmarks=True)
            # boxes = cvDnnDetectFaces(reference_img[i], display = True)

            boxes = (np.array(boxes[0])).astype(int)
            input_img = np.array(reference_img[i])[
                boxes[1] : boxes[3] + 1, boxes[0] : boxes[2] + 1
            ].copy()
            input_img = cv2.resize(
                input_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC
            )
            input_img = loader((input_img - 127.5) / 128.0).type(torch.FloatTensor)
            reference_cropped_img.append(input_img)

        print("Saving Image embeddings.....")
        torch.save({"reference": reference_cropped_img}, database_embeddings_path)
        print("Embeddings saved successfully !!!")

    return device, classes, loader, reference_cropped_img, yolov5, resnet, mtcnn, model


if __name__ == "__main__":
    main()
