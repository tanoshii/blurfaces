from collections import deque
import boto3
import cv2
import numpy as np
import argparse

session = boto3.Session(profile_name='default')
dynamodb = session.resource('dynamodb')
table = dynamodb.Table('rek-vid')

net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 
    'res10_300x300_ssd_iter_140000.caffemodel'
)
    
def getRekognitionDetections(filename):
    data = table.get_item(Key={ 'filename': filename })
    if not 'Item' in data:
        print 'Results for video "{}" not found'.format(filename)
        return []
    return data['Item']['detections']

def getBox(face):
    box = [face['x'], face['y'], face['x'] + face['w'], face['y'] + face['h']]
    return np.array(box).astype("int")

def detectFacesOpenCV(frame):
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.4:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        boxes.append(box.astype("int"))

    return boxes


def getDetectionsInFrame(frame, number, queue, rekDetections, interval):
    if (len(queue) == interval):
        queue.popleft()
    
    # Detections from OpenCV    
    boxes = detectFacesOpenCV(frame)
    
    # Detections from Rekognition
    print 'Frame {}'.format(number)
    while(rekDetections and rekDetections[0]['frame'] <= number + interval / 2):
        detections = rekDetections.popleft()
        for face in detections['faces']:
            boxes.append(getBox(face))
    
    queue.append(boxes)


def applyBlur(filename, interval):
    results = getRekognitionDetections(filename) 
    rekDetections = deque(results)
    
    capture = cv2.VideoCapture('input/{}'.format(filename))
    ret, frame = capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter('output/{}.avi'.format(filename), 
        fourcc, fps, (width, height))
    queue = deque([])


    while (capture.isOpened() and ret):
        number = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        getDetectionsInFrame(frame, number, queue, rekDetections, interval)
        
        for boxes in queue:
            for (x1, y1, x2, y2) in boxes:
                roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = cv2.blur(roi, (101, 101))

        out.write(frame)
        ret, frame = capture.read()

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    
    
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, 
    help="video file name")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-i", "--interval", type=int, default=10, 
    help="interval for fix detections")

args = vars(ap.parse_args())
applyBlur(args['filename'], args['interval'])
