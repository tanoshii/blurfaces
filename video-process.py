from collections import deque
import boto3
import cv2
import numpy as np

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
        if confidence < 0.3:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        boxes.append(box.astype("int"))

    return boxes

def getDetectionsInFrame(frame, number, queue, rekDetections):
    if (len(queue) == 6):
        queue.popleft()
    
    # Detections from OpenCV    
    boxes = detectFacesOpenCV(frame)
    
    # Detections from Rekognition
    print rekDetections[0]['frame']
    if (rekDetections and rekDetections[0]['frame'] == number):
        detections = rekDetections.popleft()
        for face in detections['faces']:
            print getBox(face)
            boxes.append(getBox(face))
    
    queue.append(boxes)

    
def getInterval(results):  
    a = results[0]['frame']
    b = results[1]['frame']
    interval = b - a
    if (interval % 2 == 0):
        interval = interval - 1
    return interval
    
def getFaces(queue, interval, frameNumber):
    if not queue:
        return None
        
    frame = queue[0]['frame']
    minFrame = int(frame - interval/2)
    maxFrame = int(frame + interval/2)
    if (frameNumber >= maxFrame):
        return queue.popleft()['faces']
    if (frameNumber >= minFrame):
        return queue[0]['faces']
    return None

def applyBlur(filename):
    results = getRekognitionDetections(filename) 
    #interval = getInterval(results)
    rekDetections = deque(results)
    
    capture = cv2.VideoCapture(filename)
    ret, frame = capture.read()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280,720))
    queue = deque([])


    while (capture.isOpened() and ret):
        number = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        #faces = getFaces(queue, interval, frameNumber)
        getDetectionsInFrame(frame, number, queue, rekDetections)
        
        for boxes in queue:
            for (x1, y1, x2, y2) in boxes:
                roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = cv2.blur(roi, (101, 101))

        out.write(frame)
        ret, frame = capture.read()

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    
filename = 'hackathon.mp4'
applyBlur(filename)
