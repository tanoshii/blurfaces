import json
import boto3

rek = boto3.client('rekognition')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('rek-vid')

def lambda_handler(event, context):
    message = event['Records'][0]['Sns']['Message']
    result = json.loads(message)
    
    if (result['Status'] == 'SUCCEEDED'):
        detections = getAllDetections(result['JobId'])
        table.put_item(Item={
            'filename': result['Video']['S3ObjectName'],
            'detections': detections
        })
        
        return detections
        
    return result['Status']

def getAllDetections(jobId):
    response = getResponse(jobId)
    detections = getDetections(response)
    
    while 'NextToken' in response:
        response = getResponse(jobId, response['NextToken'])
        detections.extend(getDetections(response))
    
    return detections

def getResponse(jobId, nextToken=None):
    if (nextToken):
        return rek.get_face_detection(JobId=jobId, NextToken=nextToken)
    return rek.get_face_detection(JobId=jobId)
    
def getDetections(response):
    metadata = response['VideoMetadata']
    faces = response['Faces']

    currentTimestamp = 0
    detections = []
    group = None
    
    for data in faces:
        face = data['Face']
        
        if ('BoundingBox' in face):
            timestamp = data['Timestamp']
            if (not group or timestamp != currentTimestamp):
                currentTimestamp = timestamp
                frame = int(round(metadata['FrameRate'] * timestamp / 1000))
                group = {'frame': frame, 'faces': []}
                detections.append(group)
            
            rect = toRect(face, metadata)
            group['faces'].append(rect)
            
    return detections

def toRect(face, metadata):    
    frameWidth = metadata['FrameWidth']
    frameHeight = metadata['FrameHeight']
    
    x = face['BoundingBox']['Left'] * frameWidth
    w = face['BoundingBox']['Width'] * frameWidth
    y = face['BoundingBox']['Top'] * frameHeight
    h = face['BoundingBox']['Height'] * frameHeight
    
    x = x if x > 0 else 0
    y = y if y > 0 else 0
    w = w if w < frameWidth else frameWidth
    h = h if h < frameHeight else frameHeight
    
    return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
