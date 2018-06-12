import urllib
import boto3

rek = boto3.client('rekognition')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    name = urllib.unquote_plus(event['Records'][0]['s3']['object']['key'])
    print('Received file {} in bucket {}.', name, bucket)
    
    response = rek.start_face_detection(
        Video={
            'S3Object': {
                'Bucket': bucket,
                'Name': name
            },
        },
        NotificationChannel={
            'SNSTopicArn': 'arn:aws:sns:us-east-1:875330872118:AmazonRekognition_rek-vid',
            'RoleArn': 'arn:aws:iam::875330872118:role/stefany'
        },
        FaceAttributes='DEFAULT'
    )
    
    return response['JobId']

