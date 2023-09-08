import os
import boto3
import io
from botocore.exceptions import NoCredentialsError
from inference import generate_video
import gdown


def upload_to_s3(file_name, s3_file_name="",):

    ACCESS_KEY = 'AKIAY5MJ2OSGLEMCB3FC'
    SECRET_KEY = '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'
    bucket_name = 'covermatic-voice'
    region = 'us-east-2'
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY)

    try:
        file_binary = open(file_name, "rb").read()
        file_as_binary = io.BytesIO(file_binary)
        s3.upload_fileobj(file_as_binary, bucket_name, s3_file_name,  # Use upload_fileobj instead of upload_file
                          ExtraArgs={'ACL': 'public-read', 'ContentType': 'video/mp4'})
        print("Upload Successful")
        url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_file_name}"
        return url
    except FileNotFoundError:
        print("The file was not found")
        url = None
        return url
    except NoCredentialsError:
        print("Credentials not available")
        url = None
        return url

def generate_and_upload_video(face_link,audio_link):
    if os.path.isfile("w2l/checkpoints/wav2lip_gan.pth"):
        pass
    else:
        print("downloading model")
        download_model()

    output_video_path=generate_video(face_link,audio_link,"checkpoints/wav2lip_gan.pth")
    video_url = upload_to_s3(file_name=output_video_path, s3_file_name="wav2lipGeneratedVideo")
    try:
        if os.path.isfile(output_video_path):
            os.remove(output_video_path)
    except:
        pass

    return video_url
def download_model():
    model_url="https://drive.google.com/file/d/1SwUhChoHpDX47bFxJmk6pg1zOR7fFxD5/view?usp=drive_link"
    local_path="checkpoints/wav2lip_gan.pth"
    gdown.download(model_url,local_path,quiet=False)
    print("Model downloaded successfully")

l=generate_and_upload_video(face_link="https://cdn.britannica.com/47/188747-050-1D34E743/Bill-Gates-2011.jpg",audio_link="https://voiceage.com/wbsamples/in_mono/Conference.wav")
print(l)