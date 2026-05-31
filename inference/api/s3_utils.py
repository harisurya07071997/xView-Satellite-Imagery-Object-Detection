import boto3
from PIL import Image
from io import BytesIO
from pathlib import Path


s3 = boto3.client("s3")

BUCKET_NAME = "aws-hari-ml-model-artifacts"

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

def download_model_from_s3():

    s3_key = "xview/model/best.pt"

    WEIGHTS_DIR.mkdir(exist_ok=True)

    local_model_path = WEIGHTS_DIR / "best.pt"

    # skip if already downloaded
    if local_model_path.exists():
        print("Model already exists locally.")
        return str(local_model_path)

    print("Downloading model from S3...")

    s3.download_file(
        BUCKET_NAME,
        s3_key,
        str(local_model_path)
    )

    # validate download
    if local_model_path.stat().st_size == 0:
        raise ValueError("Downloaded model is empty.")

    print("Model downloaded successfully.")

    return str(local_model_path)


def upload_annotated_image_to_s3(image_path):

    image_id = Path(image_path).name

    s3_key = f"xview/result/{image_id}"

    buffer = BytesIO()

    with Image.open(image_path) as pil_image:

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image.save(buffer, format="JPEG")

    buffer.seek(0)

    s3.upload_fileobj(
        buffer,
        BUCKET_NAME,
        s3_key,
        ExtraArgs={
            "ContentType": "image/jpeg"
        }
    )

    s3_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"

    print(f"Result Image uploaded successfully in {s3_url}")

    return s3_url