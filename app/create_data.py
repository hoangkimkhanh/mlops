import torch
import os
import numpy as np
from PIL import Image
import argparse
from config import Config
from model import VIT_MSN
from tqdm import tqdm
import uuid
from io import BytesIO
from loguru import logger
from utils import get_index
from google.cloud import storage
from google.oauth2 import service_account

INDEX_NAME = Config.INDEX_NAME
index = get_index(INDEX_NAME)
logger.info(f"Connect to index {INDEX_NAME} successfully")

GCS_BUCKET_NAME = Config.GCS_BUCKET_NAME
key_path = "dynamic-branch-441814-f1-45971c71ec3a.json"
credentials = service_account.Credentials.from_service_account_file(key_path)
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
logger.info(f"Connected to GCS bucket {GCS_BUCKET_NAME} successfully")

DEVICE = Config.DEVICE
model = VIT_MSN(device=DEVICE)
model.eval()
logger.info(f"Load model to {DEVICE} successfully")

def run(list_file, batch_size=16):
    features = []
    metadata = []
    img_batch = []
    num = 0
    for idx, img_path in tqdm(enumerate(list_file), desc="Gen features: "):
        num += 1
        img = Image.open(img_path).convert("RGB")
        unique_id = str(uuid.uuid4())
        file_extension = img_path.split(".")[-1].upper()
        if file_extension == 'JPG':
            file_extension = 'JPEG'
        gcs_file_path = f"images/{unique_id}.{file_extension}"
        
        img_batch.append(img)
        metadata.append({"gcs_path": gcs_file_path, "file_name": os.path.basename(img_path)})

        blob = bucket.blob(gcs_file_path)
        img_buffer = BytesIO()
        img.save(img_buffer, format=file_extension)
        img_buffer.seek(0)
        blob.upload_from_file(img_buffer)
        logger.info(f"Uploaded {img_path} to GCS as {gcs_file_path}")

        if num % batch_size == 0 or idx == len(list_file) - 1:
            feature_batch = model.get_features(img_batch).tolist()
            features.extend(feature_batch)
            metadata.append({"gcs_path": gcs_file_path, "file_name": os.path.basename(img_path)})
            index.upsert([(str(uuid.uuid4()), feature, meta) for feature, meta in zip(features, metadata)])
            features = []
            metadata = []
            img_batch = []
    
def count_uploaded_images():
    blobs = bucket.list_blobs(prefix="images/") 
    uploaded_images_count = sum(1 for _ in blobs) 
    logger.info(f"Total images uploaded to GCS: {uploaded_images_count}")

def count_uploaded_features():
    query_result = index.describe_index_stats()
    uploaded_features_count = query_result['vector_count']
    logger.info(f"Total features uploaded to Pinecone index: {uploaded_features_count}")
    return uploaded_features_count

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/oxbuild/images')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    list_files = os.listdir(args.input)
    list_files = [os.path.join(args.input, _name) for _name in list_files]

    run(list_file=list_files, batch_size=16)
    count_uploaded_images()
    count_uploaded_features()