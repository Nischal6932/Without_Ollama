import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def download_model_from_s3():
    """
    Download ML model from AWS S3
    """
    bucket_name = "your-model-bucket"
    model_key = "models/plant_disease_efficientnet.keras"
    model_path = "plant_disease_efficientnet.keras"
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return model_path
    
    try:
        # Use IAM role in production or credentials in dev
        s3 = boto3.client('s3')
        
        print(f"📥 Downloading model from S3: {bucket_name}/{model_key}")
        s3.download_file(bucket_name, model_key, model_path)
        
        print(f"✅ Model downloaded successfully: {model_path}")
        return model_path
        
    except NoCredentialsError:
        print("❌ AWS credentials not found")
        return None
    except ClientError as e:
        print(f"❌ S3 download failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None
