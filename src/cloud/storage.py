"""
Cloud Storage Integration

Upload/download telemetry data to AWS S3 or Google Cloud Storage.

Usage:
    from src.cloud import CloudStorage

    # AWS S3
    storage = CloudStorage(provider='aws', credentials={'key': '...', 'secret': '...'})
    storage.upload_session(session_id=1, bucket='my-f1-data')

    # Google Cloud Storage
    storage = CloudStorage(provider='gcp', credentials={'project': '...'})
    storage.upload_session(session_id=1, bucket='my-f1-data')
"""

from typing import Dict, Optional
import json
from pathlib import Path


class CloudStorage:
    """
    Cloud storage abstraction for AWS S3 and Google Cloud Storage

    Supports uploading telemetry data and trained models to cloud.
    """

    def __init__(
        self,
        provider: str = 'aws',  # 'aws' or 'gcp'
        credentials: Optional[Dict] = None
    ):
        """
        Initialize cloud storage

        Args:
            provider: Cloud provider ('aws' or 'gcp')
            credentials: Cloud credentials dictionary
        """
        self.provider = provider
        self.credentials = credentials or {}

        # Try to initialize cloud client
        if provider == 'aws':
            try:
                import boto3
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=credentials.get('key'),
                    aws_secret_access_key=credentials.get('secret'),
                    region_name=credentials.get('region', 'us-east-1')
                )
                self.available = True
            except ImportError:
                print("boto3 not installed. Install with: pip install boto3")
                self.available = False
            except Exception as e:
                print(f"AWS S3 initialization failed: {e}")
                self.available = False

        elif provider == 'gcp':
            try:
                from google.cloud import storage
                self.client = storage.Client(project=credentials.get('project'))
                self.available = True
            except ImportError:
                print("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
                self.available = False
            except Exception as e:
                print(f"GCP Storage initialization failed: {e}")
                self.available = False

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def upload_session(
        self,
        session_id: int,
        bucket: str,
        prefix: str = 'sessions/'
    ) -> bool:
        """
        Upload session data to cloud

        Args:
            session_id: Database session ID
            bucket: Cloud storage bucket name
            prefix: Object key prefix

        Returns:
            Success status
        """
        if not self.available:
            print("Cloud storage not available")
            return False

        try:
            # Export session to local file first
            from ..export import ParquetExporter

            exporter = ParquetExporter()
            local_dir = f'temp_export_{session_id}'
            exporter.export_session(session_id, output_dir=local_dir)

            # Upload files
            key_prefix = f"{prefix}session_{session_id}/"

            for file_path in Path(local_dir).glob('*.parquet'):
                object_key = f"{key_prefix}{file_path.name}"

                if self.provider == 'aws':
                    self.client.upload_file(str(file_path), bucket, object_key)
                elif self.provider == 'gcp':
                    bucket_obj = self.client.bucket(bucket)
                    blob = bucket_obj.blob(object_key)
                    blob.upload_from_filename(str(file_path))

                print(f"  ✓ Uploaded {file_path.name} to {object_key}")

            # Cleanup
            import shutil
            shutil.rmtree(local_dir)

            return True

        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def upload_model(
        self,
        model_path: str,
        bucket: str,
        object_key: str
    ) -> bool:
        """
        Upload trained model to cloud

        Args:
            model_path: Local model file path
            bucket: Cloud storage bucket
            object_key: Object key in bucket

        Returns:
            Success status
        """
        if not self.available:
            return False

        try:
            if self.provider == 'aws':
                self.client.upload_file(model_path, bucket, object_key)
            elif self.provider == 'gcp':
                bucket_obj = self.client.bucket(bucket)
                blob = bucket_obj.blob(object_key)
                blob.upload_from_filename(model_path)

            print(f"✓ Model uploaded to {object_key}")
            return True

        except Exception as e:
            print(f"Model upload failed: {e}")
            return False

    def download_model(
        self,
        bucket: str,
        object_key: str,
        local_path: str
    ) -> bool:
        """
        Download trained model from cloud

        Args:
            bucket: Cloud storage bucket
            object_key: Object key in bucket
            local_path: Local destination path

        Returns:
            Success status
        """
        if not self.available:
            return False

        try:
            if self.provider == 'aws':
                self.client.download_file(bucket, object_key, local_path)
            elif self.provider == 'gcp':
                bucket_obj = self.client.bucket(bucket)
                blob = bucket_obj.blob(object_key)
                blob.download_to_filename(local_path)

            print(f"✓ Model downloaded to {local_path}")
            return True

        except Exception as e:
            print(f"Model download failed: {e}")
            return False
