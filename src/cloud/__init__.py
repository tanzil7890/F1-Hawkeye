"""
F1 Telemetry Cloud Integration Module

Cloud storage and compute integration for AWS and GCP.

Modules:
- CloudStorage: Upload/download to S3 or GCS
- CloudTraining: Train models on cloud GPUs

Usage:
    from src.cloud import CloudStorage

    storage = CloudStorage(provider='aws')
    storage.upload_session(session_id=1, bucket='f1-telemetry')
"""

from .storage import CloudStorage

__all__ = ['CloudStorage']
