"""
F1 Telemetry REST API Module

FastAPI-based REST API for external access.

Endpoints:
- GET /sessions - List sessions
- GET /sessions/{id} - Get session details
- GET /predictions - Get ML predictions
- POST /alerts - Subscribe to alerts

Usage:
    # Run API server
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

__all__ = []
