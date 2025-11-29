"""
F1 Telemetry REST API

FastAPI-based API for external access to telemetry data and predictions.

Run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

API Docs: http://localhost:8000/docs
"""

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

from typing import List
from datetime import datetime

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="F1 Telemetry API",
        description="REST API for F1 telemetry data and ML predictions",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class SessionResponse(BaseModel):
        id: int
        track_name: str
        session_type: str
        created_at: str

    class PredictionRequest(BaseModel):
        current_lap: int
        current_position: int
        tyre_age: int
        avg_wear: float
        total_laps: int

    # Endpoints
    @app.get("/")
    def root():
        """API root"""
        return {
            "name": "F1 Telemetry API",
            "version": "1.0.0",
            "docs": "/docs"
        }

    @app.get("/sessions", response_model=List[SessionResponse])
    def list_sessions(limit: int = 10):
        """List recent sessions"""
        try:
            from ..database import db_manager
            sessions = db_manager.get_recent_sessions(limit=limit)

            return [
                SessionResponse(
                    id=s.id,
                    track_name=s.track_name,
                    session_type=s.session_type,
                    created_at=s.created_at.isoformat()
                )
                for s in sessions
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/{session_id}")
    def get_session(session_id: int):
        """Get session details"""
        try:
            from ..database import db_manager, get_statistics
            session = db_manager.get_session_by_id(session_id)

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            return {
                "id": session.id,
                "track_name": session.track_name,
                "session_type": session.session_type,
                "created_at": session.created_at.isoformat(),
                "stats": get_statistics(session_id)
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predictions")
    def get_predictions(request: PredictionRequest):
        """Get ML predictions for current state"""
        try:
            from ..ml.inference import RealTimePredictor

            predictor = RealTimePredictor()
            predictor.load_models('models/')

            state = request.model_dump()
            predictions = predictor.predict(state)

            return predictions

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/anomalies/{session_id}/{driver_index}")
    def detect_anomalies(session_id: int, driver_index: int = 0):
        """Detect anomalies in session"""
        try:
            from ..advanced import AnomalyDetector

            detector = AnomalyDetector()
            anomalies = detector.detect_all_anomalies(session_id, driver_index)

            return {
                "session_id": session_id,
                "driver_index": driver_index,
                "anomalies": {
                    anomaly_type: [
                        {
                            "lap": a.lap,
                            "type": a.type,
                            "severity": a.severity,
                            "description": a.description,
                            "score": a.score
                        }
                        for a in anomaly_list
                    ]
                    for anomaly_type, anomaly_list in anomalies.items()
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

else:
    # Dummy app if FastAPI not available
    class DummyApp:
        def __call__(self, *args, **kwargs):
            return {"error": "FastAPI not installed"}

    app = DummyApp()
