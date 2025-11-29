"""
Async Data Writer for F1 Telemetry Platform

Handles non-blocking database writes at 60 FPS:
- Background thread consumes write queue
- Batch processing for performance
- Automatic session management
- Thread-safe queuing

Critical for real-time performance - DB writes never block UDP reception.
"""

import threading
import queue
import time
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from .db_manager import db_manager
from .models import (
    SessionModel, LapModel, TelemetrySnapshotModel,
    TyreDataModel, DamageEventModel, PitStopModel, WeatherSampleModel
)


class TelemetryDataWriter:
    """
    Non-blocking data writer with background thread

    Usage:
        writer = TelemetryDataWriter()
        writer.start()

        # Queue data (instant, non-blocking)
        writer.queue_lap(session_id, driver_idx, lap_data)

        writer.stop()
    """

    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        """
        Initialize data writer

        Args:
            batch_size: Commit after N records (improves performance)
            flush_interval: Force commit after N seconds
        """
        self.write_queue = queue.Queue(maxsize=10000)  # Hold 10k writes
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.worker_thread = None
        self.running = False
        self._lock = threading.Lock()

        # Current session tracking
        self.current_session_id = None
        self.current_session_uid = None

        print(f"[DataWriter] Initialized (batch={batch_size}, flush={flush_interval}s)")

    def start(self):
        """Start background writer thread"""
        if self.running:
            print("[DataWriter] Already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="TelemetryDataWriter",
            daemon=True
        )
        self.worker_thread.start()
        print("[DataWriter] Started background thread")

    def stop(self, timeout: float = 5.0):
        """
        Stop background writer thread

        Args:
            timeout: Wait up to N seconds for queue to flush
        """
        if not self.running:
            return

        self.running = False

        # Wait for queue to empty
        start_time = time.time()
        while not self.write_queue.empty() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

        remaining = self.write_queue.qsize()
        if remaining > 0:
            print(f"[DataWriter] Stopped with {remaining} writes pending")
        else:
            print("[DataWriter] Stopped (queue flushed)")

    def create_session(self, track_id: int, track_name: str,
                      session_type: str, **kwargs) -> int:
        """
        Create new session (synchronous)

        Returns:
            session_id (database primary key)
        """
        session_uid = f"{track_name}_{session_type}_{int(time.time())}"

        with db_manager.get_session() as session:
            session_model = SessionModel(
                session_uid=session_uid,
                track_id=track_id,
                track_name=track_name,
                session_type=session_type,
                created_at=datetime.utcnow(),
                **kwargs
            )
            session.add(session_model)
            session.commit()
            session.refresh(session_model)

            self.current_session_id = session_model.id
            self.current_session_uid = session_uid

            print(f"[DataWriter] Created session {self.current_session_id}: {session_uid}")
            return session_model.id

    def queue_lap(self, driver_index: int, lap_number: int, lap_data: Dict[str, Any]):
        """Queue lap data for writing"""
        if self.current_session_id is None:
            return  # No active session

        self._queue_write('lap', {
            'session_id': self.current_session_id,
            'driver_index': driver_index,
            'lap_number': lap_number,
            **lap_data
        })

    def queue_telemetry(self, driver_index: int, lap_number: int, telemetry_data: Dict[str, Any]):
        """Queue telemetry snapshot (60 FPS)"""
        if self.current_session_id is None:
            return

        self._queue_write('telemetry', {
            'session_id': self.current_session_id,
            'driver_index': driver_index,
            'lap_number': lap_number,
            **telemetry_data
        })

    def queue_tyre_data(self, driver_index: int, lap_number: int, tyre_data: Dict[str, Any]):
        """Queue tyre wear/temperature data"""
        if self.current_session_id is None:
            return

        self._queue_write('tyre', {
            'session_id': self.current_session_id,
            'driver_index': driver_index,
            'lap_number': lap_number,
            **tyre_data
        })

    def queue_damage_event(self, driver_index: int, lap_number: int, damage_data: Dict[str, Any]):
        """Queue damage event"""
        if self.current_session_id is None:
            return

        self._queue_write('damage', {
            'session_id': self.current_session_id,
            'driver_index': driver_index,
            'lap_number': lap_number,
            **damage_data
        })

    def queue_pit_stop(self, driver_index: int, lap_number: int, pit_data: Dict[str, Any]):
        """Queue pit stop data"""
        if self.current_session_id is None:
            return

        self._queue_write('pitstop', {
            'session_id': self.current_session_id,
            'driver_index': driver_index,
            'lap_number': lap_number,
            **pit_data
        })

    def queue_weather_sample(self, weather_data: Dict[str, Any]):
        """Queue weather sample"""
        if self.current_session_id is None:
            return

        self._queue_write('weather', {
            'session_id': self.current_session_id,
            **weather_data
        })

    def _queue_write(self, data_type: str, data: Dict[str, Any]):
        """Internal: Queue a write operation"""
        try:
            self.write_queue.put_nowait({
                'type': data_type,
                'data': data,
                'timestamp': time.time()
            })
        except queue.Full:
            print(f"[DataWriter] WARNING: Queue full, dropping {data_type} write")

    def _worker_loop(self):
        """Background thread: Process write queue"""
        print("[DataWriter] Worker thread started")
        batch = []
        last_flush = time.time()

        while self.running or not self.write_queue.empty():
            try:
                # Get item with timeout
                item = self.write_queue.get(timeout=0.1)
                batch.append(item)

                # Flush batch if full or time elapsed
                should_flush = (
                    len(batch) >= self.batch_size or
                    (time.time() - last_flush) >= self.flush_interval
                )

                if should_flush:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

            except queue.Empty:
                # Flush partial batch if timeout
                if batch and (time.time() - last_flush) >= self.flush_interval:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

        # Final flush
        if batch:
            self._flush_batch(batch)

        print("[DataWriter] Worker thread stopped")

    def _flush_batch(self, batch: list):
        """Write batch to database"""
        if not batch:
            return

        try:
            with db_manager.get_session() as session:
                for item in batch:
                    model = self._create_model(item['type'], item['data'])
                    if model:
                        session.add(model)

                session.commit()

            # Success - print summary
            type_counts = {}
            for item in batch:
                t = item['type']
                type_counts[t] = type_counts.get(t, 0) + 1

            summary = ', '.join(f"{t}:{c}" for t, c in type_counts.items())
            # print(f"[DataWriter] Flushed {len(batch)} records ({summary})")

        except Exception as e:
            print(f"[DataWriter] ERROR flushing batch: {e}")

    def _create_model(self, data_type: str, data: Dict[str, Any]):
        """Create SQLAlchemy model from data dict"""
        try:
            if data_type == 'lap':
                return LapModel(**data)
            elif data_type == 'telemetry':
                return TelemetrySnapshotModel(**data)
            elif data_type == 'tyre':
                return TyreDataModel(**data)
            elif data_type == 'damage':
                return DamageEventModel(**data)
            elif data_type == 'pitstop':
                return PitStopModel(**data)
            elif data_type == 'weather':
                return WeatherSampleModel(**data)
            else:
                print(f"[DataWriter] Unknown data type: {data_type}")
                return None
        except Exception as e:
            print(f"[DataWriter] Error creating {data_type} model: {e}")
            return None

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.write_queue.qsize()

    def is_running(self) -> bool:
        """Check if writer is running"""
        return self.running


# === Global Instance ===
telemetry_writer = TelemetryDataWriter()


# === Convenience Functions ===

def start_data_writer():
    """Start global data writer"""
    telemetry_writer.start()


def stop_data_writer():
    """Stop global data writer"""
    telemetry_writer.stop()


def get_writer():
    """Get global data writer instance"""
    return telemetry_writer
