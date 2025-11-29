"""
Database Manager for F1 Telemetry Platform

Handles:
- Database connection management (SQLite/PostgreSQL)
- Schema creation and migrations
- Session lifecycle
- Transaction management
- Connection pooling
- Thread-safety for concurrent writes

Thread-safe singleton pattern for global access.
"""

import os
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
from datetime import datetime

from sqlalchemy import create_engine, event, Index
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool

from .models import (
    Base,
    SessionModel,
    LapModel,
    TelemetrySnapshotModel,
    TyreDataModel,
    DamageEventModel,
    PitStopModel,
    WeatherSampleModel,
    SCHEMA_VERSION
)


class DatabaseManager:
    """
    Thread-safe singleton database manager

    Usage:
        db_manager = DatabaseManager()
        db_manager.initialize()

        with db_manager.get_session() as session:
            session.add(lap_model)
            session.commit()
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern - only one instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize database manager (called only once)"""
        if hasattr(self, '_initialized'):
            return

        self.engine = None
        self.session_factory = None
        self.Session = None
        self._db_path = None
        self._initialized = False

    def initialize(self,
                   db_path: Optional[str] = None,
                   db_type: str = 'sqlite',
                   echo: bool = False):
        """
        Initialize database connection and create schema

        Args:
            db_path: Database file path (SQLite) or connection string (PostgreSQL)
            db_type: 'sqlite' or 'postgresql'
            echo: If True, log all SQL statements (useful for debugging)
        """
        if self._initialized:
            print("[Database] Already initialized")
            return

        with self._lock:
            if db_type == 'sqlite':
                self._initialize_sqlite(db_path, echo)
            elif db_type == 'postgresql':
                self._initialize_postgresql(db_path, echo)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            # Create all tables
            self._create_schema()

            # Create performance indices
            self._create_indices()

            self._initialized = True
            print(f"[Database] Initialized successfully: {self._db_path}")
            print(f"[Database] Schema version: {SCHEMA_VERSION}")

    def _initialize_sqlite(self, db_path: Optional[str], echo: bool):
        """Initialize SQLite database"""
        # Default path: project_root/f1_telemetry.db
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / 'f1_telemetry.db'
        else:
            db_path = Path(db_path)

        # Create directory if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = str(db_path)

        # Create engine with optimizations
        self.engine = create_engine(
            f'sqlite:///{self._db_path}',
            echo=echo,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            connect_args={
                'check_same_thread': False,  # Allow multi-threading
                'timeout': 30  # 30 second timeout for locks
            }
        )

        # Enable SQLite performance optimizations
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            # Performance optimizations
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            cursor.execute("PRAGMA synchronous=NORMAL")  # Balanced durability
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.execute("PRAGMA temp_store=MEMORY")  # In-memory temp tables
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
            cursor.close()

        self._create_session_factory()

    def _initialize_postgresql(self, connection_string: str, echo: bool):
        """Initialize PostgreSQL database"""
        if not connection_string:
            raise ValueError("PostgreSQL connection string required")

        self._db_path = connection_string

        # Create engine with connection pooling
        self.engine = create_engine(
            connection_string,
            echo=echo,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600  # Recycle connections after 1 hour
        )

        self._create_session_factory()

    def _create_session_factory(self):
        """Create thread-safe session factory"""
        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        # Scoped session for thread-local storage
        self.Session = scoped_session(self.session_factory)

    def _create_schema(self):
        """Create all database tables"""
        Base.metadata.create_all(self.engine)
        print(f"[Database] Created {len(Base.metadata.tables)} tables")

    def _create_indices(self):
        """Create composite indices for query performance"""
        with self.engine.connect() as conn:
            # Composite index on laps (session_id, lap_number)
            Index('idx_laps_session_lap',
                  LapModel.session_id,
                  LapModel.lap_number).create(conn, checkfirst=True)

            # Composite index on laps (session_id, driver_index, lap_number)
            Index('idx_laps_session_driver_lap',
                  LapModel.session_id,
                  LapModel.driver_index,
                  LapModel.lap_number).create(conn, checkfirst=True)

            # Composite index on telemetry (session_id, driver_index, lap_number)
            Index('idx_telemetry_session_driver_lap',
                  TelemetrySnapshotModel.session_id,
                  TelemetrySnapshotModel.driver_index,
                  TelemetrySnapshotModel.lap_number).create(conn, checkfirst=True)

            # Timestamp indices for time-series queries
            Index('idx_telemetry_timestamp',
                  TelemetrySnapshotModel.timestamp).create(conn, checkfirst=True)

            Index('idx_weather_timestamp',
                  WeatherSampleModel.timestamp).create(conn, checkfirst=True)

            Index('idx_damage_timestamp',
                  DamageEventModel.timestamp).create(conn, checkfirst=True)

            conn.commit()

        print("[Database] Created performance indices")

    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session (context manager)

        Usage:
            with db_manager.get_session() as session:
                session.add(model)
                session.commit()

        Automatically handles:
        - Session creation
        - Rollback on error
        - Session cleanup
        """
        if not self._initialized:
            raise RuntimeError("DatabaseManager not initialized. Call initialize() first.")

        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[Database] Session error: {e}")
            raise
        finally:
            session.close()

    def create_session_record(self,
                            session_uid: str,
                            track_id: int,
                            track_name: str,
                            session_type: str,
                            **kwargs) -> SessionModel:
        """
        Create a new session record

        Args:
            session_uid: Unique session identifier
            track_id: Track ID from game
            track_name: Track name (e.g., "Monaco")
            session_type: "Practice", "Qualifying", or "Race"
            **kwargs: Additional session metadata

        Returns:
            SessionModel instance
        """
        with self.get_session() as session:
            session_model = SessionModel(
                session_uid=session_uid,
                track_id=track_id,
                track_name=track_name,
                session_type=session_type,
                **kwargs
            )
            session.add(session_model)
            session.commit()
            session.refresh(session_model)  # Get auto-generated ID
            print(f"[Database] Created session: {session_uid} ({track_name}, {session_type})")
            return session_model

    def get_session_by_uid(self, session_uid: str) -> Optional[SessionModel]:
        """Get session by unique identifier"""
        with self.get_session() as session:
            return session.query(SessionModel).filter_by(session_uid=session_uid).first()

    def get_recent_sessions(self, limit: int = 10):
        """Get most recent sessions"""
        with self.get_session() as session:
            return session.query(SessionModel)\
                         .order_by(SessionModel.created_at.desc())\
                         .limit(limit)\
                         .all()

    def get_statistics(self) -> dict:
        """
        Get database statistics

        Returns:
            dict with counts for each table
        """
        if not self._initialized:
            return {}

        stats = {}
        with self.get_session() as session:
            stats['sessions'] = session.query(SessionModel).count()
            stats['laps'] = session.query(LapModel).count()
            stats['telemetry_snapshots'] = session.query(TelemetrySnapshotModel).count()
            stats['tyre_data'] = session.query(TyreDataModel).count()
            stats['damage_events'] = session.query(DamageEventModel).count()
            stats['pit_stops'] = session.query(PitStopModel).count()
            stats['weather_samples'] = session.query(WeatherSampleModel).count()

        return stats

    def vacuum(self):
        """
        Optimize database (SQLite only)

        Reclaims unused space and rebuilds indices
        Run periodically to maintain performance
        """
        if 'sqlite' not in str(self._db_path):
            print("[Database] VACUUM only supported for SQLite")
            return

        with self.engine.connect() as conn:
            conn.execute("VACUUM")
            conn.commit()
        print("[Database] VACUUM completed")

    def close(self):
        """Close database connection"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        self._initialized = False
        print("[Database] Closed connection")

    def __del__(self):
        """Cleanup on destruction"""
        self.close()

    def __repr__(self):
        status = "initialized" if self._initialized else "not initialized"
        return f"<DatabaseManager({status}, path='{self._db_path}')>"


# === Global Instance ===
# Singleton instance for application-wide use
db_manager = DatabaseManager()


# === Convenience Functions ===

def initialize_database(db_path: Optional[str] = None,
                       db_type: str = 'sqlite',
                       echo: bool = False):
    """
    Initialize the global database manager

    Args:
        db_path: Database path or connection string
        db_type: 'sqlite' or 'postgresql'
        echo: Enable SQL logging
    """
    db_manager.initialize(db_path, db_type, echo)


def get_db_session():
    """Get database session (context manager)"""
    return db_manager.get_session()


def get_statistics():
    """Get database statistics"""
    return db_manager.get_statistics()
