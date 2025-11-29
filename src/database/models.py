"""
SQLAlchemy ORM Models for F1 Telemetry Database

Defines 7 core tables for storing telemetry data:
1. sessions - Session metadata
2. laps - Lap-by-lap data
3. telemetry_snapshots - High-frequency telemetry (60 FPS)
4. tyre_data - Tyre wear and temperature
5. damage_events - Damage tracking
6. pit_stops - Pit stop timing and strategy
7. weather_samples - Weather evolution

Schema Version: 1.0
"""

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class SessionModel(Base):
    """
    Session metadata - Top-level entity for each F1 session

    Tracks:
    - Session identification (UID, type, track)
    - Weather conditions
    - Timing information
    - Session rules (laps, duration)

    Relationships:
    - One-to-many with laps, telemetry, tyres, damage, pit_stops, weather
    """
    __tablename__ = 'sessions'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Session Identification
    session_uid = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Track Information
    track_id = Column(Integer, nullable=False)
    track_name = Column(String(100))
    track_length = Column(Float)  # meters

    # Session Type
    session_type = Column(String(50), nullable=False)  # Practice/Qualifying/Race
    formula_type = Column(String(20))  # F1/F2/F3

    # Weather Conditions (initial)
    weather = Column(String(50))  # Clear/LightCloud/Overcast/Rain/Storm
    air_temp = Column(Float)  # Celsius
    track_temp = Column(Float)  # Celsius

    # Session Rules
    total_laps = Column(Integer)
    session_duration = Column(Integer)  # seconds

    # Session State
    safety_car_status = Column(Integer)  # 0=none, 1=full, 2=virtual
    session_time_left = Column(Integer)  # seconds

    # Game Version
    game_version = Column(String(20))  # F1 2022/2023/2024/2025
    packet_format = Column(Integer)  # UDP packet format version

    # Relationships
    laps = relationship("LapModel", back_populates="session", cascade="all, delete-orphan")
    telemetry = relationship("TelemetrySnapshotModel", back_populates="session", cascade="all, delete-orphan")
    tyre_data = relationship("TyreDataModel", back_populates="session", cascade="all, delete-orphan")
    damage_events = relationship("DamageEventModel", back_populates="session", cascade="all, delete-orphan")
    pit_stops = relationship("PitStopModel", back_populates="session", cascade="all, delete-orphan")
    weather_samples = relationship("WeatherSampleModel", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session(id={self.id}, track='{self.track_name}', type='{self.session_type}', date='{self.created_at}')>"


class LapModel(Base):
    """
    Lap-by-lap data for each driver

    Tracks:
    - Lap times and sector splits
    - Position and gaps
    - Tyre strategy (compound, age)
    - Fuel and ERS status
    - Performance metrics

    Granularity: One record per driver per lap
    """
    __tablename__ = 'laps'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, index=True)

    # Driver Identification
    driver_index = Column(Integer, nullable=False)  # 0-21
    driver_name = Column(String(100))
    team_id = Column(Integer)
    team_name = Column(String(100))
    race_number = Column(Integer)

    # Lap Information
    lap_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Lap Times (milliseconds)
    lap_time_ms = Column(Integer)  # Total lap time
    sector1_time_ms = Column(Integer)
    sector2_time_ms = Column(Integer)
    sector3_time_ms = Column(Integer)

    # Position & Gaps
    position = Column(Integer)
    grid_position = Column(Integer)  # Starting position
    gap_to_leader_ms = Column(Integer)  # Gap to P1
    gap_to_ahead_ms = Column(Integer)  # Gap to car ahead

    # Tyre Strategy
    tyre_compound = Column(String(20))  # SOFT/MEDIUM/HARD/INTER/WET
    visual_tyre_compound = Column(String(10))  # C1/C2/C3/C4/C5
    tyre_age_laps = Column(Integer)

    # Tyre Wear (percentage)
    tyre_wear_fl = Column(Float)  # Front-left
    tyre_wear_fr = Column(Float)  # Front-right
    tyre_wear_rl = Column(Float)  # Rear-left
    tyre_wear_rr = Column(Float)  # Rear-right

    # Fuel & ERS
    fuel_remaining_laps = Column(Float)
    fuel_in_tank_kg = Column(Float)
    ers_percent = Column(Float)  # Battery charge 0-100
    ers_deployed_lap = Column(Float)  # MJ deployed this lap
    ers_harvested_lap = Column(Float)  # MJ harvested this lap

    # Damage (percentage)
    front_left_wing_damage = Column(Float)
    front_right_wing_damage = Column(Float)
    rear_wing_damage = Column(Float)
    floor_damage = Column(Float)
    diffuser_damage = Column(Float)
    sidepod_damage = Column(Float)

    # Penalties
    warnings = Column(Integer)  # Corner cutting warnings
    time_penalties_seconds = Column(Integer)
    num_pit_stops = Column(Integer)

    # Lap Status
    current_lap_invalid = Column(Boolean, default=False)
    lap_valid_bit_flags = Column(Integer)  # Bitfield of validity

    # Performance Metrics
    speed_trap_speed = Column(Float)  # km/h
    drs_activated = Column(Boolean, default=False)

    # Relationship
    session = relationship("SessionModel", back_populates="laps")

    def __repr__(self):
        return f"<Lap(driver='{self.driver_name}', lap={self.lap_number}, time={self.lap_time_ms}ms, pos={self.position})>"


class TelemetrySnapshotModel(Base):
    """
    High-frequency telemetry snapshots (60 FPS)

    Tracks:
    - Vehicle dynamics (speed, position, orientation)
    - Driver inputs (throttle, brake, steering)
    - Car state (gear, DRS, ERS mode)

    Granularity: ~60 records per second per driver
    Warning: This table grows VERY large (millions of rows)
    """
    __tablename__ = 'telemetry_snapshots'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, index=True)

    # Timing
    driver_index = Column(Integer, nullable=False, index=True)
    lap_number = Column(Integer, index=True)
    lap_distance = Column(Float)  # meters into lap
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Position & Motion
    world_position_x = Column(Float)
    world_position_y = Column(Float)
    world_position_z = Column(Float)
    velocity_x = Column(Float)
    velocity_y = Column(Float)
    velocity_z = Column(Float)

    # Speed & Performance
    speed_kph = Column(Float)  # km/h
    engine_rpm = Column(Integer)
    engine_temp = Column(Float)  # Celsius

    # Driver Inputs
    throttle = Column(Float)  # 0.0-1.0
    brake = Column(Float)  # 0.0-1.0
    steering = Column(Float)  # -1.0 to 1.0
    clutch = Column(Float)  # 0.0-1.0

    # Car State
    gear = Column(Integer)  # -1=Reverse, 0=Neutral, 1-8=Gears
    drs_active = Column(Boolean, default=False)
    ers_deploy_mode = Column(Integer)  # 0=none, 1=low, 2=medium, 3=high, 4=overtake, 5=hotlap

    # Tyre Temps (surface, Celsius)
    tyre_surface_temp_fl = Column(Float)
    tyre_surface_temp_fr = Column(Float)
    tyre_surface_temp_rl = Column(Float)
    tyre_surface_temp_rr = Column(Float)

    # Tyre Temps (inner, Celsius)
    tyre_inner_temp_fl = Column(Float)
    tyre_inner_temp_fr = Column(Float)
    tyre_inner_temp_rl = Column(Float)
    tyre_inner_temp_rr = Column(Float)

    # Brake Temps (Celsius)
    brake_temp_fl = Column(Float)
    brake_temp_fr = Column(Float)
    brake_temp_rl = Column(Float)
    brake_temp_rr = Column(Float)

    # Relationship
    session = relationship("SessionModel", back_populates="telemetry")

    def __repr__(self):
        return f"<TelemetrySnapshot(driver={self.driver_index}, lap={self.lap_number}, speed={self.speed_kph}kph)>"


class TyreDataModel(Base):
    """
    Tyre wear and temperature data per lap

    Tracks:
    - Wear progression per corner
    - Temperature evolution
    - Compound degradation rates

    Granularity: One record per driver per lap (or per stint)
    """
    __tablename__ = 'tyre_data'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, index=True)

    # Driver & Timing
    driver_index = Column(Integer, nullable=False)
    lap_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Tyre Compound
    compound = Column(String(20))  # SOFT/MEDIUM/HARD/INTER/WET
    visual_compound = Column(String(10))  # C1/C2/C3/C4/C5
    tyre_age_laps = Column(Integer)

    # Wear (percentage 0-100)
    wear_fl = Column(Float)
    wear_fr = Column(Float)
    wear_rl = Column(Float)
    wear_rr = Column(Float)
    wear_avg = Column(Float)  # Average across 4 corners

    # Wear Delta (change from previous lap)
    wear_delta_fl = Column(Float)
    wear_delta_fr = Column(Float)
    wear_delta_rl = Column(Float)
    wear_delta_rr = Column(Float)

    # Surface Temperature (Celsius)
    surface_temp_fl = Column(Float)
    surface_temp_fr = Column(Float)
    surface_temp_rl = Column(Float)
    surface_temp_rr = Column(Float)
    surface_temp_avg = Column(Float)

    # Inner Temperature (Celsius)
    inner_temp_fl = Column(Float)
    inner_temp_fr = Column(Float)
    inner_temp_rl = Column(Float)
    inner_temp_rr = Column(Float)
    inner_temp_avg = Column(Float)

    # Pressure (PSI)
    pressure_fl = Column(Float)
    pressure_fr = Column(Float)
    pressure_rl = Column(Float)
    pressure_rr = Column(Float)

    # Relationship
    session = relationship("SessionModel", back_populates="tyre_data")

    def __repr__(self):
        return f"<TyreData(driver={self.driver_index}, lap={self.lap_number}, compound='{self.compound}', avg_wear={self.wear_avg}%)>"


class DamageEventModel(Base):
    """
    Damage events and progression tracking

    Tracks:
    - When damage occurred (lap, timestamp)
    - Damage type and severity
    - Impact on performance

    Granularity: One record per damage change
    """
    __tablename__ = 'damage_events'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, index=True)

    # Driver & Timing
    driver_index = Column(Integer, nullable=False)
    lap_number = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Aerodynamic Damage (percentage 0-100)
    front_left_wing_damage = Column(Float, default=0.0)
    front_right_wing_damage = Column(Float, default=0.0)
    rear_wing_damage = Column(Float, default=0.0)
    floor_damage = Column(Float, default=0.0)
    diffuser_damage = Column(Float, default=0.0)
    sidepod_damage = Column(Float, default=0.0)

    # Damage Delta (change from previous)
    front_left_wing_delta = Column(Float)
    front_right_wing_delta = Column(Float)
    rear_wing_delta = Column(Float)
    floor_delta = Column(Float)
    diffuser_delta = Column(Float)
    sidepod_delta = Column(Float)

    # Tyre Damage (percentage 0-100)
    tyre_damage_fl = Column(Float, default=0.0)
    tyre_damage_fr = Column(Float, default=0.0)
    tyre_damage_rl = Column(Float, default=0.0)
    tyre_damage_rr = Column(Float, default=0.0)

    # Engine Damage
    engine_damage = Column(Float, default=0.0)
    gearbox_damage = Column(Float, default=0.0)

    # Event Context
    damage_type = Column(String(50))  # Contact/Kerb/Puncture/Mechanical
    severity = Column(String(20))  # Minor/Moderate/Major/Critical

    # Relationship
    session = relationship("SessionModel", back_populates="damage_events")

    def __repr__(self):
        return f"<DamageEvent(driver={self.driver_index}, lap={self.lap_number}, type='{self.damage_type}', severity='{self.severity}')>"


class PitStopModel(Base):
    """
    Pit stop timing and strategy

    Tracks:
    - Entry/exit timing
    - Duration in pit lane
    - Tyre changes
    - Repairs performed

    Granularity: One record per pit stop
    """
    __tablename__ = 'pit_stops'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, index=True)

    # Driver & Timing
    driver_index = Column(Integer, nullable=False)
    lap_number = Column(Integer, nullable=False, index=True)
    pit_stop_number = Column(Integer)  # 1st, 2nd, 3rd stop

    # Pit Timing
    entry_timestamp = Column(DateTime, nullable=False)
    exit_timestamp = Column(DateTime)
    duration_ms = Column(Integer)  # Total pit stop duration

    # Position Changes
    position_before = Column(Integer)
    position_after = Column(Integer)
    positions_lost = Column(Integer)  # Can be negative (gained)

    # Tyre Changes
    tyres_changed = Column(Boolean, default=True)
    tyre_compound_before = Column(String(20))
    tyre_compound_after = Column(String(20))

    # Fuel & Repairs
    fuel_added_kg = Column(Float)
    front_wing_changed = Column(Boolean, default=False)

    # Strategy
    stop_type = Column(String(20))  # Planned/Emergency/Penalty
    undercut_attempt = Column(Boolean, default=False)

    # Relationship
    session = relationship("SessionModel", back_populates="pit_stops")

    def __repr__(self):
        return f"<PitStop(driver={self.driver_index}, lap={self.lap_number}, duration={self.duration_ms}ms, tyres='{self.tyre_compound_after}')>"


class WeatherSampleModel(Base):
    """
    Weather evolution throughout session

    Tracks:
    - Weather changes over time
    - Temperature variations
    - Rain probability and intensity
    - Track condition evolution

    Granularity: Sampled every 30-60 seconds or on change
    """
    __tablename__ = 'weather_samples'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False, index=True)

    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    session_time_elapsed_seconds = Column(Integer)

    # Weather Conditions
    weather = Column(String(50))  # Clear/LightCloud/Overcast/LightRain/HeavyRain/Storm
    weather_forecast_minutes = Column(Integer)  # How many minutes ahead (0-60)

    # Temperature
    air_temp = Column(Float)  # Celsius
    track_temp = Column(Float)  # Celsius
    air_temp_change = Column(Float)  # Delta from previous
    track_temp_change = Column(Float)  # Delta from previous

    # Rain
    rain_percentage = Column(Integer)  # 0-100
    rain_intensity = Column(Integer)  # 0=none, 1=light, 2=medium, 3=heavy

    # Track Condition
    track_condition = Column(String(20))  # Dry/Damp/Wet/Soaked/Flooded

    # Relationship
    session = relationship("SessionModel", back_populates="weather_samples")

    def __repr__(self):
        return f"<WeatherSample(weather='{self.weather}', air={self.air_temp}°C, track={self.track_temp}°C, rain={self.rain_percentage}%)>"


# === Database Schema Version ===
SCHEMA_VERSION = "1.0"

# === Indices for Performance ===
# Composite indices are created in db_manager.py for:
# - (session_id, lap_number) on laps, tyre_data
# - (session_id, driver_index, lap_number) on telemetry_snapshots
# - (session_id, timestamp) on all time-series tables
