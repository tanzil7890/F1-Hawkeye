"""
Parquet Exporter for F1 Telemetry Data

Exports database sessions to Parquet format optimized for:
- Machine learning training (pandas/sklearn/TensorFlow)
- Fast columnar queries (10-100x faster than CSV)
- Compact storage (5-10x compression vs CSV)
- Apache Arrow ecosystem compatibility

Parquet advantages:
- Column-oriented (only read columns you need)
- Built-in compression (snappy, gzip, lz4)
- Schema enforcement
- Predicate pushdown (filter before reading)
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..database import get_db_session
from ..database.models import (
    SessionModel, LapModel, TelemetrySnapshotModel,
    TyreDataModel, DamageEventModel, PitStopModel, WeatherSampleModel
)


class ParquetExporter:
    """
    Export F1 telemetry sessions to Parquet format

    Usage:
        exporter = ParquetExporter()
        exporter.export_session(
            session_id=1,
            output_dir='exports_parquet/',
            compression='snappy'  # or 'gzip', 'lz4', 'brotli'
        )

    Reading Parquet:
        import pandas as pd
        laps = pd.read_parquet('Monaco_Race_20251128_laps.parquet')
        print(laps.head())
    """

    def __init__(self):
        self.exported_files = []

    def export_session(self,
                      session_id: int,
                      output_dir: str = 'exports_parquet',
                      compression: str = 'snappy',
                      include_telemetry: bool = False):
        """
        Export complete session to Parquet files

        Args:
            session_id: Database session ID
            output_dir: Output directory path
            compression: 'snappy' (default, fast), 'gzip' (smaller), 'lz4', 'brotli'
            include_telemetry: Export telemetry_snapshots (huge file)

        Returns:
            List of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.exported_files = []

        with get_db_session() as session:
            # Get session metadata
            session_model = session.query(SessionModel).get(session_id)
            if not session_model:
                raise ValueError(f"Session {session_id} not found")

            # Create filename prefix
            prefix = f"{session_model.track_name}_{session_model.session_type}_{session_model.created_at.strftime('%Y%m%d_%H%M%S')}"

            # Export each data type as Parquet
            self._export_laps(session_id, session, output_path, prefix, compression)
            self._export_tyre_data(session_id, session, output_path, prefix, compression)
            self._export_damage_events(session_id, session, output_path, prefix, compression)
            self._export_pit_stops(session_id, session, output_path, prefix, compression)
            self._export_weather_samples(session_id, session, output_path, prefix, compression)

            if include_telemetry:
                print(f"[ParquetExporter] WARNING: Exporting telemetry (large file, but compressed)")
                self._export_telemetry(session_id, session, output_path, prefix, compression)

        print(f"[ParquetExporter] Exported {len(self.exported_files)} Parquet files to {output_dir}")
        return self.exported_files

    def _export_laps(self, session_id, session, output_path, prefix, compression):
        """Export laps to Parquet"""
        laps = session.query(LapModel).filter_by(session_id=session_id).all()

        if not laps:
            return

        # Convert to DataFrame
        data = [{
            'lap_number': lap.lap_number,
            'driver_index': lap.driver_index,
            'driver_name': lap.driver_name,
            'team_name': lap.team_name,
            'race_number': lap.race_number,
            'lap_time_ms': lap.lap_time_ms,
            'sector1_time_ms': lap.sector1_time_ms,
            'sector2_time_ms': lap.sector2_time_ms,
            'sector3_time_ms': lap.sector3_time_ms,
            'position': lap.position,
            'grid_position': lap.grid_position,
            'gap_to_leader_ms': lap.gap_to_leader_ms,
            'gap_to_ahead_ms': lap.gap_to_ahead_ms,
            'tyre_compound': lap.tyre_compound,
            'visual_tyre_compound': lap.visual_tyre_compound,
            'tyre_age_laps': lap.tyre_age_laps,
            'tyre_wear_fl': lap.tyre_wear_fl,
            'tyre_wear_fr': lap.tyre_wear_fr,
            'tyre_wear_rl': lap.tyre_wear_rl,
            'tyre_wear_rr': lap.tyre_wear_rr,
            'fuel_remaining_laps': lap.fuel_remaining_laps,
            'fuel_in_tank_kg': lap.fuel_in_tank_kg,
            'ers_percent': lap.ers_percent,
            'ers_deployed_lap': lap.ers_deployed_lap,
            'ers_harvested_lap': lap.ers_harvested_lap,
            'front_left_wing_damage': lap.front_left_wing_damage,
            'front_right_wing_damage': lap.front_right_wing_damage,
            'rear_wing_damage': lap.rear_wing_damage,
            'floor_damage': lap.floor_damage,
            'diffuser_damage': lap.diffuser_damage,
            'sidepod_damage': lap.sidepod_damage,
            'warnings': lap.warnings,
            'time_penalties_seconds': lap.time_penalties_seconds,
            'num_pit_stops': lap.num_pit_stops,
            'current_lap_invalid': lap.current_lap_invalid,
            'speed_trap_speed': lap.speed_trap_speed,
            'drs_activated': lap.drs_activated,
            'timestamp': lap.timestamp
        } for lap in laps]

        df = pd.DataFrame(data)

        # Export to Parquet with optimization
        filename = output_path / f"{prefix}_laps.parquet"
        df.to_parquet(
            filename,
            compression=compression,
            index=False,
            engine='pyarrow'
        )

        file_size = filename.stat().st_size / 1024  # KB
        self.exported_files.append(str(filename))
        print(f"[ParquetExporter] Exported {len(laps)} laps ({file_size:.1f} KB): {filename.name}")

    def _export_telemetry(self, session_id, session, output_path, prefix, compression):
        """Export telemetry snapshots to Parquet"""
        # Query in chunks to avoid memory issues
        chunk_size = 10000
        offset = 0
        chunk_num = 0

        while True:
            telemetry_chunk = session.query(TelemetrySnapshotModel)\
                                    .filter_by(session_id=session_id)\
                                    .offset(offset)\
                                    .limit(chunk_size)\
                                    .all()

            if not telemetry_chunk:
                break

            # Convert to DataFrame
            data = [{
                'driver_index': t.driver_index,
                'lap_number': t.lap_number,
                'lap_distance': t.lap_distance,
                'world_position_x': t.world_position_x,
                'world_position_y': t.world_position_y,
                'world_position_z': t.world_position_z,
                'velocity_x': t.velocity_x,
                'velocity_y': t.velocity_y,
                'velocity_z': t.velocity_z,
                'speed_kph': t.speed_kph,
                'engine_rpm': t.engine_rpm,
                'engine_temp': t.engine_temp,
                'throttle': t.throttle,
                'brake': t.brake,
                'steering': t.steering,
                'clutch': t.clutch,
                'gear': t.gear,
                'drs_active': t.drs_active,
                'ers_deploy_mode': t.ers_deploy_mode,
                'tyre_surface_temp_fl': t.tyre_surface_temp_fl,
                'tyre_surface_temp_fr': t.tyre_surface_temp_fr,
                'tyre_surface_temp_rl': t.tyre_surface_temp_rl,
                'tyre_surface_temp_rr': t.tyre_surface_temp_rr,
                'tyre_inner_temp_fl': t.tyre_inner_temp_fl,
                'tyre_inner_temp_fr': t.tyre_inner_temp_fr,
                'tyre_inner_temp_rl': t.tyre_inner_temp_rl,
                'tyre_inner_temp_rr': t.tyre_inner_temp_rr,
                'brake_temp_fl': t.brake_temp_fl,
                'brake_temp_fr': t.brake_temp_fr,
                'brake_temp_rl': t.brake_temp_rl,
                'brake_temp_rr': t.brake_temp_rr,
                'timestamp': t.timestamp
            } for t in telemetry_chunk]

            df = pd.DataFrame(data)

            # Export chunk to Parquet
            filename = output_path / f"{prefix}_telemetry_chunk{chunk_num}.parquet"
            df.to_parquet(
                filename,
                compression=compression,
                index=False,
                engine='pyarrow'
            )

            file_size = filename.stat().st_size / (1024 * 1024)  # MB
            self.exported_files.append(str(filename))
            print(f"[ParquetExporter] Exported telemetry chunk {chunk_num} ({file_size:.1f} MB): {filename.name}")

            offset += chunk_size
            chunk_num += 1

    def _export_tyre_data(self, session_id, session, output_path, prefix, compression):
        """Export tyre data to Parquet"""
        tyres = session.query(TyreDataModel).filter_by(session_id=session_id).all()

        if not tyres:
            return

        data = [{
            'driver_index': t.driver_index,
            'lap_number': t.lap_number,
            'compound': t.compound,
            'visual_compound': t.visual_compound,
            'tyre_age_laps': t.tyre_age_laps,
            'wear_fl': t.wear_fl,
            'wear_fr': t.wear_fr,
            'wear_rl': t.wear_rl,
            'wear_rr': t.wear_rr,
            'wear_avg': t.wear_avg,
            'wear_delta_fl': t.wear_delta_fl,
            'wear_delta_fr': t.wear_delta_fr,
            'wear_delta_rl': t.wear_delta_rl,
            'wear_delta_rr': t.wear_delta_rr,
            'surface_temp_fl': t.surface_temp_fl,
            'surface_temp_fr': t.surface_temp_fr,
            'surface_temp_rl': t.surface_temp_rl,
            'surface_temp_rr': t.surface_temp_rr,
            'surface_temp_avg': t.surface_temp_avg,
            'inner_temp_fl': t.inner_temp_fl,
            'inner_temp_fr': t.inner_temp_fr,
            'inner_temp_rl': t.inner_temp_rl,
            'inner_temp_rr': t.inner_temp_rr,
            'inner_temp_avg': t.inner_temp_avg,
            'pressure_fl': t.pressure_fl,
            'pressure_fr': t.pressure_fr,
            'pressure_rl': t.pressure_rl,
            'pressure_rr': t.pressure_rr,
            'timestamp': t.timestamp
        } for t in tyres]

        df = pd.DataFrame(data)

        filename = output_path / f"{prefix}_tyres.parquet"
        df.to_parquet(filename, compression=compression, index=False, engine='pyarrow')

        file_size = filename.stat().st_size / 1024
        self.exported_files.append(str(filename))
        print(f"[ParquetExporter] Exported {len(tyres)} tyre records ({file_size:.1f} KB): {filename.name}")

    def _export_damage_events(self, session_id, session, output_path, prefix, compression):
        """Export damage events to Parquet"""
        damage = session.query(DamageEventModel).filter_by(session_id=session_id).all()

        if not damage:
            return

        data = [{
            'driver_index': d.driver_index,
            'lap_number': d.lap_number,
            'damage_type': d.damage_type,
            'severity': d.severity,
            'front_left_wing_damage': d.front_left_wing_damage,
            'front_right_wing_damage': d.front_right_wing_damage,
            'rear_wing_damage': d.rear_wing_damage,
            'floor_damage': d.floor_damage,
            'diffuser_damage': d.diffuser_damage,
            'sidepod_damage': d.sidepod_damage,
            'front_left_wing_delta': d.front_left_wing_delta,
            'front_right_wing_delta': d.front_right_wing_delta,
            'rear_wing_delta': d.rear_wing_delta,
            'floor_delta': d.floor_delta,
            'diffuser_delta': d.diffuser_delta,
            'sidepod_delta': d.sidepod_delta,
            'tyre_damage_fl': d.tyre_damage_fl,
            'tyre_damage_fr': d.tyre_damage_fr,
            'tyre_damage_rl': d.tyre_damage_rl,
            'tyre_damage_rr': d.tyre_damage_rr,
            'engine_damage': d.engine_damage,
            'gearbox_damage': d.gearbox_damage,
            'timestamp': d.timestamp
        } for d in damage]

        df = pd.DataFrame(data)

        filename = output_path / f"{prefix}_damage.parquet"
        df.to_parquet(filename, compression=compression, index=False, engine='pyarrow')

        file_size = filename.stat().st_size / 1024
        self.exported_files.append(str(filename))
        print(f"[ParquetExporter] Exported {len(damage)} damage events ({file_size:.1f} KB): {filename.name}")

    def _export_pit_stops(self, session_id, session, output_path, prefix, compression):
        """Export pit stops to Parquet"""
        pitstops = session.query(PitStopModel).filter_by(session_id=session_id).all()

        if not pitstops:
            return

        data = [{
            'driver_index': p.driver_index,
            'lap_number': p.lap_number,
            'pit_stop_number': p.pit_stop_number,
            'entry_timestamp': p.entry_timestamp,
            'exit_timestamp': p.exit_timestamp,
            'duration_ms': p.duration_ms,
            'position_before': p.position_before,
            'position_after': p.position_after,
            'positions_lost': p.positions_lost,
            'tyres_changed': p.tyres_changed,
            'tyre_compound_before': p.tyre_compound_before,
            'tyre_compound_after': p.tyre_compound_after,
            'fuel_added_kg': p.fuel_added_kg,
            'front_wing_changed': p.front_wing_changed,
            'stop_type': p.stop_type,
            'undercut_attempt': p.undercut_attempt
        } for p in pitstops]

        df = pd.DataFrame(data)

        filename = output_path / f"{prefix}_pitstops.parquet"
        df.to_parquet(filename, compression=compression, index=False, engine='pyarrow')

        file_size = filename.stat().st_size / 1024
        self.exported_files.append(str(filename))
        print(f"[ParquetExporter] Exported {len(pitstops)} pit stops ({file_size:.1f} KB): {filename.name}")

    def _export_weather_samples(self, session_id, session, output_path, prefix, compression):
        """Export weather samples to Parquet"""
        weather = session.query(WeatherSampleModel).filter_by(session_id=session_id).all()

        if not weather:
            return

        data = [{
            'session_time_elapsed_seconds': w.session_time_elapsed_seconds,
            'weather': w.weather,
            'weather_forecast_minutes': w.weather_forecast_minutes,
            'air_temp': w.air_temp,
            'track_temp': w.track_temp,
            'air_temp_change': w.air_temp_change,
            'track_temp_change': w.track_temp_change,
            'rain_percentage': w.rain_percentage,
            'rain_intensity': w.rain_intensity,
            'track_condition': w.track_condition,
            'timestamp': w.timestamp
        } for w in weather]

        df = pd.DataFrame(data)

        filename = output_path / f"{prefix}_weather.parquet"
        df.to_parquet(filename, compression=compression, index=False, engine='pyarrow')

        file_size = filename.stat().st_size / 1024
        self.exported_files.append(str(filename))
        print(f"[ParquetExporter] Exported {len(weather)} weather samples ({file_size:.1f} KB): {filename.name}")

    def export_for_ml_training(self, session_ids: list, output_dir: str = 'ml_training_data'):
        """
        Export multiple sessions optimized for ML training

        Creates single combined Parquet files for each data type across sessions
        Ideal for training models on historical data

        Args:
            session_ids: List of session IDs to export
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"[ParquetExporter] Exporting {len(session_ids)} sessions for ML training...")

        # Combine all laps from all sessions
        all_laps = []
        all_tyres = []

        with get_db_session() as session:
            for session_id in session_ids:
                laps = session.query(LapModel).filter_by(session_id=session_id).all()
                tyres = session.query(TyreDataModel).filter_by(session_id=session_id).all()

                all_laps.extend(laps)
                all_tyres.extend(tyres)

        # Export combined datasets
        if all_laps:
            laps_data = [{
                'session_id': lap.session_id,
                'lap_number': lap.lap_number,
                'driver_index': lap.driver_index,
                'lap_time_ms': lap.lap_time_ms,
                'tyre_compound': lap.tyre_compound,
                'tyre_age_laps': lap.tyre_age_laps,
                'tyre_wear_fl': lap.tyre_wear_fl,
                'tyre_wear_fr': lap.tyre_wear_fr,
                'tyre_wear_rl': lap.tyre_wear_rl,
                'tyre_wear_rr': lap.tyre_wear_rr,
                'fuel_remaining_laps': lap.fuel_remaining_laps,
                # Add more fields as needed for training
            } for lap in all_laps]

            df_laps = pd.DataFrame(laps_data)
            filename = output_path / 'training_laps.parquet'
            df_laps.to_parquet(filename, compression='snappy', index=False)
            print(f"[ParquetExporter] ML Training: {len(all_laps)} laps → {filename.name}")

        if all_tyres:
            tyres_data = [{
                'session_id': t.session_id,
                'lap_number': t.lap_number,
                'driver_index': t.driver_index,
                'compound': t.compound,
                'tyre_age_laps': t.tyre_age_laps,
                'wear_avg': t.wear_avg,
                'wear_delta_fl': t.wear_delta_fl,
                'wear_delta_fr': t.wear_delta_fr,
                'wear_delta_rl': t.wear_delta_rl,
                'wear_delta_rr': t.wear_delta_rr,
                'surface_temp_avg': t.surface_temp_avg,
            } for t in all_tyres]

            df_tyres = pd.DataFrame(tyres_data)
            filename = output_path / 'training_tyres.parquet'
            df_tyres.to_parquet(filename, compression='snappy', index=False)
            print(f"[ParquetExporter] ML Training: {len(all_tyres)} tyre records → {filename.name}")

        print(f"[ParquetExporter] ML training data ready in {output_dir}")
