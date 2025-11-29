"""
CSV Exporter for F1 Telemetry Data

Exports database sessions to human-readable CSV files for:
- Excel/Google Sheets analysis
- External tools integration
- Data sharing

Exports:
- Session metadata (single row)
- Laps (one CSV per session)
- Telemetry snapshots (optional, can be huge)
- Tyre data
- Damage events
- Pit stops
- Weather samples
"""

import csv
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from ..database import get_db_session
from ..database.models import (
    SessionModel, LapModel, TelemetrySnapshotModel,
    TyreDataModel, DamageEventModel, PitStopModel, WeatherSampleModel
)


class CSVExporter:
    """
    Export F1 telemetry sessions to CSV format

    Usage:
        exporter = CSVExporter()
        exporter.export_session(
            session_id=1,
            output_dir='exports/',
            include_telemetry=False  # Telemetry is huge
        )
    """

    def __init__(self):
        self.exported_files = []

    def export_session(self,
                      session_id: int,
                      output_dir: str = 'exports',
                      include_telemetry: bool = False) -> List[str]:
        """
        Export complete session to CSV files

        Args:
            session_id: Database session ID
            output_dir: Output directory path
            include_telemetry: Export telemetry_snapshots (warning: huge file)

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

            # Export each data type
            self._export_session_metadata(session_model, output_path, prefix)
            self._export_laps(session_id, session, output_path, prefix)
            self._export_tyre_data(session_id, session, output_path, prefix)
            self._export_damage_events(session_id, session, output_path, prefix)
            self._export_pit_stops(session_id, session, output_path, prefix)
            self._export_weather_samples(session_id, session, output_path, prefix)

            if include_telemetry:
                print(f"[CSVExporter] WARNING: Exporting telemetry (can be 100MB+ file)")
                self._export_telemetry(session_id, session, output_path, prefix)

        print(f"[CSVExporter] Exported {len(self.exported_files)} files to {output_dir}")
        return self.exported_files

    def _export_session_metadata(self, session_model, output_path, prefix):
        """Export session metadata to CSV"""
        filename = output_path / f"{prefix}_session.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Field', 'Value'])

            # Write all session fields
            writer.writerow(['session_id', session_model.id])
            writer.writerow(['session_uid', session_model.session_uid])
            writer.writerow(['track_id', session_model.track_id])
            writer.writerow(['track_name', session_model.track_name])
            writer.writerow(['track_length', session_model.track_length])
            writer.writerow(['session_type', session_model.session_type])
            writer.writerow(['formula_type', session_model.formula_type])
            writer.writerow(['weather', session_model.weather])
            writer.writerow(['air_temp', session_model.air_temp])
            writer.writerow(['track_temp', session_model.track_temp])
            writer.writerow(['total_laps', session_model.total_laps])
            writer.writerow(['session_duration', session_model.session_duration])
            writer.writerow(['safety_car_status', session_model.safety_car_status])
            writer.writerow(['game_version', session_model.game_version])
            writer.writerow(['created_at', session_model.created_at])

        self.exported_files.append(str(filename))
        print(f"[CSVExporter] Exported session metadata: {filename.name}")

    def _export_laps(self, session_id, session, output_path, prefix):
        """Export lap data to CSV"""
        laps = session.query(LapModel)\
                     .filter_by(session_id=session_id)\
                     .order_by(LapModel.lap_number, LapModel.driver_index)\
                     .all()

        if not laps:
            return

        filename = output_path / f"{prefix}_laps.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'lap_number', 'driver_index', 'driver_name', 'team_name', 'race_number',
                'lap_time_ms', 'sector1_ms', 'sector2_ms', 'sector3_ms',
                'position', 'gap_to_leader_ms', 'gap_to_ahead_ms',
                'tyre_compound', 'tyre_age_laps',
                'tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr',
                'fuel_remaining_laps', 'ers_percent',
                'front_left_wing_damage', 'front_right_wing_damage', 'rear_wing_damage',
                'floor_damage', 'diffuser_damage', 'sidepod_damage',
                'warnings', 'time_penalties_seconds', 'num_pit_stops',
                'speed_trap_speed', 'drs_activated', 'timestamp'
            ])

            # Write lap data
            for lap in laps:
                writer.writerow([
                    lap.lap_number, lap.driver_index, lap.driver_name, lap.team_name, lap.race_number,
                    lap.lap_time_ms, lap.sector1_time_ms, lap.sector2_time_ms, lap.sector3_time_ms,
                    lap.position, lap.gap_to_leader_ms, lap.gap_to_ahead_ms,
                    lap.tyre_compound, lap.tyre_age_laps,
                    lap.tyre_wear_fl, lap.tyre_wear_fr, lap.tyre_wear_rl, lap.tyre_wear_rr,
                    lap.fuel_remaining_laps, lap.ers_percent,
                    lap.front_left_wing_damage, lap.front_right_wing_damage, lap.rear_wing_damage,
                    lap.floor_damage, lap.diffuser_damage, lap.sidepod_damage,
                    lap.warnings, lap.time_penalties_seconds, lap.num_pit_stops,
                    lap.speed_trap_speed, lap.drs_activated, lap.timestamp
                ])

        self.exported_files.append(str(filename))
        print(f"[CSVExporter] Exported {len(laps)} laps: {filename.name}")

    def _export_telemetry(self, session_id, session, output_path, prefix):
        """Export telemetry snapshots to CSV (WARNING: Large file!)"""
        telemetry = session.query(TelemetrySnapshotModel)\
                          .filter_by(session_id=session_id)\
                          .order_by(TelemetrySnapshotModel.timestamp)\
                          .all()

        if not telemetry:
            return

        filename = output_path / f"{prefix}_telemetry.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'driver_index', 'lap_number', 'lap_distance',
                'world_pos_x', 'world_pos_y', 'world_pos_z',
                'velocity_x', 'velocity_y', 'velocity_z',
                'speed_kph', 'engine_rpm', 'engine_temp',
                'throttle', 'brake', 'steering', 'clutch',
                'gear', 'drs_active', 'ers_deploy_mode',
                'tyre_surf_fl', 'tyre_surf_fr', 'tyre_surf_rl', 'tyre_surf_rr',
                'tyre_inner_fl', 'tyre_inner_fr', 'tyre_inner_rl', 'tyre_inner_rr',
                'brake_temp_fl', 'brake_temp_fr', 'brake_temp_rl', 'brake_temp_rr',
                'timestamp'
            ])

            # Write telemetry data
            for t in telemetry:
                writer.writerow([
                    t.driver_index, t.lap_number, t.lap_distance,
                    t.world_position_x, t.world_position_y, t.world_position_z,
                    t.velocity_x, t.velocity_y, t.velocity_z,
                    t.speed_kph, t.engine_rpm, t.engine_temp,
                    t.throttle, t.brake, t.steering, t.clutch,
                    t.gear, t.drs_active, t.ers_deploy_mode,
                    t.tyre_surface_temp_fl, t.tyre_surface_temp_fr, t.tyre_surface_temp_rl, t.tyre_surface_temp_rr,
                    t.tyre_inner_temp_fl, t.tyre_inner_temp_fr, t.tyre_inner_temp_rl, t.tyre_inner_temp_rr,
                    t.brake_temp_fl, t.brake_temp_fr, t.brake_temp_rl, t.brake_temp_rr,
                    t.timestamp
                ])

        self.exported_files.append(str(filename))
        print(f"[CSVExporter] Exported {len(telemetry)} telemetry snapshots: {filename.name}")

    def _export_tyre_data(self, session_id, session, output_path, prefix):
        """Export tyre data to CSV"""
        tyres = session.query(TyreDataModel)\
                      .filter_by(session_id=session_id)\
                      .order_by(TyreDataModel.lap_number, TyreDataModel.driver_index)\
                      .all()

        if not tyres:
            return

        filename = output_path / f"{prefix}_tyres.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'driver_index', 'lap_number', 'compound', 'tyre_age_laps',
                'wear_fl', 'wear_fr', 'wear_rl', 'wear_rr', 'wear_avg',
                'wear_delta_fl', 'wear_delta_fr', 'wear_delta_rl', 'wear_delta_rr',
                'surf_temp_fl', 'surf_temp_fr', 'surf_temp_rl', 'surf_temp_rr', 'surf_temp_avg',
                'inner_temp_fl', 'inner_temp_fr', 'inner_temp_rl', 'inner_temp_rr', 'inner_temp_avg',
                'pressure_fl', 'pressure_fr', 'pressure_rl', 'pressure_rr',
                'timestamp'
            ])

            # Write tyre data
            for t in tyres:
                writer.writerow([
                    t.driver_index, t.lap_number, t.compound, t.tyre_age_laps,
                    t.wear_fl, t.wear_fr, t.wear_rl, t.wear_rr, t.wear_avg,
                    t.wear_delta_fl, t.wear_delta_fr, t.wear_delta_rl, t.wear_delta_rr,
                    t.surface_temp_fl, t.surface_temp_fr, t.surface_temp_rl, t.surface_temp_rr, t.surface_temp_avg,
                    t.inner_temp_fl, t.inner_temp_fr, t.inner_temp_rl, t.inner_temp_rr, t.inner_temp_avg,
                    t.pressure_fl, t.pressure_fr, t.pressure_rl, t.pressure_rr,
                    t.timestamp
                ])

        self.exported_files.append(str(filename))
        print(f"[CSVExporter] Exported {len(tyres)} tyre records: {filename.name}")

    def _export_damage_events(self, session_id, session, output_path, prefix):
        """Export damage events to CSV"""
        damage = session.query(DamageEventModel)\
                       .filter_by(session_id=session_id)\
                       .order_by(DamageEventModel.timestamp)\
                       .all()

        if not damage:
            return

        filename = output_path / f"{prefix}_damage.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'driver_index', 'lap_number', 'damage_type', 'severity',
                'front_left_wing', 'front_right_wing', 'rear_wing',
                'floor', 'diffuser', 'sidepod',
                'front_left_wing_delta', 'front_right_wing_delta', 'rear_wing_delta',
                'floor_delta', 'diffuser_delta', 'sidepod_delta',
                'tyre_fl', 'tyre_fr', 'tyre_rl', 'tyre_rr',
                'engine', 'gearbox',
                'timestamp'
            ])

            # Write damage data
            for d in damage:
                writer.writerow([
                    d.driver_index, d.lap_number, d.damage_type, d.severity,
                    d.front_left_wing_damage, d.front_right_wing_damage, d.rear_wing_damage,
                    d.floor_damage, d.diffuser_damage, d.sidepod_damage,
                    d.front_left_wing_delta, d.front_right_wing_delta, d.rear_wing_delta,
                    d.floor_delta, d.diffuser_delta, d.sidepod_delta,
                    d.tyre_damage_fl, d.tyre_damage_fr, d.tyre_damage_rl, d.tyre_damage_rr,
                    d.engine_damage, d.gearbox_damage,
                    d.timestamp
                ])

        self.exported_files.append(str(filename))
        print(f"[CSVExporter] Exported {len(damage)} damage events: {filename.name}")

    def _export_pit_stops(self, session_id, session, output_path, prefix):
        """Export pit stops to CSV"""
        pitstops = session.query(PitStopModel)\
                         .filter_by(session_id=session_id)\
                         .order_by(PitStopModel.lap_number)\
                         .all()

        if not pitstops:
            return

        filename = output_path / f"{prefix}_pitstops.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'driver_index', 'lap_number', 'pit_stop_number',
                'entry_timestamp', 'exit_timestamp', 'duration_ms',
                'position_before', 'position_after', 'positions_lost',
                'tyres_changed', 'compound_before', 'compound_after',
                'fuel_added_kg', 'front_wing_changed',
                'stop_type', 'undercut_attempt'
            ])

            # Write pit stop data
            for p in pitstops:
                writer.writerow([
                    p.driver_index, p.lap_number, p.pit_stop_number,
                    p.entry_timestamp, p.exit_timestamp, p.duration_ms,
                    p.position_before, p.position_after, p.positions_lost,
                    p.tyres_changed, p.tyre_compound_before, p.tyre_compound_after,
                    p.fuel_added_kg, p.front_wing_changed,
                    p.stop_type, p.undercut_attempt
                ])

        self.exported_files.append(str(filename))
        print(f"[CSVExporter] Exported {len(pitstops)} pit stops: {filename.name}")

    def _export_weather_samples(self, session_id, session, output_path, prefix):
        """Export weather samples to CSV"""
        weather = session.query(WeatherSampleModel)\
                        .filter_by(session_id=session_id)\
                        .order_by(WeatherSampleModel.timestamp)\
                        .all()

        if not weather:
            return

        filename = output_path / f"{prefix}_weather.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'session_time_elapsed', 'weather', 'forecast_minutes',
                'air_temp', 'track_temp', 'air_temp_change', 'track_temp_change',
                'rain_percentage', 'rain_intensity', 'track_condition',
                'timestamp'
            ])

            # Write weather data
            for w in weather:
                writer.writerow([
                    w.session_time_elapsed_seconds, w.weather, w.weather_forecast_minutes,
                    w.air_temp, w.track_temp, w.air_temp_change, w.track_temp_change,
                    w.rain_percentage, w.rain_intensity, w.track_condition,
                    w.timestamp
                ])

        self.exported_files.append(str(filename))
        print(f"[CSVExporter] Exported {len(weather)} weather samples: {filename.name}")

    def export_all_sessions(self, output_dir: str = 'exports', include_telemetry: bool = False):
        """Export all sessions in database"""
        with get_db_session() as session:
            sessions = session.query(SessionModel).all()

            print(f"[CSVExporter] Exporting {len(sessions)} sessions...")
            for sess in sessions:
                try:
                    self.export_session(sess.id, output_dir, include_telemetry)
                except Exception as e:
                    print(f"[CSVExporter] Error exporting session {sess.id}: {e}")

        print(f"[CSVExporter] Complete! All sessions exported to {output_dir}")
