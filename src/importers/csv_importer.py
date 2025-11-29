"""
CSV Importer for F1 Telemetry Data

Imports historical CSV data into the database for:
- Seeding ML training data
- Loading past sessions for analysis
- Migrating from legacy data formats

Supports CSV files from:
- This application's CSV exporter
- External F1 telemetry tools
- Manual data exports
"""

import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..database import db_manager
from ..database.data_writer import telemetry_writer
from ..database.models import SessionModel, LapModel


class CSVImporter:
    """
    Import F1 telemetry data from CSV files

    Usage:
        importer = CSVImporter()

        # Import existing sample data
        importer.import_lap_csv(
            'data/2025-Australian Grand Prix-Practice 1.csv',
            track_name='Melbourne',
            session_type='Practice'
        )
    """

    def __init__(self):
        self.imported_records = 0

    def import_lap_csv(self,
                      csv_path: str,
                      track_name: str,
                      session_type: str,
                      track_id: int = 0) -> int:
        """
        Import lap data from CSV file

        Args:
            csv_path: Path to CSV file
            track_name: Track name (e.g., "Melbourne")
            session_type: "Practice", "Qualifying", or "Race"
            track_id: Track ID (default 0)

        Returns:
            session_id (database primary key)
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"[CSVImporter] Importing {csv_file.name}...")

        # Read CSV with pandas (auto-detects format)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")

        # Create session
        session_uid = f"{track_name}_{session_type}_{csv_file.stem}_{int(datetime.now().timestamp())}"

        with db_manager.get_session() as session:
            session_model = SessionModel(
                session_uid=session_uid,
                track_id=track_id,
                track_name=track_name,
                session_type=session_type,
                created_at=datetime.now()
            )
            session.add(session_model)
            session.commit()
            session.refresh(session_model)

            session_id = session_model.id
            print(f"[CSVImporter] Created session {session_id}: {session_uid}")

        # Import laps
        self.imported_records = 0

        for _, row in df.iterrows():
            lap_data = self._parse_lap_row(row)
            if lap_data:
                telemetry_writer.queue_lap(
                    driver_index=lap_data.get('driver_index', 0),
                    lap_number=lap_data.get('lap_number', 1),
                    lap_data={
                        **lap_data,
                        'session_id': session_id
                    }
                )
                self.imported_records += 1

        # Wait for writes to flush
        import time
        time.sleep(2)

        print(f"[CSVImporter] Imported {self.imported_records} laps from {csv_file.name}")
        return session_id

    def _parse_lap_row(self, row) -> Optional[dict]:
        """Parse a lap row from CSV (flexible column names)"""
        try:
            # Try common column names
            lap_data = {}

            # Driver info
            if 'driver_name' in row:
                lap_data['driver_name'] = str(row['driver_name'])
            elif 'Driver' in row:
                lap_data['driver_name'] = str(row['Driver'])

            if 'driver_index' in row:
                lap_data['driver_index'] = int(row['driver_index'])
            elif 'DriverIndex' in row:
                lap_data['driver_index'] = int(row['DriverIndex'])
            else:
                lap_data['driver_index'] = 0

            # Lap number
            if 'lap_number' in row:
                lap_data['lap_number'] = int(row['lap_number'])
            elif 'Lap' in row:
                lap_data['lap_number'] = int(row['Lap'])
            elif 'LapNumber' in row:
                lap_data['lap_number'] = int(row['LapNumber'])
            else:
                lap_data['lap_number'] = 1

            # Lap time
            if 'lap_time_ms' in row:
                lap_data['lap_time_ms'] = int(row['lap_time_ms'])
            elif 'LapTime' in row:
                lap_data['lap_time_ms'] = self._parse_time_string(row['LapTime'])
            elif 'Time' in row:
                lap_data['lap_time_ms'] = self._parse_time_string(row['Time'])

            # Sectors
            if 'sector1_time_ms' in row:
                lap_data['sector1_time_ms'] = int(row['sector1_time_ms'])
            if 'sector2_time_ms' in row:
                lap_data['sector2_time_ms'] = int(row['sector2_time_ms'])
            if 'sector3_time_ms' in row:
                lap_data['sector3_time_ms'] = int(row['sector3_time_ms'])

            # Position
            if 'position' in row:
                lap_data['position'] = int(row['position'])
            elif 'Position' in row:
                lap_data['position'] = int(row['Position'])

            # Tyre compound
            if 'tyre_compound' in row:
                lap_data['tyre_compound'] = str(row['tyre_compound'])
            elif 'Compound' in row:
                lap_data['tyre_compound'] = str(row['Compound'])
            elif 'TyreCompound' in row:
                lap_data['tyre_compound'] = str(row['TyreCompound'])

            # Tyre age
            if 'tyre_age_laps' in row:
                lap_data['tyre_age_laps'] = int(row['tyre_age_laps'])
            elif 'TyreAge' in row:
                lap_data['tyre_age_laps'] = int(row['TyreAge'])

            # Speed
            if 'speed_trap_speed' in row:
                lap_data['speed_trap_speed'] = float(row['speed_trap_speed'])
            elif 'Speed' in row:
                lap_data['speed_trap_speed'] = float(row['Speed'])

            # Only import if we have minimum required fields
            if 'lap_time_ms' in lap_data or 'LapTime' in str(row):
                return lap_data

            return None

        except Exception as e:
            print(f"[CSVImporter] Error parsing row: {e}")
            return None

    def _parse_time_string(self, time_str: str) -> int:
        """
        Convert time string to milliseconds

        Supports formats:
        - "1:23.456" → 83456 ms
        - "83456" → 83456 ms
        - "83.456" → 83456 ms
        """
        try:
            time_str = str(time_str).strip()

            # Already in ms
            if time_str.isdigit():
                return int(time_str)

            # Format: "1:23.456"
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds_parts = parts[1].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

                total_ms = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
                return total_ms

            # Format: "83.456"
            if '.' in time_str:
                seconds = float(time_str)
                return int(seconds * 1000)

            return 0

        except:
            return 0

    def import_directory(self, directory: str, default_track: str = "Unknown"):
        """
        Import all CSV files from a directory

        Args:
            directory: Path to directory containing CSV files
            default_track: Default track name if not in filename
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        csv_files = list(dir_path.glob('*.csv'))
        print(f"[CSVImporter] Found {len(csv_files)} CSV files in {directory}")

        for csv_file in csv_files:
            try:
                # Try to extract track name from filename
                track_name = default_track
                session_type = "Practice"

                if 'Grand Prix' in csv_file.name:
                    parts = csv_file.name.split('-')
                    if len(parts) >= 2:
                        track_name = parts[0].replace('2025-', '').strip()
                        session_type = parts[1].replace('.csv', '').strip()

                self.import_lap_csv(
                    str(csv_file),
                    track_name=track_name,
                    session_type=session_type
                )

            except Exception as e:
                print(f"[CSVImporter] Error importing {csv_file.name}: {e}")

        print(f"[CSVImporter] Import complete!")

    def import_sample_data(self):
        """
        Import sample data from /data/ directory

        Convenience method to load existing sample CSVs
        """
        data_dir = Path(__file__).parent.parent.parent / 'data'

        if not data_dir.exists():
            print(f"[CSVImporter] No /data/ directory found")
            return

        self.import_directory(str(data_dir), default_track="Melbourne")
