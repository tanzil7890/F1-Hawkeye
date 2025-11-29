"""
Data Export Module
Exports F1 telemetry data to various formats for analysis and ML training.

Exporters:
- CSVExporter: Human-readable CSV files for Excel/analysis
- ParquetExporter: ML-optimized columnar format (5-10x smaller, 10-100x faster queries)

Usage:
    from src.export import CSVExporter, ParquetExporter

    # Export to CSV
    csv_exp = CSVExporter()
    csv_exp.export_session(session_id=1, output_dir='exports/')

    # Export to Parquet (for ML)
    parquet_exp = ParquetExporter()
    parquet_exp.export_session(session_id=1, output_dir='ml_data/')
"""

from .csv_exporter import CSVExporter
from .parquet_exporter import ParquetExporter

__all__ = ['CSVExporter', 'ParquetExporter']
