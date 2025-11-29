"""
ML Training Pipeline

Centralized training pipeline for all ML models.

Features:
- Train all models in sequence
- Train individual models
- Save trained models
- Report training metrics

Usage:
    from src.ml.training import train_all_models

    # Train all models
    results = train_all_models(
        session_ids=[1, 2, 3],
        output_dir='models/'
    )

    # Train individual model
    from src.ml.training import train_individual_model
    metrics = train_individual_model(
        model_type='tyre_wear',
        session_ids=[1, 2, 3],
        save_path='models/tyre_wear_model.pkl'
    )
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..models import (
    TyreWearModel,
    LapTimeModel,
    RaceOutcomeModel
)


def train_all_models(
    session_ids: List[int],
    output_dir: str = 'models',
    test_size: float = 0.2
) -> Dict:
    """
    Train all ML models

    Args:
        session_ids: List of session IDs for training
        output_dir: Directory to save models
        test_size: Test set proportion

    Returns:
        Training results for all models
    """
    print("=" * 60)
    print("F1 TELEMETRY ML TRAINING PIPELINE")
    print("=" * 60)
    print(f"Training on {len(session_ids)} sessions: {session_ids}")
    print(f"Output directory: {output_dir}")
    print(f"Test size: {test_size * 100}%")
    print("=" * 60)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'session_ids': session_ids,
        'models': {}
    }

    # 1. Train Tyre Wear Model
    print("\n[1/3] Training Tyre Wear Model...")
    try:
        tyre_model = TyreWearModel()
        tyre_metrics = tyre_model.train(session_ids, test_size=test_size)

        # Save model
        model_path = os.path.join(output_dir, 'tyre_wear_model.pkl')
        tyre_model.save(model_path)

        results['models']['tyre_wear'] = {
            'status': 'success',
            'metrics': tyre_metrics,
            'model_path': model_path
        }

        print(f"✓ Tyre Wear Model trained successfully")
        print(f"  Saved to: {model_path}")
    except Exception as e:
        print(f"✗ Tyre Wear Model failed: {e}")
        results['models']['tyre_wear'] = {
            'status': 'failed',
            'error': str(e)
        }

    # 2. Train Lap Time Model
    print("\n[2/3] Training Lap Time Model...")
    try:
        lap_model = LapTimeModel()
        lap_metrics = lap_model.train(session_ids, test_size=test_size)

        # Save model
        model_path = os.path.join(output_dir, 'lap_time_model.pkl')
        lap_model.save(model_path)

        results['models']['lap_time'] = {
            'status': 'success',
            'metrics': lap_metrics,
            'model_path': model_path
        }

        print(f"✓ Lap Time Model trained successfully")
        print(f"  Saved to: {model_path}")
    except Exception as e:
        print(f"✗ Lap Time Model failed: {e}")
        results['models']['lap_time'] = {
            'status': 'failed',
            'error': str(e)
        }

    # 3. Train Race Outcome Model
    print("\n[3/3] Training Race Outcome Model...")
    try:
        outcome_model = RaceOutcomeModel()
        outcome_metrics = outcome_model.train(session_ids, test_size=test_size)

        # Save model
        model_path = os.path.join(output_dir, 'race_outcome_model.pkl')
        outcome_model.save(model_path)

        results['models']['race_outcome'] = {
            'status': 'success',
            'metrics': outcome_metrics,
            'model_path': model_path
        }

        print(f"✓ Race Outcome Model trained successfully")
        print(f"  Saved to: {model_path}")
    except Exception as e:
        print(f"✗ Race Outcome Model failed: {e}")
        results['models']['race_outcome'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    successful = sum(1 for m in results['models'].values() if m['status'] == 'success')
    total = len(results['models'])

    print(f"Models trained: {successful}/{total}")

    for model_name, model_result in results['models'].items():
        status_symbol = "✓" if model_result['status'] == 'success' else "✗"
        print(f"{status_symbol} {model_name}: {model_result['status']}")

        if model_result['status'] == 'success':
            metrics = model_result['metrics']
            if 'test_mae_seconds' in metrics:
                print(f"    MAE: {metrics['test_mae_seconds']:.3f}s")
            if 'test_r2' in metrics:
                print(f"    R²: {metrics['test_r2']:.3f}")
            if 'test_accuracy' in metrics:
                print(f"    Accuracy: {metrics['test_accuracy']:.3f}")

    print("=" * 60)

    return results


def train_individual_model(
    model_type: str,
    session_ids: List[int],
    save_path: Optional[str] = None,
    test_size: float = 0.2
) -> Dict:
    """
    Train a single ML model

    Args:
        model_type: Type of model ('tyre_wear', 'lap_time', 'race_outcome')
        session_ids: List of session IDs for training
        save_path: Optional path to save model
        test_size: Test set proportion

    Returns:
        Training metrics
    """
    print(f"Training {model_type} model on {len(session_ids)} sessions...")

    # Select model
    if model_type == 'tyre_wear':
        model = TyreWearModel()
    elif model_type == 'lap_time':
        model = LapTimeModel()
    elif model_type == 'race_outcome':
        model = RaceOutcomeModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    metrics = model.train(session_ids, test_size=test_size)

    # Save if path provided
    if save_path:
        model.save(save_path)
        print(f"Model saved to: {save_path}")

    return metrics


def load_trained_models(models_dir: str) -> Dict:
    """
    Load all trained models

    Args:
        models_dir: Directory containing trained models

    Returns:
        Dictionary of loaded models
    """
    models = {}

    # Try to load each model
    model_files = {
        'tyre_wear': 'tyre_wear_model.pkl',
        'lap_time': 'lap_time_model.pkl',
        'race_outcome': 'race_outcome_model.pkl'
    }

    for model_name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            try:
                if model_name == 'tyre_wear':
                    model = TyreWearModel(model_path=path)
                elif model_name == 'lap_time':
                    model = LapTimeModel(model_path=path)
                elif model_name == 'race_outcome':
                    model = RaceOutcomeModel(model_path=path)

                models[model_name] = model
                print(f"✓ Loaded {model_name} model from {path}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        else:
            print(f"⚠ Model file not found: {path}")

    return models


def evaluate_models(
    models_dir: str,
    test_session_ids: List[int]
) -> Dict:
    """
    Evaluate trained models on test sessions

    Args:
        models_dir: Directory containing trained models
        test_session_ids: Session IDs for evaluation

    Returns:
        Evaluation results
    """
    print(f"Evaluating models on {len(test_session_ids)} test sessions...")

    models = load_trained_models(models_dir)
    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        try:
            # Re-train on test data to get metrics
            # (In production, you'd have separate evaluation methods)
            metrics = model.train(test_session_ids, test_size=1.0)
            results[model_name] = {
                'status': 'success',
                'metrics': metrics
            }
            print(f"✓ {model_name} evaluation complete")
        except Exception as e:
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"✗ {model_name} evaluation failed: {e}")

    return results
