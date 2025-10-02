"""
Drift detection using Evidently
"""
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def detect_drift(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report.as_dict()

def drift_detected(report_dict: dict, threshold: float = 0.5) -> bool:
    return report_dict["metrics"][0]["result"]["dataset_drift"] and \
           report_dict["metrics"][0]["result"]["drift_share"] > threshold
