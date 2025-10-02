"""
Drift detection using Evidently
"""
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

def detect_drift(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report.as_dict()
