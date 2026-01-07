import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_fitpulse_data(days: int = 7, interval_minutes: int = 5) -> pd.DataFrame:
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    timestamps = []
    heart_rates = []
    sleep_hours = []
    step_counts = []
    calories_burned = []
    activity_types = []

    def get_activity_for_time(hour: int) -> str:
        if 0 <= hour < 6:
            return "Sleeping"
        elif 6 <= hour < 9:
            return "Walking"
        elif 9 <= hour < 12:
            return "Resting"
        elif 12 <= hour < 14:
            return "Running"
        elif 14 <= hour < 18:
            return "Walking"
        elif 18 <= hour < 20:
            return "Workout"
        else:
            return "Resting"

    total_minutes = days * 24 * 60
    interval = interval_minutes

    sleep_counter = 0.0

    for minute in range(0, total_minutes, interval):
        timestamp = start_time + timedelta(minutes=minute)
        timestamps.append(timestamp)

        hour = timestamp.hour
        activity = get_activity_for_time(hour)
        activity_types.append(activity)

        if activity == "Sleeping":
            base_hr = 55
        elif activity == "Resting":
            base_hr = 65
        elif activity == "Walking":
            base_hr = 85
        elif activity == "Running":
            base_hr = 130
        else:
            base_hr = 120

        hr = base_hr + np.random.normal(0, 4)
        heart_rates.append(int(max(45, min(180, hr))))

        if activity == "Sleeping":
            sleep_counter += interval / 60
            sleep_hours.append(round(sleep_counter, 2))
        else:
            sleep_counter = 0
            sleep_hours.append(0.0)

        if activity in ("Sleeping", "Resting"):
            steps = np.random.randint(0, 10)
        elif activity == "Walking":
            steps = np.random.randint(30, 70)
        elif activity == "Running":
            steps = np.random.randint(100, 150)
        else:
            steps = np.random.randint(60, 120)
        step_counts.append(steps)

        calories = steps * 0.04 + hr * 0.01
        calories_burned.append(round(calories, 2))

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate_bpm": heart_rates,
            "sleep_hours": sleep_hours,
            "step_count": step_counts,
            "calories_burned": calories_burned,
            "activity_type": activity_types,
        }
    )

    return df


def save_sample_data(output_path: str = "data/Sample_data.csv") -> pd.DataFrame:
    df = generate_fitpulse_data()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample data saved to {path}")
    return df


if __name__ == "__main__":
    save_sample_data()
