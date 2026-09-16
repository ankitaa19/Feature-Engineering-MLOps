"""Generate reproducible CSV datasets required by the assignment notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS_DIR = ROOT / "Feature-Engineering-Assignments"
RNG = np.random.default_rng(42)


def inject_missing(series: pd.Series, mask: np.ndarray) -> pd.Series:
    out = series.copy()
    out.loc[mask] = np.nan
    return out


def generate_student_performance() -> None:
    n = 600
    student_id = np.arange(10001, 10001 + n)
    enrollment_date = pd.date_range("2024-01-01", periods=n, freq="D")

    city = RNG.choice(
        ["Mumbai", "Pune", "Delhi", "Bengaluru", "Hyderabad"],
        size=n,
        p=[0.28, 0.18, 0.20, 0.18, 0.16],
    )
    course = RNG.choice(
        ["Data Science", "Full Stack", "Cloud", "AI Foundations"],
        size=n,
        p=[0.36, 0.25, 0.20, 0.19],
    )
    batch_type = RNG.choice(["Weekday", "Weekend", "Hybrid"], size=n, p=[0.50, 0.32, 0.18])
    income_bracket = RNG.choice(
        ["Low", "Middle", "Upper-middle", "High"],
        size=n,
        p=[0.18, 0.48, 0.24, 0.10],
    ).astype(object)

    attendance_pct = np.clip(RNG.normal(78, 11, size=n), 35, 100)
    weekly_study_hours = np.clip(RNG.lognormal(mean=2.0, sigma=0.55, size=n), 1, 35)
    prev_exam_score = np.clip(RNG.normal(67, 14, size=n), 20, 98)
    doubt_sessions_attended = RNG.poisson(lam=3.2, size=n)
    mock_test_3 = np.clip(
        0.45 * prev_exam_score + 1.7 * weekly_study_hours + RNG.normal(15, 8, size=n),
        20,
        100,
    )

    city_bonus = pd.Series(city).map(
        {"Mumbai": 2.0, "Pune": 1.2, "Delhi": 1.4, "Bengaluru": 2.2, "Hyderabad": 1.0}
    ).to_numpy()
    course_bonus = pd.Series(course).map(
        {"Data Science": 2.0, "Full Stack": 1.4, "Cloud": 1.7, "AI Foundations": 1.1}
    ).to_numpy()
    final_score = np.clip(
        0.36 * prev_exam_score
        + 0.24 * attendance_pct
        + 1.55 * np.log1p(weekly_study_hours) * 10
        + 0.9 * doubt_sessions_attended
        + 0.16 * mock_test_3
        + city_bonus
        + course_bonus
        + RNG.normal(0, 5.5, size=n),
        0,
        100,
    )

    feedback_text = RNG.choice(
        [
            "Needs more revision",
            "Consistent performer",
            "Improving steadily",
            "Excellent classroom participation",
            "Irregular practice",
        ],
        size=n,
        p=[0.18, 0.30, 0.25, 0.16, 0.11],
    ).astype(object)

    study_missing = RNG.random(n) < 0.11
    prev_missing = (attendance_pct < 67) & (RNG.random(n) < 0.24)
    mock_missing = (mock_test_3 < 55) & (RNG.random(n) < 0.38)
    income_missing = RNG.random(n) < 0.09
    feedback_missing = RNG.random(n) < 0.07

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
            "city": city,
            "course": course,
            "batch_type": batch_type,
            "attendance_pct": attendance_pct.round(2),
            "weekly_study_hours": inject_missing(
                pd.Series(weekly_study_hours.round(2)), study_missing
            ),
            "prev_exam_score": inject_missing(pd.Series(prev_exam_score.round(2)), prev_missing),
            "doubt_sessions_attended": doubt_sessions_attended,
            "mock_test_3": inject_missing(pd.Series(mock_test_3.round(2)), mock_missing),
            "income_bracket": inject_missing(pd.Series(income_bracket), income_missing),
            "feedback_text": inject_missing(pd.Series(feedback_text), feedback_missing),
            "final_score": final_score.round(2),
        }
    )

    for out_path in [
        ROOT / "student_performance_raw.csv",
        ASSIGNMENTS_DIR / "student_performance_raw.csv",
    ]:
        df.to_csv(out_path, index=False)


def generate_daily_batch_performance() -> None:
    n = 730
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    dow = dates.dayofweek.to_numpy()

    discount_pct = RNG.choice([0, 5, 10, 15, 20, 25], size=n, p=[0.30, 0.20, 0.18, 0.15, 0.12, 0.05])
    is_holiday = ((dates.month == 12) & (dates.day >= 20)).astype(int)
    competitor_promo_flag = RNG.binomial(1, 0.16, size=n)
    weekend = np.isin(dow, [5, 6]).astype(int)

    base_spend = 3200 + 420 * np.sin(2 * np.pi * t / 30) + 260 * weekend
    marketing_spend = np.clip(base_spend + 65 * discount_pct + RNG.normal(0, 240, size=n), 1200, None)
    website_visits = np.clip(
        900
        + 0.28 * marketing_spend
        + 18 * discount_pct
        + 120 * weekend
        - 80 * competitor_promo_flag
        + RNG.normal(0, 85, size=n),
        300,
        None,
    )
    website_visits_copy = website_visits + RNG.normal(0, 3, size=n)
    lag2_spend = pd.Series(marketing_spend).shift(2).bfill().to_numpy()

    new_enrollments = np.clip(
        30
        + 0.0026 * lag2_spend
        + 0.014 * website_visits
        + 0.82 * discount_pct
        + 4.5 * weekend
        + 0.018 * lag2_spend * discount_pct / 100
        + 0.017 * t
        + 6 * is_holiday
        - 5.5 * competitor_promo_flag
        + RNG.normal(0, 4.5, size=n),
        5,
        None,
    )

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "new_enrollments": np.rint(new_enrollments).astype(int),
            "marketing_spend": marketing_spend.round(2),
            "discount_pct": discount_pct,
            "website_visits": np.rint(website_visits).astype(int),
            "website_visits_copy": website_visits_copy.round(2),
            "competitor_promo_flag": competitor_promo_flag,
            "is_holiday": is_holiday,
            "random_noise_1": RNG.normal(0, 1, size=n).round(4),
            "random_noise_2": RNG.choice(["A", "B", "C", "D"], size=n, p=[0.60, 0.24, 0.12, 0.04]),
        }
    )

    for out in [
        ROOT / "data" / "raw",
        ASSIGNMENTS_DIR / "data" / "raw",
    ]:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "daily_batch_performance.csv", index=False)


def generate_pca_regression_data() -> None:
    n = 800
    n_features = 60
    n_latent = 6

    latent = RNG.normal(size=(n, n_latent))
    loadings = RNG.normal(size=(n_latent, n_features))
    X = latent @ loadings + RNG.normal(0, 0.18, size=(n, n_features))
    coefs = np.array([24, -18, 14, 10, -8, 6])
    score = 72 + latent @ coefs + RNG.normal(0, 3.0, size=n)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, n_features + 1)])
    df.insert(0, "student_id", np.arange(20001, 20001 + n))
    df["score"] = score.round(3)
    for out_path in [
        ROOT / "pca_regression_data.csv",
        ASSIGNMENTS_DIR / "pca_regression_data.csv",
    ]:
        df.to_csv(out_path, index=False)


def generate_customer_churn() -> None:
    n = 600

    tenure_months = RNG.integers(1, 73, size=n)
    monthly_charges = np.clip(RNG.normal(72, 18, size=n), 25, 140)
    contract_type = RNG.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.54, 0.28, 0.18],
    )
    support_calls = RNG.poisson(1.6, size=n)

    contract_risk = pd.Series(contract_type).map(
        {"Month-to-month": 0.9, "One year": -0.25, "Two year": -0.9}
    ).to_numpy()
    logits = (
        -2.05
        - 0.022 * tenure_months
        + 0.017 * (monthly_charges - 70)
        + 0.33 * support_calls
        + contract_risk
        + RNG.normal(0, 0.45, size=n)
    )
    probs = 1 / (1 + np.exp(-logits))
    churn = RNG.binomial(1, probs)

    # Keep the dataset close to the assignment description: roughly 149 churned
    # customers and 451 non-churned customers with post-cancellation fields empty.
    target_churn = 149
    if churn.sum() != target_churn:
        order = np.argsort(probs)
        churn[:] = 0
        churn[order[-target_churn:]] = 1

    days_since_cancellation = np.full(n, np.nan)
    final_bill_amount = np.full(n, np.nan)
    churn_idx = np.where(churn == 1)[0]
    days_since_cancellation[churn_idx] = RNG.integers(1, 180, size=len(churn_idx))
    final_bill_amount[churn_idx] = (
        monthly_charges[churn_idx] * RNG.uniform(0.8, 1.8, size=len(churn_idx))
        + 18 * support_calls[churn_idx]
        + RNG.normal(0, 12, size=len(churn_idx))
    )

    df = pd.DataFrame(
        {
            "customer_id": np.arange(50001, 50001 + n),
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "contract_type": contract_type,
            "support_calls": support_calls,
            "days_since_cancellation": days_since_cancellation,
            "final_bill_amount": np.round(final_bill_amount, 2),
            "churn": churn,
        }
    )

    for out_path in [
        ROOT / "customer_churn_a5.csv",
        ASSIGNMENTS_DIR / "customer_churn_a5.csv",
    ]:
        df.to_csv(out_path, index=False)


def main() -> None:
    generate_student_performance()
    generate_daily_batch_performance()
    generate_pca_regression_data()
    generate_customer_churn()
    print("Generated assignment datasets:")
    for path in [
        ROOT / "student_performance_raw.csv",
        ASSIGNMENTS_DIR / "student_performance_raw.csv",
        ROOT / "data" / "raw" / "daily_batch_performance.csv",
        ASSIGNMENTS_DIR / "data" / "raw" / "daily_batch_performance.csv",
        ROOT / "pca_regression_data.csv",
        ASSIGNMENTS_DIR / "pca_regression_data.csv",
        ROOT / "customer_churn_a5.csv",
        ASSIGNMENTS_DIR / "customer_churn_a5.csv",
    ]:
        print(f" - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
