import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares

# Data
df = (pd.read_csv("datasets/COVID-19_Daily_Counts_of_Cases__Hospitalizations__and_Deaths_20250923.csv")
        .assign(date=lambda d: pd.to_datetime(d["date_of_interest"], format="%m/%d/%Y", errors="coerce"))
        .sort_values("date"))
df = df[["date","CASE_COUNT"]].copy()
df["CASE_COUNT"] = pd.to_numeric(df["CASE_COUNT"], errors="coerce").fillna(0.0)
df["cases_smooth"] = df["CASE_COUNT"].rolling(3, center=True, min_periods=1).mean()

# Constants
N, gamma = 8_300_000, 1/7
targets = [pd.Timestamp("2020-03-05"), pd.Timestamp("2020-04-17"),
           pd.Timestamp("2020-06-24"), pd.Timestamp("2020-09-02")]

# SIR (fractions)
def sir_rhs(y, t, beta, gamma):
    s, i, r = y
    return (-beta*s*i, beta*s*i - gamma*i, gamma*i)

def simulate(beta, s0, i0, r0, t):
    s, i, r = odeint(sir_rhs, (s0,i0,r0), t, args=(beta,gamma)).T
    return s, i, r, beta*s*i  # incidence (fraction/day)

# Fit on [df.min_date, cutoff]: params = (beta, i0, alpha)
def fit_window(cutoff, i0_guess=10/N):
    d = df[(df["date"] >= df["date"].min()) & (df["date"] <= cutoff)].reset_index(drop=True)
    t = (d["date"] - d["date"].iloc[0]).dt.days.values.astype(float)
    y = d["cases_smooth"].values.astype(float)

    def resid(theta):
        beta, i0, alpha = np.exp(theta)
        s0 = max(1 - i0, 1e-9)
        _, _, _, inc = simulate(beta, s0, i0, 0.0, t)
        return alpha * (inc * N) - y

    th0 = np.log([0.25, max(i0_guess, 1/N), 5.0])
    sol = least_squares(resid, th0, method="trf", loss="soft_l1", f_scale=50.0, max_nfev=4000)
    beta, i0, alpha = np.exp(sol.x)
    s0 = max(1 - i0, 1e-9)
    s, i, r, inc = simulate(beta, s0, i0, 0.0, t)
    return {"dates": d["date"].tolist(), "t": t, "y": y, "beta": beta, "alpha": alpha,
            "s": s, "i": i, "r": r, "inc_cases": inc * N}

# Forecast from last fitted state (+60d)
def forecast(fit, days=60):
    s0, i0, r0 = fit["s"][-1], fit["i"][-1], fit["r"][-1]
    tf = np.arange(0, days+1, dtype=float)
    s, i, r, inc = simulate(fit["beta"], s0, i0, r0, tf)
    return {"tf": tf, "inc_cases_f": inc * N}

# Run 4 models and plot
for idx, cutoff in enumerate(targets, 1):
    fit = fit_window(cutoff)
    fore = forecast(fit, 60)
    last = fit["dates"][-1]
    future_dates = [last + pd.Timedelta(days=int(x)) for x in fore["tf"][1:]]
    all_dates = fit["dates"] + future_dates
    model_cases = fit["alpha"] * np.concatenate([fit["inc_cases"], fore["inc_cases_f"][1:]])

    span = df[(df["date"] >= fit["dates"][0]) & (df["date"] <= future_dates[-1])]
    plt.figure(figsize=(9,4))
    plt.title(f"Model {idx}: Fit through {cutoff.date()} and 60-day forecast\n"
              f"β={fit['beta']:.3f}, γ={gamma:.3f}, α={fit['alpha']:.2f}")
    plt.plot(span["date"], span["cases_smooth"], color="gray", label="Actual daily cases")
    plt.plot(all_dates, model_cases, color="purple", label="Predicted (fit + forecast)")
    plt.axvline(last, color="blue", linestyle="--", label="Forecast start")
    plt.xlabel("Date"); plt.ylabel("Daily cases"); plt.legend(); plt.tight_layout(); plt.show()
