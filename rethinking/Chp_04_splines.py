import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo 
    import pymc as pm
    from importlib import reload
    import util 
    from pathlib import Path
    import arviz as az
    from scipy.stats import norm
    #import arviz.plots as azp

    df = pd.read_csv("./rethinking/data/bus19_durations.csv")


    # Mean duration for each departure time
    mean_duration = (
        df.groupby("departure_hour")["duration_seconds"]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(12, 5))

    plt.scatter(
            df["departure_hour"],
            df["duration_seconds"],
            alpha=0.5,
            s=15
    )

    # draws the mean line
    plt.plot(
        mean_duration.index,
        mean_duration.values,
        marker="o",
        color="red",
        linewidth=2,
        label="Mean duration"
    )


    plt.xticks(
            np.arange(0, 25, 1),
            [f"{h:02d}:00" for h in range(25)],
            rotation=45
        )

    plt.xlabel("Departure time")
    plt.ylabel("Journey duration (seconds)")
    plt.title(f"Jonction → Vernier | Line 19")

    plt.grid(alpha=0.3)

    #plt.ylim(top=4000)
    plt.ylim(1500, 2500)

    plt.tight_layout()
    plt.show()
    return Path, az, df, mo, norm, np, pd, plt, pm, reload, util


@app.cell
def _(df, mo, np):
    df.head()

    mo.md(r"""
    $$
    \boxed{
    duration_i \sim \text{Normal}(\mu_i,\sigma)
    }
    $$

    $$
    \boxed{
    \mu_i = \alpha + \sum_{j=1}^{K} \beta_j B_j(x_i)
    }
    $$

    Where:
    * $x_i$ is departure time in hours, from 0 to 24.
    * $B_j(x_i)$ is the value of spline basis function $j$.
    * $\beta_j$ is its coefficient.
    * $\sigma$ is residual journey-duration variation.
    """)

    hours = sorted(df["departure_hour"].unique())
    print(hours[0:5])
    print(np.min(hours))
    print(np.max(hours))
    return


@app.cell
def _(np, pd):
    # Exploring basis function

    from patsy import dmatrix, build_design_matrices

    def explore_b_spline():
        x_slots = np.arange(48) / 2  # 0, 0.5, ..., 23.5
    
        B_spline = dmatrix(
            "bs(x, df=8, degree=1, include_intercept=False, "
            "lower_bound=-2.9375, upper_bound=26.4375) - 1",
            {"x": x_slots}
        )
    
    
        # Preserve Patsy's metadata
        design_info = B_spline.design_info
    
        B_slots = build_design_matrices(
            [design_info],
            {"x": x_slots}
        )[0]
    
        B_slots = np.asarray(B_slots, dtype=float)
    
        B_df = pd.DataFrame(
            B_slots,
            index=[f"{int(x):02d}:{int((x % 1) * 60):02d}" for x in x_slots],
            columns=[f"B{i+1}" for i in range(B_slots.shape[1])]
        )
    
        #print(B_df.round(3).to_string())
        k1, k7 = np.quantile(x_slots, [1/8, 7/8])
    
        # Set boundaries so the basis functions have value 0.5
        lower_bound = -k1
        upper_bound = max(x_slots) * 2 - k7
    
        print(f"LB {lower_bound} UB {upper_bound} k1 {k1} k7 {k7}")
    
        print(B_slots[:5])   # first 5 rows
        print(B_slots[-5:])  # last 5 rows

    explore_b_spline()
    return build_design_matrices, dmatrix


@app.cell
def _(df, dmatrix, np):

    # Extract data
    x = df["departure_hour"].to_numpy(dtype=float)
    y = df["duration_seconds"].to_numpy(dtype=float)

    assert np.all(np.isfinite(x) & np.isfinite(y))


    # Create cubic B-spline basis
    # Construct 8 cubic B-spline basis functions over the 0–24 hour domain.
    # Each basis function is a smooth, piecewise cubic polynomial (degree=3).
    # include_intercept=False omits the constant basis component.
    # Subtracting 1 removes the intercept column from the resulting design matrix,
    # since the regression model already includes its own intercept.
    B_spline = dmatrix(
        "bs(x, df=8, degree=1, include_intercept=False, "
        "lower_bound=-2.9375, upper_bound=26.4375) - 1",
        {"x": x}
    )


    # Preserve Patsy's metadata
    design_info = B_spline.design_info

    B = np.asarray(B_spline, dtype=float)

    print("Observations:", len(y))
    print("Basis shape:", B.shape)
    print(len(x))
    print(len(y))
    B
    return B, design_info, x, y


@app.cell
def _(B, Path, pm, reload, util, y):

    # in case we are editinng
    reload(util)

    with pm.Model() as model:

        # Overall average journey duration
        alpha = pm.Normal(
            "alpha",
            mu=y.mean(),
            sigma=500
        )

        # Spline coefficients
        # we learn 8 values
        w = pm.Normal(
            "w",
            mu=0,
            sigma=250,
            shape=B.shape[1]
        )

        # Expected duration
        # B has len(x) ; but we are not learning B
        mu = pm.Deterministic(
            "mu",
            alpha + pm.math.dot(B, w)
        )

        # Observation noise
        sigma = pm.HalfNormal(
            "sigma",
            sigma=300
        )

        # Likelihood
        duration = pm.Normal(
            "duration",
            mu=mu,
            sigma=sigma,
            observed=y
        )

        idata = util.sample_or_load(
                    lambda: pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=42
        ), Path(f"chap4_splines_1.trace"))
    return (idata,)


@app.cell
def _(B, az, build_design_matrices, design_info, idata, norm, np):
    # Regular grid of departure times
    x_grid = np.linspace(0, 24, 200)

    # Construct the same spline basis for prediction
    B_grid = build_design_matrices(
        [design_info],
        {"x": x_grid}
    )[0]

    B_grid = np.asarray(B_grid, dtype=float)

    # Extract posterior samples
    # Original shapes: (4, 2000) and (4, 2000, 8)
    alpha_samples = idata.posterior["alpha"].values.reshape(-1)
    w_samples = idata.posterior["w"].values.reshape(-1, B.shape[1])
    sigma_samples = idata.posterior["sigma"].values.reshape(-1)

    print(alpha_samples.shape)  # (8000,)
    print(w_samples.shape)      # (8000, 8)
    print(sigma_samples.shape)  # (8000,)

    # Central 89% posterior predictive interval:
    # 5.5% in each tail, leaving 89% in the middle.
    tail = (1 - 0.89) / 2

    # Corresponding Normal quantile (approximately 1.598).
    z = norm.ppf(1 - tail)

    sigma_samples = idata.posterior["sigma"].values.reshape(-1)

    # Calculate expected duration for every posterior draw
    # Transpose B_grid to align matrix multiplication dimensions
    # B_grid.T: (8, 200)
    #
    # Matrix multiplication:
    # (8000, 8) @ (8, 200) = (8000, 200)
    #
    #
    # Broadcasting adds the intercept to each of the 200 predictions
    # Expected duration for every posterior draw
    # (8000, 8) @ (8, 200) = (8000, 200)
    mu_samples = (
        alpha_samples[:, None]
        + np.matmul(w_samples, B_grid.T)
    )


    # Pair each sampled mean curve with its corresponding sigma.
    lower = np.quantile(
        mu_samples - z * sigma_samples[:, None],
        tail,
        axis=0
    )

    upper = np.quantile(
        mu_samples + z * sigma_samples[:, None],
        1 - tail,
        axis=0
    )


    # Combine chains and draws
    mu_samples = mu_samples.reshape(-1, len(x_grid))

    # Posterior mean and 89% equal-tailed interval
    mu_mean = mu_samples.mean(axis=0)
    mu_interval = np.quantile(
        mu_samples,
        [0.055, 0.945],
        axis=0
    )

    print(mu_mean.shape)       # (200,)
    print(mu_interval.shape)  # (200, 2)

    print("x_grid:", x_grid.shape)
    print("mu_samples:", mu_samples.shape)
    print("mu_interval:", mu_interval.shape)
    print("mu_interval[0]:", mu_interval[0].shape)
    print("mu_interval[1]:", mu_interval[1].shape)

    summary = az.summary(
        idata,
        var_names=["w"],
        kind="stats"
    )

    print(summary[["mean", "sd"]])

    print(az.summary(idata, var_names=["sigma"]))
    return B_grid, lower, mu_interval, mu_mean, mu_samples, upper, x_grid


@app.cell
def _(B_grid, plt, x_grid):
    plt.figure(figsize=(12, 4))

    for j in range(B_grid.shape[1]):
        plt.plot(x_grid, B_grid[:, j], label=f"B{j}")

    plt.legend()
    plt.xlabel("Departure hour")
    plt.ylabel("Basis value")
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(
    lower,
    mu_interval,
    mu_mean,
    mu_samples,
    np,
    pd,
    plt,
    upper,
    x,
    x_grid,
    y,
):

    plt.figure(figsize=(12, 5))

    # Individual journeys
    plt.scatter(
        x,
        y,
        alpha=0.2,
        s=12,
        label="Observed journeys"
    )

    # Posterior uncertainty


    print(x_grid.shape, lower.shape, upper.shape)
    # Posterior standard deviation of the fitted function
    mu_sd = mu_samples.std(axis=0)

    print("Mean SD:", mu_sd.mean())
    print("Min SD:", mu_sd.min())
    print("Max SD:", mu_sd.max())

    plt.fill_between(
        x_grid,
        lower,
        upper,
        alpha=0.3,
        color="green",
        label="89% credible interval"
    )



    # Posterior mean
    plt.plot(
        x_grid,
        mu_mean,
        color="green",
        linewidth=2,
        label="Posterior mean"
    )

    # Observed mean for each departure slot
    observed_mean = (
        pd.DataFrame({"x": x, "y": y})
        .groupby("x")["y"]
        .mean()
        .sort_index()
    )

    plt.plot(
        observed_mean.index,
        observed_mean.values,
        color="red",
        marker="o",
        linewidth=2,
        label="Observed mean"
    )



    plt.xlim(0, 24)
    plt.ylim(1500, 4000)

    plt.xticks(
        np.arange(0, 24.5, 0.5),
        [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)] + ["24:00"],
        rotation=90
    )

    print("Mean interval width:", np.mean(mu_interval[1] - mu_interval[0]))
    print("Min width:", np.min(mu_interval[1] - mu_interval[0]))
    print("Max width:", np.max(mu_interval[1] - mu_interval[0]))

    plt.xlabel("Departure time")
    plt.ylabel("Journey duration (seconds)")
    plt.title("Bayesian B-spline regression")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(lower, np, upper, x, x_grid, y):
    # seeing how many observed are actually in this interval
    # Interpolate interval bounds for each observed departure time
    lower_at_x = np.interp(x, x_grid, lower)
    upper_at_x = np.interp(x, x_grid, upper)

    # Check whether each observed duration falls inside its interval
    inside = (y >= lower_at_x) & (y <= upper_at_x)

    # Count observations
    n_inside = inside.sum()
    n_total = len(y)
    percentage = 100 * n_inside / n_total

    print(f"Inside interval: {n_inside:,} / {n_total:,}")
    print(f"Outside interval: {n_total - n_inside:,}")
    print(f"Percentage inside: {percentage:.2f}%")
    return


if __name__ == "__main__":
    app.run()
