import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np 
    import marimo as mo 
    import pandas as pd 
    import matplotlib.pyplot as plt
    import util 
    from importlib import reload 
    from pathlib import Path 
    import pymc as pm
    import numpy as np
    import arviz as az
    import arviz_plots as azp
    import scipy.stats as stats
    import statsmodels.formula.api as smf

    reload(util)
    rng = np.random.default_rng(42)

    def discrete_ex():
        n = 1000
    
        Z = rng.binomial(n=1, p=0.5, size=n)
    
        X = rng.binomial(n=1, p=(1 - Z) * 0.1 + Z * 0.9)
    
        Y = rng.binomial(n=1, p=(1 - Z) * 0.1 + Z * 0.9)
    
    
    
        print(pd.crosstab(
            X,
            Y,
             rownames=["X"],
             colnames=["Y"],
            # margins=True
        ))
    
        print("X totals:", np.bincount(X))
        print("Y totals:", np.bincount(Y))
    
        print(np.corrcoef(X, Y)[0, 1])
        # Approximately 0.63
    
        print(pd.crosstab(
            X[Z == 0],
            Y[Z == 0],
             rownames=["X"],
             colnames=["Y"],
            # margins=True
        ))
    
        print( np.corrcoef(X[Z == 0], Y[Z == 0])[0, 1] )
    
    
        print(pd.crosstab(
            X[Z == 1],
            Y[Z == 1],
             rownames=["X"],
             colnames=["Y"],
            # margins=True
        ))
    
        print( np.corrcoef(X[Z == 1], Y[Z == 1])[0, 1] )
    
        # Draw the mediation graph
        g = util.draw_causal_graph(
            edge_list=[
                ("Z", "X"),
                ("Z", "Y"),
            ],
            graph_direction="LR"
        )

        return g 
    g = discrete_ex()

    # X \not\!\perp\!\!\!\perp Y
        # X \perp\!\!\!\perp Y


    g
    return Path, az, azp, mo, np, pd, plt, pm, reload, rng, smf, stats, util


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
        Y \not\!\perp\!\!\!\perp X
        \\
        X \perp\!\!\!\perp Y \mid Z
    $$
    # Fork
    X and Y are associated, but once stratified/controled by Z, they are not
    """)
    return


@app.cell
def _(mo, np, plt):


    def cont_example():
        N = 700
        rng = np.random.default_rng(42)
    
        Z = rng.binomial(1, 0.5, N)
        X = rng.normal(2 * Z - 1)
        Y = rng.normal(2 * Z - 1)
    
        # Scatter plot
        plt.scatter(
            X[Z == 0], Y[Z == 0],
            facecolors="none", edgecolors="blue",
            linewidths=1.5, label="Z = 0"
        )
    
        plt.scatter(
            X[Z == 1], Y[Z == 1],
            color="red", label="Z = 1"
        )
    
        # Regression lines
        for z, color in [(0, "blue"), (1, "red")]:
            slope, intercept = np.polyfit(X[Z == z], Y[Z == z], 1)
            x_line = np.linspace(X.min(), X.max(), 100)
            plt.plot(x_line, intercept + slope * x_line, color=color, linewidth=3)
    
        # Overall regression line
        slope, intercept = np.polyfit(X, Y, 1)
        x_line = np.linspace(X.min(), X.max(), 100)
        plt.plot(x_line, intercept + slope * x_line, color="black", linewidth=3)
    
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    mo.output.append(mo.md("Horizontal is no correlation"))

    cont_example()
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(Path, pd):
    #import pandas as pd

    # Load data
    df = pd.read_csv(
        Path("./rethinking/data/WaffleDivorce.csv"),
        sep=";"
    )


    return (df,)


@app.cell
def _(df, plt):
    def divorce_vs_marriage():

        fig, ax = plt.subplots(figsize=(9, 6))
    
        south = df["South"] == 1
        north = ~south
    
        # Non-Southern states: filled circles
        ax.scatter(
            df.loc[north, "Marriage"],
            df.loc[north, "Divorce"],
            color="black",
            facecolors="none",
            edgecolors="blue",
            linewidths=1.8,
            s=90,
            label="Northern states"
        )
    
        # Southern states: red open circles
        ax.scatter(
            df.loc[south, "Marriage"],
            df.loc[south, "Divorce"],
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            s=90,
            label="Southern states"
        )
    
        # State abbreviations
        for _, row in df.iterrows():
            ax.annotate(
                row["Loc"],
                (row["Marriage"], row["Divorce"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8
            )
    
        ax.set_xlabel("Marriage rate")
        ax.set_ylabel("Divorce rate")
        ax.set_xlim(14, 31)
        ax.set_ylim(5.5, 12.5)
        ax.legend()
        return plt.show()

    def divorce_vs_median_age():
        fig, ax = plt.subplots(figsize=(9, 6))

        south = df["South"] == 1
        north = ~south

        ax.scatter(
            df.loc[south, "MedianAgeMarriage"],
            df.loc[south, "Divorce"],
            color="black",
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            s=90,
            label="Southern states"
        
        )
        ax.scatter(
            df.loc[north, "MedianAgeMarriage"],
            df.loc[north, "Divorce"],
            color="black",
            facecolors="none",
            edgecolors="blue",
            linewidths=1.8,
            s=90,
            label="Northern states"
        
        )

        for _, row in df.iterrows():
            ax.annotate(
                row["Loc"],
                (row["MedianAgeMarriage"], row["Divorce"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8
            )

        ax.set_xlabel("Median age of marriage")
        ax.set_ylabel("Divorce rate")
        return plt.show()

    def marriage_vs_median_age():
        fig, ax = plt.subplots(figsize=(9, 6))

        south = df["South"] == 1
        north = ~south

    
        ax.scatter(
            df.loc[south, "MedianAgeMarriage"],
            df.loc[south, "Marriage"],
            color="black",
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            s=90,
            label="Southern states"
        
        )
        ax.scatter(
            df.loc[north, "MedianAgeMarriage"],
            df.loc[north, "Marriage"],
            color="black",
            facecolors="none",
            edgecolors="blue",
            linewidths=1.8,
            s=90,
            label="Northern states"
        
        )

         
        for _, row in df.iterrows():
            ax.annotate(
                row["Loc"],
                (row["MedianAgeMarriage"], row["Marriage"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8
            )

        ax.set_xlabel("Median age of marriage")
        ax.set_ylabel("Divorce rate")
        return plt.show()

    divorce_vs_marriage()
    divorce_vs_median_age()
    marriage_vs_median_age()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * D Divorce
    * A Age
    * M Marriage

    Every value of A produces different relationship between D and M

    $$
    D_i \sim \text{Normal}(\mu_i, \sigma)
    \\
    \mu_i = \alpha + \beta_M M_i + \beta_A A_i
    \\
    \mu_i = \underbrace{\alpha + \beta_A A_i}_{intercept} + \beta_M M_i
    $$
    """)
    return


@app.cell
def _(np, plt, rng):
    def _():
        # Prior predictive simulation
        n = 20

        a = rng.normal(0, 0.2, n)

        # kept under 1 otherwise var. explained fully 
        bM = rng.normal(0, .5, n)
        bA = rng.normal(0, .5, n)

        fig, ax = plt.subplots()

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        ax.set_xlabel("Median age of marriage (standardized)")
        ax.set_ylabel("Divorce rate (standardized)")

        Aseq = np.linspace(-3, 3, 30)

        for i in range(n):
            mu = a[i] + bA[i] * Aseq
            ax.plot(Aseq, mu, linewidth=2, color="red")
        return plt.show()


    _()
    return


@app.cell
def _(Path, az, azp, df, pm, util):
    # mean 0, standard deviation of 1
    # 68 -> 95 -> 99.7
    D = (df["Divorce"] - df["Divorce"].mean()) / df["Divorce"].std()
    M = (df["Marriage"] - df["Marriage"].mean()) / df["Marriage"].std()

    A = (df["MedianAgeMarriage"] - df["MedianAgeMarriage"].mean()) / df["MedianAgeMarriage"].std()

    with pm.Model() as divorce_model:

        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=0.2)
        beta_M = pm.Normal("beta_M", mu=0, sigma=0.5)
        beta_A = pm.Normal("beta_A", mu=0, sigma=0.5)

        sigma = pm.Exponential("sigma", lam=1)

        # Linear predictor
        mu = alpha + beta_M * M + beta_A * A

        # Likelihood
        D_obs = pm.Normal("D_obs", mu=mu, sigma=sigma, observed=D)

        # Sample posterior
        trace = util.sample_or_load(
                        lambda: pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            random_seed=42
        ), Path("marriage_divorce.trace"))

    az.summary(trace, ci_prob=0.89)

    pc = azp.plot_forest(
        trace,
        var_names=["alpha", "beta_M", "beta_A", "sigma"],
        combined=True,
        ci_kind="hdi",
        ci_probs=(0.50, 0.89),
    
    )

    axes = pc.get_viz("plot")

    for ax in axes.values.flat:
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    pc
    return A, trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    p(D \mid \text{do}(M))
    $$

    means the distribution of D when we intervene ("do") M
    """)
    return


@app.cell
def _(util):
    print("No intervention")

    util.draw_causal_graph(
            edge_list=[
                ("A", "M"),
                ("A", "D"),
                ("M", "D")
            ],
            graph_direction="LR"
        )
    return


@app.cell
def _(util):
    print("do(M)")

    # A-> M removed because we are setting M
    util.draw_causal_graph(
            edge_list=[
                ("A", "D"),
                ("M", "D")
            ],
            graph_direction="LR"
        )
    return


@app.cell
def _(A, np, plt, trace):
    def _():
        rng = np.random.default_rng(42)

        n = 1_000

        post = trace["posterior"].to_dataset().stack(sample=("chain", "draw"))

        alpha = post["alpha"].values
        beta_M = post["beta_M"].values
        beta_A = post["beta_A"].values
        sigma = post["sigma"].values

        As = rng.choice(A.to_numpy(), size=n, replace=True)

        # Select 1,000 posterior draws
        idx = rng.choice(len(alpha), size=n, replace=False)

        alpha_s = alpha[idx]
        beta_M_s = beta_M[idx]
        beta_A_s = beta_A[idx]
        sigma_s = sigma[idx]

        DM0 = rng.normal(
            loc=alpha_s + beta_M_s * 0 + beta_A_s * As,
            scale=sigma_s,
        )

        DM1 = rng.normal(
            loc=alpha_s + beta_M_s * 1 + beta_A_s * As,
            scale=sigma_s,
        )

        M10_contrast = DM1 - DM0

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(M10_contrast, bins=40, density=True, alpha=0.4)
        ax.set_xlabel("Effect of a 1 SD increase in marriage rate")
        ax.set_ylabel("Density")
        ax.set_title("Posterior distribution of the contrast")
        return plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    Y \not\!\perp\!\!\!\perp X
    \\
    X \perp\!\!\!\perp Y \mid Z
    $$

    # Pipe

    X and Y are not independent, given Z, they are independent
    """)
    return


@app.cell
def _(util):
    # Draw the mediation graph
    util.draw_causal_graph(
            edge_list=[
                ("X", "Z"),
                ("Z", "Y"),
            ],
            graph_direction="LR"
        )

    return


@app.cell
def _(np, pd, rng):
    def pipe_ex():
        n = 1000
    
        X = rng.binomial(n=1, p=0.5, size=n)
    
        Z = rng.binomial(n=1, p=(1 - X) * 0.1 + X * 0.9)
    
        Y = rng.binomial(n=1, p=(1 - Z) * 0.1 + Z * 0.9)

        print(pd.crosstab(
                X,
                Y,
                 rownames=["X"],
                 colnames=["Y"],
                # margins=True
            ))
    
        
        print(np.corrcoef(X, Y)[0, 1])

        # controlling for Z shows X and Y not correlated
        print(pd.crosstab(
                X[Z == 0],
                Y[Z == 0],
                 rownames=["X"],
                 colnames=["Y"],
                # margins=True
            ))
    
        print( np.corrcoef(X[Z == 0], Y[Z == 0])[0, 1] )


        print(pd.crosstab(
            X[Z == 1],
            Y[Z == 1],
             rownames=["X"],
             colnames=["Y"],
            # margins=True
        ))

        print( np.corrcoef(X[Z == 1], Y[Z == 1])[0, 1] )


    pipe_ex()
    return


@app.cell
def _(mo, np, plt):


    def pipe_cont_example():
        N = 700
        rng = np.random.default_rng(42)

        X = rng.normal(loc=0, scale=1, size=N)
        # roughly 10% for X==-2 or 90% for X==2
        p = 1 / (1 + np.exp(-X))
        Z = rng.binomial(1, p, size=N)
        Y = rng.normal(loc=2 * Z - 1, scale=1)
    
        # Scatter plot, controlling for Z
        plt.scatter(
            X[Z == 0], Y[Z == 0],
            facecolors="none", edgecolors="blue",
            linewidths=1.5, label="Z = 0"
        )
    
        plt.scatter(
            X[Z == 1], Y[Z == 1],
            color="red", label="Z = 1"
        )
    
        # Regression lines
        for z, color in [(0, "blue"), (1, "red")]:
            slope, intercept = np.polyfit(X[Z == z], Y[Z == z], 1)
            x_line = np.linspace(X.min(), X.max(), 100)
            plt.plot(x_line, intercept + slope * x_line, color=color, linewidth=3)
    
        # Overall regression line
        slope, intercept = np.polyfit(X, Y, 1)
        x_line = np.linspace(X.min(), X.max(), 100)
        plt.plot(x_line, intercept + slope * x_line, color="black", linewidth=3)
    
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    mo.output.append(mo.md("Horizontal is no correlation"))

    pipe_cont_example()
    return


@app.cell
def _(util):
    util.draw_causal_graph(
        [
            ("H0", "H1"),
            ("T", "H1"),
            ("T", "F"),
            ("F", "H1")
        ],
        node_props={
            "T": {"color": "red", "label": "T, treatment"},
            "F": {"label": "F, fungus"},
            "H0": {"label": "H0, height at t0"},
            "H1": {"label": "H1, height at t1"}
        },
        edge_props={
            ("T", "H1"): {"color": "red"},
            ("T", "F"): {"color": "red"},
        },
        graph_direction="LR"
    )
    # https://youtu.be/mBEA7PKDmiY?t=2350
    # https://github.com/dustinstansbury/statistical-rethinking-2023/blob/main/Lecture%2005%20-%20Elemental%20Confounds.ipynb
    # treatment could also affect height directly

    # If we stratify by F (control for it, block it), then we only get direct effect of treatment on H1 ; we want total causal effect
    return


@app.cell
def _(util):
    # Collider

    util.draw_causal_graph(
                edge_list=[
                    ("X", "Z"),
                    ("Y", "Z"),
                ],
                graph_direction="LR"
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    Y \perp\!\!\!\perp X
    \\
    Y \not\!\perp\!\!\!\perp X \mid Z
    $$

    # Collider

    X and Y are not correlated / independent
    But given Z, they are correlated, not independent
    """)
    return


@app.cell
def _(np, pd, rng):
    def collider():
        n= 1000
        X = rng.binomial(n=1, p=0.5, size=n)
        Y = rng.binomial(n=1, p=0.5, size=n)
        p = np.where(X + Y > 0, 0.9, 0.2)
        Z = rng.binomial(n=1, p=p, size=n)

        print(np.corrcoef(X, Y)[0, 1])
        print(pd.crosstab(
                X,
                Y,
                 rownames=["X"],
                 colnames=["Y"],
                # margins=True
            ))

        # at same levels of Z, there is a correlation
        print(pd.crosstab(
                X[Z == 0],
                Y[Z == 0],
                 rownames=["X"],
                 colnames=["Y"],
                # margins=True
            ))
    
        print( np.corrcoef(X[Z == 0], Y[Z == 0])[0, 1] )


        print(pd.crosstab(
            X[Z == 1],
            Y[Z == 1],
             rownames=["X"],
             colnames=["Y"],
            # margins=True
        ))

        print( np.corrcoef(X[Z == 1], Y[Z == 1])[0, 1] )
    collider()

    # the data can produce different samples
    return


@app.cell
def _(mo, np, plt):
    def collider_cont_example():
        N = 700
        rng = np.random.default_rng(42)

        X = rng.normal(loc=0, scale=1, size=N)    
        Y = rng.normal(loc=0, scale=1, size=N)    
        p = 1 / (1 + np.exp(-(2 * X + 2 * Y - 2)))
        Z = rng.binomial(n=1, p=p, size=N)
    

        # Scatter plot, controlling for Z
        plt.scatter(
            X[Z == 0], Y[Z == 0],
            facecolors="none", edgecolors="blue",
            linewidths=1.5, label="Z = 0"
        )

        plt.scatter(
            X[Z == 1], Y[Z == 1],
            color="red", label="Z = 1"
        )

        # Regression lines
        for z, color in [(0, "blue"), (1, "red")]:
            slope, intercept = np.polyfit(X[Z == z], Y[Z == z], 1)
            x_line = np.linspace(X.min(), X.max(), 100)
            plt.plot(x_line, intercept + slope * x_line, color=color, linewidth=3)

        # Overall regression line
        slope, intercept = np.polyfit(X, Y, 1)
        x_line = np.linspace(X.min(), X.max(), 100)
        plt.plot(x_line, intercept + slope * x_line, color="black", linewidth=3)

        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        mo.output.append(mo.md("Horizontal is no correlation"))

    # at same Z they are correlated
    collider_cont_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Say we have grants, and trustworthiness that are not correlated, but the selection makes them correlated
    """)
    return


@app.cell
def _(np, pd, plt, reload, smf, stats, util):
    reload(util)

    def _():
        np.random.seed(123)
        n_samples = 200

        # N and T are independent
        N = stats.norm.rvs(size=n_samples)
        T = stats.norm.rvs(size=n_samples)

        # Award criterion; either are large enough to threshold
        A = np.where(N + T > 2, 1, 0)

        for awarded in [0, 1]:
            color = 'gray' if not awarded else 'C0'
            N_A = N[A==awarded]
            T_A = T[A==awarded]
            util.plot_scatter(N_A, T_A, color=color)

        awarded_model = smf.ols("T ~ N", data=pd.DataFrame({"N": N_A, "T": T_A})).fit()
        util.plot_line(N_A, awarded_model.predict(), color='k', label='Post-selection\nTrend')
        plt.xlabel("Newsworthiness")
        plt.ylabel("Trustworthiness")
        plt.axis("square")
        return plt.legend();


    _()
    return


@app.cell
def _(util):
    util.draw_causal_graph(
                edge_list=[
                    ("N", "A"),
                    ("T", "A"),
                ],
                graph_direction="LR"
            )

    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The descendant

    $$
    Y \not\!\perp\!\!\!\perp X
    \\[2em]
    \overbrace{Y \perp\!\!\!\perp X \mid A}^{\text{if strong  enough}}
    $$

    X and Y are correlated through Z (not independant)

    A holds information about Z

    Once stratified by A, X and Y are less associated
    """)
    return


@app.cell
def _(np, pd, rng):
    def descendant_ex():
        n = 1000

        X = rng.binomial(n=1, p=0.5, size=n)
        Z = rng.binomial(n=1, p=(1 - X) * 0.1 + X * 0.9)    
        Y = rng.binomial(n=1, p=(1 - Z) * 0.1 + Z * 0.9)
        A = rng.binomial(n=1, p=(1 - Z) * 0.1 + Z * 0.9)

        print(pd.crosstab(
            X,
            Y,
             rownames=["X"],
             colnames=["Y"],
            # margins=True
        ))


        # correlation is reduced 
    
        print(np.corrcoef(X, Y)[0, 1])

        print(pd.crosstab(
                X[A == 0],
                Y[A == 0],
                 rownames=["X"],
                 colnames=["Y"],
                # margins=True
        ))

        print( np.corrcoef(X[A == 0], Y[A == 0])[0, 1] )


        print(pd.crosstab(
            X[A == 1],
            Y[A == 1],
             rownames=["X"],
             colnames=["Y"],
            # margins=True
        ))

        print( np.corrcoef(X[A == 1], Y[A == 1])[0, 1] )


    descendant_ex()
    return


if __name__ == "__main__":
    app.run()
