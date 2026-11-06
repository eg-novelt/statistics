import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


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
    import seaborn as sns

    reload(util)
    return Path, az, mo, np, pd, plt, pm, smf, sns, stats, util


@app.cell
def _(util):
    util.draw_causal_graph(
        edge_list=[
            ("U", "Z"),
            ("Z", "X"),        
            ("X", "Y"),
            ("U", "Y")
        ],
        node_props={
            "Z": {"color": "red"},
        },
        edge_props={
            #("U", "Z"): {"color": "red"},
            #("Z", "X"): {"color": "red"}
        },
        graph_direction="LR"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Find a set of control variables that close/block all backdoor paths.

    Block the pipe $X \perp\!\!\!\perp U \mid Z$

    $$
    p(Y | do(X)) = \sum_z p(Y \mid X, Z)P(Z = z)
    \\[2em]
    Y_i \sim \text{Normal}(\mu_i, \sigma)
    \\
    \mu_i=\alpha+\beta_XX_i + \beta_ZZ_i
    $$

    $$
    \\[2em]
    \begin{aligned}
    U &\sim \text{Bernoulli}(0.5)
    \\
    Z &\sim \text{Normal}(\beta_{UZ}U, 1)
    \\
    X &\sim \text{Normal}(\beta_{ZX}Z, 1)
    \\
    Y &\sim \text{Normal}(\alpha + \beta_{XY}X + \beta_{UY}U, 1)
    \\
    \end{aligned}
    $$

    Note the ols command is close to this $Y = \alpha + \text{coef}X + \text{coef}Z + \epsilon$

    Unstratified / confounded

    Note exponential is a > 0 value with mean 1

    and we are using priors too
    $$
    \\[2em]
    \begin{aligned}
    \alpha &\sim \text{Normal}(0, 1)
    \\
    \beta_{XY} &\sim \text{Normal}(0, 1)
    \\
    \sigma_Y &\sim {\text{Exponential}(1)}
    \\
    \mu &= \alpha + \beta_{XY}X
    \\
    Y &\sim \text{Normal}(\mu, \sigma_Y)
    \\
    \end{aligned}
    $$

    Stratified / confounded

    $$
    \begin{aligned}
     Y &\sim \text{Normal}(\mu_Y, \sigma_Y)
     \\
    \mu_Y &= \alpha + \beta_{XY}X + \beta_{Z}Z
    \\
    \alpha &\sim \text{Normal}(0, 1)
    \\
    \beta_{XY} &\sim \text{Normal}(0, 1)
    \\
    \beta_{Z} &\sim \text{Normal}(0, 1)
    \\
    \sigma_Y &\sim \text{Exponential}(1)
    \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(np, pd, smf, stats):
    # Simulate
    # https://youtu.be/uanZZLlzKHw?t=1367
    # some ref python https://github.com/dustinstansbury/statistical-rethinking-2023/blob/main/Lecture%2006%20-%20Good%20%26%20Bad%20Controls.ipynb
    # slides https://github.com/rmcelreath/stat_rethinking_2024
    # https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-06
    def simulate_confounded_y():
        #np.random.seed(42)
        # note the beta_XY can vary quite a bit, using sample 42 yields 0.11 as XY
        np.random.seed(42)
        n_samples = 200
    
        alpha = 0
        # simulating no causal effect to make it easier to see
        beta_XY = 0
        beta_UY = -1
        beta_UZ = -1
        beta_ZX = 1

        # scale is standard deviation
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        U = stats.bernoulli.rvs(p=.5, size=n_samples)
        Z = stats.norm.rvs(loc=beta_UZ * U) # , scale=1
        X = stats.norm.rvs(loc=beta_ZX * Z)
        Y = stats.norm.rvs(loc=alpha + beta_XY * X + beta_UY * U) # , scale=1

        return X, Y, Z

    X1, Y1, Z1 = simulate_confounded_y()
    X2, Y2, Z2 = simulate_confounded_y()

    print(np.unique(X1 - X2))
    print(np.unique(Y1 - Y2))
    print(np.unique(Z1 - Z2))


    df = pd.DataFrame({
        "X": X1,
        "Y": Y1,
        "Z": Z1
    })

    # to show the slope of X on Y 
    result = smf.ols("Y ~ X + Z", data=df).fit()

    real_XY = result.params["X"]
    
    result.summary()

    return real_XY, simulate_confounded_y


@app.cell
def _(Path, az, pm, simulate_confounded_y, util):
    # Unstratified model, ignoring Z and U
    def train_unstrat_model():
        with pm.Model() as unstratified_model:
            # Priors
            alpha = pm.Normal("alpha", 0, 1)
            beta_XY = pm.Normal("beta_XY", 0, 1)
            sigma = pm.Exponential("sigma", 1)

            X, Yobs, _ = simulate_confounded_y()
        
            # Likelihood
            mu = alpha + beta_XY * X
            Y = pm.Normal("Y", mu=mu, sigma=sigma, observed=Yobs)
            unstratified_model_inference = util.sample_or_load(
                    lambda:pm.sample(), Path("chp6_1.trace"))

            return unstratified_model_inference

    def train_strat_model():
        with pm.Model() as unstratified_model:
            # Priors
            alpha = pm.Normal("alpha", 0, 1)
            beta_XY = pm.Normal("beta_XY", 0, 1)
            beta_Z = pm.Normal("beta_Z", 0, 1)
            sigma = pm.Exponential("sigma", 1)

            X, Yobs, Z = simulate_confounded_y()
        
            # Likelihood
            mu = alpha + beta_XY * X + beta_Z * Z 
            Y = pm.Normal("Y", mu=mu, sigma=sigma, observed=Yobs)
            stratified_model_inference = util.sample_or_load(
                    lambda:pm.sample(), Path("chp6_2.trace"))

            return stratified_model_inference

    unstratified_model_inference = train_unstrat_model()     
    stratified_model_inference = train_strat_model()     

    az.summary(unstratified_model_inference, ci_prob=0.94)

    return stratified_model_inference, unstratified_model_inference


@app.cell
def _(az, stratified_model_inference):
    az.summary(stratified_model_inference, ci_prob=0.94)

    # note this beta_Z is meaningless, not Z on Y ; only influences Y through X
    return


@app.cell
def _(az, unstratified_model_inference):
    az.hdi(data=unstratified_model_inference, prob=.94)
    return


@app.cell
def _(
    plt,
    real_XY,
    sns,
    stratified_model_inference,
    unstratified_model_inference,
):
    _, ax = plt.subplots(figsize=(8, 3))

    sns.kdeplot(
        x=unstratified_model_inference.posterior.beta_XY.values.flatten(),
        color="k",
        label="Y|X",
        ax=ax,
    )

    sns.kdeplot(
        x=stratified_model_inference.posterior.beta_XY.values.flatten(),
        color="C0",
        label="Y|X,Z",
        ax=ax,
    )


    ax.axvline(
        real_XY,
        color="C0",
        linestyle="--",
        alpha=.5,
        label=f"Actual={real_XY:.2f}",
    )

    ax.set(
        xlim=(-.3, .3),
        xlabel="posterior beta_XY",
        ylabel="density",
    )

    ax.legend()
    return


if __name__ == "__main__":
    app.run()
