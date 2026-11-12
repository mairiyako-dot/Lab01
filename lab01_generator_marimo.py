import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Lab01 · датчики для модели ионного канала")


@app.cell(hide_code=True)
def _():
    import marimo as mo, numpy as np, math, matplotlib.pyplot as plt
    from scipy import stats
    COLORS = {"open": "#1f77b4", "closed": "#2ca02c", "current": "#ff7f0e",
              "theory": "#d62728", "gen": "#9467bd", "real": "#7f7f7f"}
    plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.titleweight": "bold",
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    return COLORS, math, mo, np, plt, stats


@app.cell(hide_code=True)
def _(math, np):
    class MT19937:
        """Вихрь Мерсенна, сверен с эталоном авторов (зерно 5489)."""
        N, M = 624, 397
        def __init__(self, seed=5489):
            self.mt = [0] * self.N; self.mt[0] = seed & 0xffffffff
            for i in range(1, self.N):
                self.mt[i] = (1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & 0xffffffff
            self.idx = self.N; self.calls = 0
        def _twist(self):
            for i in range(self.N):
                y = (self.mt[i] & 0x80000000) | (self.mt[(i+1) % self.N] & 0x7fffffff)
                v = self.mt[(i+self.M) % self.N] ^ (y >> 1)
                self.mt[i] = v ^ 0x9908b0df if y & 1 else v
            self.idx = 0
        def u32(self):
            if self.idx >= self.N: self._twist()
            y = self.mt[self.idx]; self.idx += 1
            y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >> 18
            return y
        def random(self):
            self.calls += 1
            a, b = self.u32() >> 5, self.u32() >> 6
            return (a*67108864.0 + b + 0.5) / 9007199254740992.0

    _r = MT19937(5489)
    assert [_r.u32() for _ in range(3)] == [3499211612, 581869302, 3890346734]

    def exp_inverse(rng, lam, n):
        return np.array([-math.log(rng.random()) / lam for _ in range(n)])

    def _one_neumann(rng):
        k = 0
        while True:
            u = rng.random(); prev = u; run = 1
            while True:
                v = rng.random()
                if v < prev: prev = v; run += 1
                else: break
            if run % 2 == 1: return k + u
            k += 1

    def exp_neumann(rng, lam, n):
        return np.array([_one_neumann(rng) for _ in range(n)]) / lam

    def normal_box_muller(rng, mu, sigma, n):
        out = np.empty(n)
        for i in range(0, n, 2):
            r = math.sqrt(-2 * math.log(rng.random())); t = 2 * math.pi * rng.random()
            out[i] = r * math.cos(t)
            if i + 1 < n: out[i + 1] = r * math.sin(t)
        return mu + sigma * out

    def normal_polar(rng, mu, sigma, n):
        out = np.empty(n); i = 0
        while i < n:
            v1, v2 = 2 * rng.random() - 1, 2 * rng.random() - 1
            s = v1 * v1 + v2 * v2
            if s >= 1 or s == 0: continue
            f = math.sqrt(-2 * math.log(s) / s)
            out[i] = v1 * f; i += 1
            if i < n: out[i] = v2 * f; i += 1
        return mu + sigma * out
    return MT19937, exp_inverse, exp_neumann, normal_box_muller, normal_polar


@app.cell(hide_code=True)
def _(mo):
    dist = mo.ui.radio(["Exp(λ) · T_open", "N(μ, σ²) · I_open"], value="Exp(λ) · T_open", label="распределение")
    method = mo.ui.radio(["обратная функция / Бокс — Мюллер", "Нейман / полярный"],
                         value="обратная функция / Бокс — Мюллер", label="метод")
    lam = mo.ui.slider(10, 300, step=0.1, value=97.4, label="λ, 1/с", show_value=True)
    mu = mo.ui.slider(10, 25, step=0.01, value=17.57, label="μ, пА", show_value=True)
    sigma = mo.ui.slider(0.2, 5, step=0.01, value=1.73, label="σ, пА", show_value=True)
    n = mo.ui.slider(200, 30000, step=200, value=5000, label="n", show_value=True)
    seed = mo.ui.number(1, 10**6, value=2026, label="зерно генератора")
    mo.vstack([
        mo.md("# Lab01 · интерактивные датчики для модели ионного канала\n\n"
              "Exp(λ) — время в открытом состоянии $T_{open}$ (по данным λ̂ = 97,4 1/с). "
              "N(μ, σ²) — ток в открытом состоянии $I_{open}$ (по данным μ̂ = 17,57 пА, σ̂ = 1,73 пА). "
              "Все методы работают на собственном вихре Мерсенна."),
        mo.hstack([dist, method], justify="start", gap=2),
        mo.hstack([lam, n, seed], justify="start", gap=2),
        mo.hstack([mu, sigma], justify="start", gap=2),
    ])
    return dist, lam, method, mu, n, seed, sigma


@app.cell(hide_code=True)
def _(COLORS, MT19937, dist, exp_inverse, exp_neumann, lam, method, mo, mu, n, normal_box_muller,
      normal_polar, np, plt, seed, sigma, stats):
    rng = MT19937(int(seed.value or 1))
    first = method.value.startswith("обратная")
    if dist.value.startswith("Exp"):
        x = (exp_inverse if first else exp_neumann)(rng, lam.value, n.value)
        t = np.linspace(0, np.quantile(x, .995), 300)
        pdf, law, unit = lam.value * np.exp(-lam.value * t), stats.expon(0, 1 / lam.value), "с"
        cost = 1.0 if first else np.e ** 2 / (np.e - 1)
        name = "обратная функция" if first else "Нейман (без ln)"
    else:
        x = (normal_box_muller if first else normal_polar)(rng, mu.value, sigma.value, n.value)
        t = np.linspace(x.min(), x.max(), 300)
        pdf, law, unit = stats.norm.pdf(t, mu.value, sigma.value), stats.norm(mu.value, sigma.value), "пА"
        cost = 1.0 if first else 4 / np.pi
        name = "Бокс — Мюллер" if first else "полярный Марсальи"
    ks = stats.kstest(x, law.cdf)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].hist(x, bins=60, range=(t.min(), t.max()), density=True, color=COLORS["gen"], alpha=.6, edgecolor="white", label="датчик")
    ax[0].plot(t, pdf, color=COLORS["theory"], label="теория")
    ax[0].set(title=f"{name}: плотность", xlabel=f"значение, {unit}", ylabel=f"плотность, 1/{unit}"); ax[0].legend()
    q = np.linspace(.01, .99, 99)
    ax[1].scatter(law.ppf(q), np.quantile(x, q), s=10, color=COLORS["gen"])
    lim = [law.ppf(.01), law.ppf(.99)]; ax[1].plot(lim, lim, color=COLORS["theory"], ls="--")
    ax[1].set(title="Q–Q к теории", xlabel=f"теоретические квантили, {unit}", ylabel=f"квантили датчика, {unit}")
    fig.tight_layout()

    rows = [("среднее", x.mean(), law.mean()), ("σ", x.std(ddof=1), law.std()), ("медиана", np.median(x), law.median()),
            ("обращений к U на число", rng.calls / len(x), cost)]
    table = "| метрика | датчик | теория |\n|---|---|---|\n" + "\n".join(f"| {a} | {b:.5g} | {c:.5g} |" for a, b, c in rows)
    verdict = "не отвергается" if ks.pvalue > .05 else "отвергается"
    mo.vstack([fig, mo.md(f"{table}\n\n**Критерий КС:** D = {ks.statistic:.4f}, p = {ks.pvalue:.3f} → H₀ {verdict}")])
    return


if __name__ == "__main__":
    app.run()
