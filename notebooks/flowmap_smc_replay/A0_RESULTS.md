# Step A0 — is the increment estimator noise, or real $\Delta V$? (2026-09-10, no GPU)

**Question.** Resampling consumes $U_n = \hat V_n - \hat V_{n-1}$. With $\hat V_n = V_n + \varepsilon_n$,

$$\operatorname{Var}(U_n) = \operatorname{Var}(\Delta V_{\text{true}}) + \sigma_{\varepsilon,n}^{2} + \sigma_{\varepsilon,n-1}^{2}$$

CRN acts **only** on the $\sigma_\varepsilon$ terms. So: what fraction of $\operatorname{Var}(U_n)$ are they?

**Answer: 0.28–0.48 at $K = 4$ — below half on every cell measured. $\Delta V_{\text{true}}$ dominates.**
Per the plan's decision rule, **CRN is not weak here, it is inapplicable**, and this line stops.

Reproduce with `python notebooks/flowmap_smc_replay/a0_increment_decomposition.py --toy-lam 4 8 16 32`
(full output in `a0_output.txt`).

## Method

$\sigma_\varepsilon^2$ is **measured, not modelled**: split each step's recorded $(M, K)$ `r_k` into
disjoint column subsets of size $K'$ and take the variance of $\hat V$ across subsets. Two details
that changed the answer:

* **Subsets must respect the antithetic pairing.** `_antithetic_noise` lays columns out as
  $[\varepsilon_0, -\varepsilon_0, \varepsilon_1, -\varepsilon_1, \dots]$; a subset that splits a pair
  is an estimator no real run uses, and mixing split and intact subsets inflates the spread.
  Permuting *pair blocks* makes a $K'=4$ subset exactly the four draws a real $K=4$ run takes.
* **$\sigma_\varepsilon^2$ does not fall like $1/K$** (see below), so carrying a measurement from
  $K'$ to $K$ uses the **fitted** exponent. Assuming $p = 1$ understates $\sigma_\varepsilon(K)$ and
  biases the noise share *down* — the first version of this script did that and still said STOP.

**Validation.** The pipeline reproduces the established $\operatorname{sd}(U)/\operatorname{sd}(V)$
exactly — 0.671 / 0.670 / 0.346 for $K = 16/32/8$ against `RESULTS.md`'s 0.671 / 0.670 / 0.346. The
additive model was checked within-cell (rebuild $\hat V$ from a real 4-column subset, compare
$\operatorname{Var}(U)$ against $\operatorname{Var}(\Delta V) + 2\sigma^2(4)$): **0.98–1.30× on every
$K=8$ cell and the toy**. It degrades to 0.60× on the $K=16$ cell, which extrapolates furthest.

## Result

| cell | $K$ | $a = \lambda\,\mathrm{sd}_k$ | $p$ | Var$(U)$ | Var$(\Delta V)$ | **share@4** | sd(U)/sd(V) | CRN floor |
|---|---|---|---|---|---|---|---|---|
| k16 | 16 | 0.651 | 0.64 | 0.953 | 0.786 | **0.341** | 0.671 | 0.587 |
| k32 | 32 | 0.797 | 0.45 | 1.007 | 0.800 | **0.400** | 0.670 | 0.587 |
| k08b | 8 | 0.571 | 0.77 | 0.374 | 0.272 | **0.391** | 0.346 | 0.290 |
| k08max | 8 | 0.563 | 0.76 | 0.361 | 0.255 | **0.412** | 0.347 | 0.287 |
| s102max | 8 | 0.526 | 0.82 | 0.360 | 0.296 | **0.275** | 0.617 | 0.541 |
| s102soft | 8 | 0.518 | 0.60 | 0.310 | 0.230 | **0.349** | 0.454 | 0.392 |
| s103max | 8 | 0.565 | 0.36 | 0.452 | 0.276 | **0.454** | 0.432 | 0.322 |
| s103soft | 8 | 0.591 | 0.32 | 0.578 | 0.390 | **0.374** | 0.477 | 0.377 |
| toy $\lambda=4$ | 256 | 0.352 | 0.79 | 0.121 | 0.119 | **0.256** | 0.326 | 0.317 |
| toy $\lambda=8$ | 256 | 0.911 | 0.50 | 1.026 | 0.973 | **0.317** | 0.355 | 0.336 |
| toy $\lambda=16$ | 256 | 1.925 | 0.27 | 5.098 | 4.173 | **0.413** | 0.590 | 0.501 |
| toy $\lambda=32$ | 256 | 3.878 | 0.17 | 21.56 | 14.89 | **0.489** | 0.836 | 0.628 |

"CRN floor" is $\operatorname{sd}(U)/\operatorname{sd}(V)$ under **perfect** CRN
($\rho_\varepsilon = 1$), i.e. the best case that does not exist. At the planned $K = 4$ config, arm A
sits at 0.534 and perfect CRN reaches $\approx 0.42$ — a **1.2–1.4× reduction in $\operatorname{sd}(U)$**,
against a plan that needed the increment to stop being noise-dominated.

**This is robust to the one thing that could have rescued it.** Fix 1 makes $\hat V$ *noisier*
($\lambda\sigma_{\text{within}}$ rising to 1.0–3.0 by the plan's own estimate), which would raise the
noise share. The toy sweep covers exactly that: over an 11× range of $a$, 0.35 → 3.88, the share rises
only 0.256 → **0.489**, still short of half at the top of the range. A noisier lookahead does not
change the verdict.

**Antithetic sampling has already claimed part of the prize.** Turning the pairing off at $\lambda=32$
moves the share 0.489 → 0.529 — i.e. some of the error CRN would cancel is *already* being cancelled
within each step, for free, by a flag that has been on all along.

## Two findings worth keeping

**1. $\sigma_\varepsilon^2 \sim k^{-p}$ with $p = 0.17$–$0.82$, never 1 — and this explains the $K$ saturation.**
`RESULTS.md` recorded that $\operatorname{sd}(U)/\operatorname{sd}(V)$ refuses to follow $1/\sqrt K$
(measured drop $\approx 1.5×$ where the prediction reaches 5.66×) and left it unexplained. The cause is
here: the estimator error itself decays far slower than $1/k$. **It is not the antithetic pairing** —
turning it off gives $p = 0.42$ vs 0.17 at the same $\lambda$, still nowhere near 1. It is the
`logsumexp`: $p$ falls monotonically as $a$ rises (0.79 → 0.50 → 0.27 → 0.17 for $a = 0.35 \to 3.88$),
which is the signature of a max-dominated statistic, whose error shrinks like $1/\log k$ rather than
$1/k$. **Raising $K$ was never going to work, and now there is a mechanism for why.**

**2. The level predicts the endpoint; the increment barely does — re-measured on 9 cells.**
Mean over steps, lineages traced back through the parent maps:

| cell | k16 | k32 | k04 | k08b | k08max | s102max | s102soft | s103max | s103soft |
|---|---|---|---|---|---|---|---|---|---|
| $\operatorname{corr}(\hat V_n, f_{\text{final}})$ | −0.17 | 0.20 | 0.43 | **0.83** | **0.94** | **0.70** | 0.38 | 0.15 | **0.82** |
| $\operatorname{corr}(U_n, f_{\text{final}})$ | 0.12 | 0.16 | 0.14 | 0.11 | 0.11 | −0.03 | 0.07 | 0.23 | 0.29 |

This confirms the plan's table on far more data (it had four steps of one run). **But read together
with the share above it says something the plan did not anticipate:** the increment is ~61% *real*
$\Delta V$ movement and still barely predicts the endpoint. Removing every last bit of estimator noise
would leave resampling selecting on a genuine quantity that is close to uninformative about $t = 1$.
Caveat: some of this is arithmetic — consecutive levels are highly correlated, so their difference is
small and its correlation with anything is attenuated. It is not, on its own, proof of a broken twist.

## Verdict

**STOP the CRN line**, per decision (b) of 2026-09-10. CRN's ceiling is a 1.2–1.4× reduction in
$\operatorname{sd}(U)$ on a quantity that is majority real movement, and the one mechanism that could
have raised that ceiling (a noisier replay lookahead) was tested on the toy and does not.

**Fix 1 is not settled by the above** — A0 measured the *size* of CRN's target term, which was never
Fix 1's argument. It is taken up separately in the next two sections, which between them close two of
its three claims.

---

# Fix 1, part 1 — the "optimism drain" is Jensen, not an artifact

The plan's diagnosis (1) calls $\hat V/\lambda - \bar f_{\text{cand}}$ "a $\lambda$-scaled optimism …
an estimator artifact that the increments then select on", and claims Fix 1 removes it structurally.
Measured against the convexity term $\lambda\operatorname{Var}_k(r)/2$ on the recorded `r_k`:

| cell | ratio drain / $\lambda\operatorname{Var}_k/2$, all 15 steps |
|---|---|
| k08b | 0.79 – 0.96 |
| k32 | 0.84 – 1.10 |

**It is the second-order term of $\log\mathbb{E}[e^{\lambda f}] \approx \lambda\mathbb{E}[f] + \tfrac{\lambda^{2}}{2}\operatorname{Var}[f]$**,
to within ~15% at every step. So:

* It is **not an error**. $V_t/\lambda$ genuinely exceeds $\mathbb{E}[f \mid x_t]$ by
  $\lambda\operatorname{Var}_t/2$ — that convexity is what makes $V$ a soft-max over futures rather
  than a mean. It "drains" because the spread of futures collapses as $t \to 1$.
* **Fix 1 cannot remove it, and would enlarge it.** An exact-posterior lookahead has the same term on
  the *true* posterior spread, which the plan itself puts at $\lambda\sigma_{\text{within}} = 1.0$–$3.0$
  against the measured $a \approx 0.57$.

# Fix 1, part 2 — does the proposal misrank particles? (toy screen)

`a0b_lookahead_bias_gate.py`. The surviving Fix 1 argument is that the renoise lookahead estimates the
value of a hand-made proposal rather than $V_t$. The only thing an intermediate potential does is
**order particles for resampling**, so the question is whether the two orderings differ. Measured as
$\operatorname{spearman}(\hat V_{\text{renoise}}, \hat V_{\text{replay}})$ with **both at $K = 2048$**,
so estimator noise is out of it and what is left is pure proposal bias. Same untilted cloud, no
resampling, $d = 2$.

| $\lambda$ | rank bias, $t \le 0.25$ | rank bias, $t > 0.25$ | endpoint relevance $t \le 0.25$: $V_{\text{true}}$ vs $V_{\text{renoise}}$ |
|---|---|---|---|
| 4 | 0.951 | 0.998 | 0.317 vs **0.343** |
| 8 | **0.820** | 0.997 | 0.265 vs **0.336** |
| 16 | **0.700** | 0.990 | 0.239 vs **0.328** |

**The bias is real, and it is concentrated exactly where the runs collapse.** Past $t = 0.25$ the two
orderings are identical (0.99+) — the posterior has collapsed and every estimator agrees. Before it,
they diverge, and the divergence grows with $\lambda$: at $\lambda = 16$ the proposal ranks particles
at rank-correlation **0.70** with the truth. FLUX's runs collapse at **step 3** ($t = 0.1875$), inside
that window.

**But there is no evidence the corrected ordering is better.** On the one relevance metric available —
correlation with the realized $f$ at $t = 1$ — the *biased* proposal wins at every $\lambda$
(0.343 / 0.336 / 0.328 vs 0.317 / 0.265 / 0.239). This is not paradoxical: at early $t$ the true value
function is an expectation over a still-mostly-random future, so it *correctly* carries little
information about any realized endpoint, while the renoise proxy is a deterministic contraction of the
current $x$ and partly measures where the particle already is.

**$d = 2$ caveat, stated in advance and unchanged by the result:** the bias magnitude is a lower bound
— in $d = 65536$ a renoised Gaussian blob and the true posterior diverge far more. The endpoint-relevance
comparison is likewise $d$-dependent and this screen cannot settle its sign.

# Fix 1, part 3 — it doubles the Jensen exposure, and $K$ cannot pay for it

Two Jensen effects act on the potential, both governed by the candidate spread $a = \lambda\sigma_k$:

* **(A) convexity of the target**, $V = \log\mathbb{E}[e^{\lambda f}] > \lambda\mathbb{E}[f]$, size
  $\approx \lambda^{2}\sigma_k^{2}/2$. Not an error — it is the twist (this is the "drain" above).
* **(B) MC bias of the $K$-sample estimator**, $\mathbb{E}[\log\frac1K\sum e^{\lambda r}] < V$, size
  $\approx (e^{a^{2}} - 1)/2K$. **Exponential in $a^{2}$.**

**Fix 1 roughly doubles $a$ where it matters.** Toy, replay vs renoise candidate spread:

| $t$ | 0.0625 | 0.125 | 0.1875 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|---|---|
| $\sigma_k$ replay / renoise | 2.40× | 2.47× | 2.25× | 2.04× | 1.38× | 0.93× |

**And there is an exact identity for the limit: under replay, $a(t \to 0) = \lambda\operatorname{std}_p(f) = m$.**
At $t \approx 0$ the posterior *is* $p$, so the lookahead candidates are prior draws and their spread is
the prior spread. At $m = 1.25$ that is $a \approx 1.25$, matching the toy's 2.0–2.5× applied to the
measured $a_{\text{renoise}} = 0.57$.

At $K = 4$ term (B) then goes from $(e^{0.32}-1)/8 = 0.05$ to $(e^{1.8}-1)/8 = 0.66$ log-weight units —
**~14×**, against a per-step $\operatorname{sd}(U) \approx 0.6$. The bias reaches the size of the whole
increment.

**The remedy is the one already ruled out.** $K$ is how you control (B), and A0 measured that $K$ gets
*less* effective as $a$ grows: the decay exponent $p$ in $\sigma_\varepsilon^{2} \sim k^{-p}$ runs
0.79 / 0.50 / 0.27 / **0.17** at $a = 0.35 / 0.91 / 1.93 / 3.88$. Fix 1 moves us toward the regime where
raising $K$ stops working. It also **scales the wrong way with the tilt**: $a(t\to0) = m$ exactly, so at
the $m = 3$ end this project wants, `logsumexp` over $K = 4$ is a hard $\max_k$ and the potential is no
longer estimating a value function.

**Reframing $\eta$.** It is not only an arbitrary knob — it is a **Jensen-exposure control**. Small
$\eta$ gives narrow candidates, so `logmeanexp` $\approx$ mean and both (A) and (B) stay small. Fix 1
removes the knob and accepts whatever spread $p$ gives. Its cleanest stated virtue is also the loss of
the only lever keeping the estimator in its well-behaved regime.

**If Fix 1 is ever built, this is its pre-registered gate:** measure $a_{\text{replay}}$ per step and
require $(e^{a^{2}}-1)/2K \ll \operatorname{sd}(U)$. Expect failure at $K = 4$, with no $K$ that fixes it.

**Neither `deferred resampling` nor larger $M$ changes any of this** — they do not touch the lookahead,
so their Jensen exposure is exactly today's.

---

**Net: Fix 1 has a measurable effect of unknown sign.** One claim is false (the drain), one is moot
(enabling CRN), one is true but inert ($L$/$S$ are already unused), and the last — that the ordering is
wrong — is confirmed as a *difference* with no demonstrated benefit. It should not be built ahead of
$M$; if it is ever built, the specific prediction to test is a reordering confined to $t \lesssim 0.25$.

---

# A1 — the aggregation, not the lookahead, is what broke $K$

`a1_mean_aggregation.py`, offline on the same recorded `r_k`. Three aggregations of the *same*
candidates: mean ($\tau\to0$), soft ($\tau=1$, today), max ($\tau\to\infty$, tested and null).

| metric | mean | soft | max |
|---|---|---|---|
| exponent $p$ in $\sigma_\varepsilon^2 \sim k^{-p}$ | **0.90 – 1.21** | 0.32 – 0.82 | −0.03 – 0.29 |
| $\operatorname{sd}(U)/\operatorname{sd}(V)$ | **lowest in every cell** | middle | highest |
| Spearman vs soft | 0.92 – 0.98 | — | 0.93 – 0.98 |

**The mean is a textbook $1/k$ estimator; the soft value is not.** That is the mechanism behind the
$K$ saturation `RESULTS.md` recorded and could not explain — and it means **"raising $K$ buys nothing"
is a property of the aggregation, not of the lookahead**. Under the mean, a $K$ sweep should show the
$1/\sqrt K$ that never appeared.

**The deleted term points the wrong way.** $V_{\text{soft}}/\lambda = \mu_k + \lambda\sigma_k^2/2$:

| | corr($\mu_k$, $f_{t=1}$) | corr($\lambda\sigma_k^2/2$, $f_{t=1}$) | convexity share of sd($V$) |
|---|---|---|---|
| range over 11 cells | −0.16 … **+0.94** | **negative in 7 of 11**, to −0.59 | 8 – 30% |

The soft value spends 8–30% of its spread backing particles whose futures are merely *uncertain*, and
those finish **lower**. Under the target that term belongs there — but at $K = 4$ it is estimated
terribly, which is exactly what a small $p$ means.

**Temper the expectation.** Spearman(mean, soft) = 0.92–0.98 is the same "barely reorders" signature
that preceded the max's null. A large move in terminal $E_q[f]$ at $m = 1.25$ would be a surprise.
The mean's real case is (i) $K$ works again and (ii) **no Jensen exposure at all**, so unlike the soft
value it does not degrade as $m$ rises — and $a(t\to0) = m$ means the soft value *is* a hard max at
$m = 3$, $K = 4$.

---

**Next, per the plan's fallback:** $M$ (16, 32) at $K = 4$, and the resampling schedule — `uniq/M` is
monotone non-increasing because systematic resampling only destroys lineages and nothing regenerates
them. Finding 2 above is an independent argument for the same direction: if the increment carries
little endpoint information at any noise level, the lever is how many lineages survive selecting on
it, not how cleanly it is estimated.
