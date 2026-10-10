# Subsection draft: "Gradient-Guided Samplers"

Goes directly after the expectation-collapse proof subsection. LaTeX source below — copy and paste as-is.
Citation keys are placeholders of the form `CITE:<topic>`; swap them for your `.bib` keys.

```latex
\subsection{Gradient-Guided Samplers}
\label{sec:gradient-guidance-methods}

With the reward gradient now tractable, the remaining question is where to inject it into the
generative process. We instantiate three samplers, in increasing order of structural support for the
base model: a single guided pass of the model's own ODE, the same pass interleaved with Langevin
corrections, and the same pass driven by a classifier-free-guidance-sharpened velocity field. All
three share one reward, one reference set and one noise schedule, so they differ only in the
mechanism under study.

\paragraph{Setup and notation.}
We use a pretrained latent flow-matching model \citep{CITE:flow-matching, CITE:rectified-flow};
our backend is FLUX.1-dev \citep{CITE:flux}. Let $x_{t}$ denote the noisy latent at time
$t \in [1, 0]$, with $t = 1$ pure noise and $t = 0$ clean data, and let $v_{\theta}(x_{t}, t, c)$ be
the learned velocity field conditioned on the text embedding $c = E(p)$ of prompt $p$. Sampling
integrates the probability-flow ODE $dx_{t}/dt = v_{\theta}(x_{t}, t, c)$, and at any $t$ the clean
latent is estimated by the one-step projection
\begin{equation}
  \hat{x}_{0}^{(t)} = x_{t} - t \cdot v_{\theta}(x_{t}, t, c).
  \label{eq:xhat0}
\end{equation}
The reward $f$ compares score fields of \emph{clean} latents and is therefore meaningless on a noisy
$x_{t}$; we evaluate it on $\hat{x}_{0}^{(t)}$ instead and transport the gradient back under the
standard guidance approximation $\nabla_{x_{t}} f \approx \nabla_{\hat{x}_{0}} f$, i.e.\ treating
$\partial \hat{x}_{0} / \partial x_{t} = I$ \citep{CITE:universal-guidance}. The true Jacobian is
$I - t \cdot \partial v_{\theta} / \partial x_{t}$, a full $d \times d$ operator; our implementation
also supports differentiating through it exactly, which the collapse of
Section~\ref{sec:expectation-collapse} is what makes affordable, and we report both modes.

In every sampler below the \emph{magnitude} of $\nabla f$ is discarded and replaced by an intrinsic
reference scale, since the normalized IEM reward carries no units commensurable with a velocity or a
score:
\begin{equation}
  \tilde{g}(x_{t}) = \frac{\nabla_{\hat{x}_{0}} f}{\lVert \nabla_{\hat{x}_{0}} f \rVert_{2} + \epsilon}
  \cdot \lVert u(x_{t}, t) \rVert_{2},
  \qquad u \in \{v_{\theta}, s_{\theta}\}.
  \label{eq:normalized-grad}
\end{equation}
The guidance strength $\lambda$ is then a dimensionless mixing weight: $\lambda = 1$ places the
update at $45^{\circ}$ between "follow the model" and "ascend the reward", independent of the
reward's arbitrary gradient scale.

\paragraph{(1) Flow guidance.}
The first and simplest sampler nudges the model's own ODE at every step of its native schedule. One
Euler step from $t$ to $t'$, with $\Delta t = t - t'$, becomes
\begin{equation}
  x_{t'} = x_{t} - \Delta t \cdot \bigl( v_{\theta}(x_{t}, t, c) - \lambda \tilde{g}(x_{t}) \bigr),
  \qquad u = v_{\theta},
  \label{eq:flow-guidance}
\end{equation}
so the reward enters as a rotation of the transport field rather than as a separate corrective
displacement. This is a single deterministic forward pass: no tempering ladder, no particle
ensemble, no resampling, and cost dominated entirely by the one reward backward per step. It
produces genuinely restyled yet recognizable samples over an intermediate band of $\lambda$, above
which the guidance term overpowers the velocity field and the trajectory locks onto a fixed
off-manifold attractor that no longer depends on $\lambda$. The mechanism of that failure is that a
single pass has no way to \emph{re-equilibrate} onto the model's marginal $p_{t}$ after a nudge, so
each step's off-manifold error compounds; the remaining two samplers each add a different form of
structural support.

\paragraph{(2) Predictor--corrector with a Langevin corrector.}
Here each guided Euler step~\eqref{eq:flow-guidance} is treated as a predictor, and at the arrival
node $t$ we run $M$ steps of unadjusted Langevin dynamics at that fixed noise level, targeting the
tilted intermediate marginal
$q_{t}(x) \propto p_{t}(x) \exp\bigl( \lambda f(\hat{x}_{0}(x_{t})) \bigr)$. Its drift needs
$\nabla \log p_{t}$, which the velocity field supplies exactly by reparametrization,
\begin{equation}
  s_{\theta}(x_{t}, t) = -\frac{x_{t} + (1 - t) v_{\theta}(x_{t}, t, c)}{t},
  \label{eq:score-from-velocity}
\end{equation}
giving the corrector update, for $j = 0, \dots, M-1$ with $z^{(j)} \sim \mathcal{N}(0, I)$,
\begin{equation}
  x^{(j+1)} = x^{(j)} + \eta_{t}^{(j)} \bigl( s_{\theta}(x^{(j)}, t) + \lambda \tilde{g}(x^{(j)}) \bigr)
  + \sqrt{2 \eta_{t}^{(j)}} z^{(j)},
  \qquad u = s_{\theta},
  \label{eq:ula}
\end{equation}
with the step size set by the dynamic signal-to-noise rule of \citet{CITE:song-score-sde},
$\eta_{t} = 2 \bigl( \mathrm{snr} \lVert z \rVert / \lVert s_{\theta} + \lambda \tilde{g} \rVert \bigr)^{2}$.
The score term pulls the particle back toward $p_{t}$ while the reward term pushes it up $f$, which
is precisely the structural consistency that sampler~(1) loses. The dynamics are unadjusted — a
Metropolis ratio would require $\log p_{t}$, which is unavailable, whereas its gradient is — so the
stationary distribution is $q_{t}$ only in the $\eta \to 0$ limit and carries an $O(\eta)$
discretization bias at finite step size. Setting $M = 0$ recovers sampler~(1) exactly, which makes
the comparison between the two an ablation inside a single run. We swept $M \in \{1, 2, 4\}$,
bounded above by cost: the sampler needs $n_{\text{steps}}(1 + M)$ velocity evaluations and as many
reward backwards. $M = 1$ gave the best results, buying a clear increase in novelty at matched
$\lambda$ while preserving prompt recognizability; larger $M$ progressively blurred the samples, the
repeated noise injection and re-equilibration at a fixed $t$ washing out the high-frequency detail
that the predictor had already resolved.

\paragraph{(3) Classifier-free guidance on the transport field.}
The third sampler asks whether a \emph{sharper} base field is a stiffer structure that better
resists off-manifold degradation by the reward gradient. Classifier-free guidance
\citep{CITE:cfg-ho-salimans} extrapolates between the conditional and unconditional velocities,
\begin{equation}
  v^{w}_{\theta}(x_{t}, t) = v_{\theta}(x_{t}, t, \emptyset)
  + w \cdot \bigl( v_{\theta}(x_{t}, t, c) - v_{\theta}(x_{t}, t, \emptyset) \bigr),
  \label{eq:cfg}
\end{equation}
where $\emptyset$ is the null embedding and $w$ the guidance scale; $w = 1$ reduces
to~\eqref{eq:flow-guidance} exactly, and larger $w$ increasingly favours features aligned with the
prompt. We substitute $v^{w}_{\theta}$ for the \emph{transport} field only: the denoised estimate
\eqref{eq:xhat0}, the reward, and the reference norm in \eqref{eq:normalized-grad} all keep using
the true conditional velocity $v_{\theta}(\cdot, c)$. Both halves of that choice are load-bearing.
The reward is never evaluated at a $w$-extrapolated point that no reference set covers, so $f$ stays
comparable across $w$; and because CFG inflates the velocity norm, scaling the applied gradient to
$\lVert v^{w}_{\theta} \rVert$ would raise the perturbation \emph{with} $w$ and confound "a stiffer
base field resists the gradient" with "the gradient got bigger". Here $w$ moves the base field and
nothing else. We swept $w \in \{1.5, 2, 3\}$, with $w = 2$ best. Its samples are visibly
higher-quality than sampler~(1)'s at matched $\lambda$, but the gain does not translate into reach:
below a $\lambda$ threshold the stiffer field holds the prompt so well that the samples are clean
but not notably novel, and above that threshold they stop making sense — notably \emph{not} as
granular high-frequency noise, but as coherent-looking colours and shapes with no recognizable
relation to the prompt. Note that our backend is guidance-distilled: it consumes a guidance scalar
as a model input and its empty-prompt branch is not a trained unconditional model
\citep{CITE:flux, CITE:guidance-distillation}, so large $w$ is applied here on top of an already
embedded guidance baseline.

\paragraph{Summary.}
Table~\ref{tab:gradient-guidance-sweeps} collects the three samplers, the settings swept, and the
failure mode that bounds each.

\begin{table}[h]
  \centering
  \begin{tabular}{llll}
    \toprule
    Sampler & Gradient enters via & Swept & Best \\
    \midrule
    (1) Flow guidance & ODE transport, Eq.~\eqref{eq:flow-guidance} & $\lambda$ & --- \\
    (2) Predictor--corrector & ULA drift, Eq.~\eqref{eq:ula} & $M \in \{1, 2, 4\}$ & $M = 1$ \\
    (3) CFG transport & ODE transport, Eq.~\eqref{eq:cfg} & $w \in \{1.5, 2, 3\}$ & $w = 2$ \\
    \bottomrule
  \end{tabular}
  \caption{The three gradient-guided samplers. All share one frozen reward, reference set and noise
  schedule; $M = 0$ in (2) and $w = 1$ in (3) each reduce exactly to (1).}
  \label{tab:gradient-guidance-sweeps}
\end{table}
```

## Citation placeholders used

| Key | What it should point at |
|---|---|
| `CITE:flow-matching` | Lipman et al., Flow Matching for Generative Modeling |
| `CITE:rectified-flow` | Liu et al., Rectified Flow (or Albergo & Vanden-Eijnden interpolants) |
| `CITE:flux` | FLUX.1-dev / Black Forest Labs technical report — the backend |
| `CITE:universal-guidance` | Bansal et al., Universal Guidance for Diffusion Models — the $\partial \hat{x}_{0}/\partial x_{t} \approx I$ transport |
| `CITE:song-score-sde` | Song et al., Score-Based Generative Modeling through SDEs — predictor--corrector and the SNR step-size rule |
| `CITE:cfg-ho-salimans` | Ho & Salimans, Classifier-Free Diffusion Guidance |
| `CITE:guidance-distillation` | guidance distillation (Meng et al.) — why the null branch is not a trained unconditional model |

Add `CITE:sd3` (Esser et al.) if you also mention SD3/3.5 as a same-family backend, and cross-reference
the IEM reward's own citation wherever $f$ is first defined in an earlier section.

## Notes on claims left deliberately soft

- Sampler (1)'s useful $\lambda$ band is written without numbers on purpose: the project's own runs show
  the breakdown point varies about $3\times$ across initial noise draws, so a fixed interval quoted from
  one seed would be a seed-specific statement. Insert numbers only alongside the per-seed caveat.
- Sampler (2)'s "+novelty at matched $\lambda$" is stated qualitatively for the same reason — the measured
  effect is robust in direction across seeds, but it does not move the breakdown point, so avoid phrasing
  that implies a wider window.
- No scalar diagnostic ($f$, latent norm, high-frequency power fraction) reliably predicts recognizability
  across seeds, so "best" above refers to image judgement, not to a metric optimum. Worth one sentence in
  the experimental-protocol section if it is not already there.
