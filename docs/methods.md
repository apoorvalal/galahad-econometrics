# Objectives, steps and covariance

## Shared optimization model

For objective $f(\beta)$, gradient $g$ and exact Hessian $H$, TRU solves a quadratic
trust-region subproblem and ARC solves a cubic-regularized one:

$$
\min_{\|s\|\leq\Delta}\;g^Ts+\tfrac12s^THs,
\qquad
\min_s\;g^Ts+\tfrac12s^THs+\tfrac{\sigma}{3}\|s\|^3.
$$

Both adapt step control using the agreement between observed and predicted
objective reduction. Direct subproblem solvers can handle indefinite Hessians;
the adapter supplies their packed lower triangle and manages the native
initialize/import/solve/information/terminate lifecycle. Scaling changes the
optimization coordinates, and returned parameters remain in the original units.

First-order accuracy certifies stationarity at the supplied tolerance. It is not
a second-order or global-optimality certificate. Nonconvex NLS can still require
multiple starting points.

## Statistical criteria

Let $r_i=x_i^T\beta-y_i$. OLS minimizes $\sum_i r_i^2/(2n)$ and ridge adds
$\sum_j\lambda_j\beta_j^2/2$. Both have constant Hessians; the ridge intercept
penalty is zero when a scalar penalty and an added intercept are used.

Binary logit minimizes mean Bernoulli negative log-likelihood. Poisson minimizes
$n^{-1}\sum_i[\exp(\eta_i)-y_i\eta_i]$, where $\eta_i=x_i^T\beta+o_i$ includes any
supplied offset. The omitted $\log(y_i!)$ term is parameter constant. Poisson
PML permits fractional nonnegative responses; consistency then rests on its
conditional-mean restriction. Logit calculations use stable sigmoid and
log-sum-exp formulas without truncating coefficient values.

For fixed-weight linear IV/GMM, define

$$
m(\beta)=Z^T(X\beta-y)/n,\qquad D=Z^TX/n,\qquad
f(\beta)=\tfrac12m(\beta)^TWm(\beta).
$$

Then $g=D^TWm$ and $H=D^TWD$. Taking $W=(Z^TZ/n)^{-1}$ yields 2SLS. A supplied
weight remains fixed throughout optimization. Exogenous regressors must be
included in the instrument set by the caller.

For nonlinear least squares with means $\mu_i(\beta)$, residuals
$r_i=\mu_i-y_i$ and mean Jacobian $J$, the exact derivatives are

$$
f=\frac{r^Tr}{2n},\qquad g=\frac{J^Tr}{n},\qquad
H=\frac{J^TJ+\sum_i r_i\nabla^2\mu_i}{n}.
$$

The second term is retained. For $\mu_i=\exp(x_i^T\beta)$ the Hessian is
$X^T\operatorname{diag}[\mu_i(2\mu_i-y_i)]X/n$ and can be indefinite away from a
solution. Generic NLS supplies the weighted Hessian sum directly, avoiding
an observation-by-parameter-by-parameter tensor.

Synthetic control minimizes $\|Xw-y\|^2/(2n)$ subject to $w\geq0$ and
$\mathbf1^Tw=1$. Its QPB representation is $H=X^TX/n$, linear term $-X^Ty/n$
and constant $y^Ty/(2n)$. For a feasible weight vector, the convex problem's
Frank–Wolfe gap $g^Tw-\min_j g_j$ is an independently checked optimality bound.
The native output is not clipped or renormalized to manufacture feasibility.

## Inference normalization

For additive mean criteria, let $s_i$ be the individual loss gradient and let
$H$ be the mean-criterion Hessian at the estimate. HC0 uses

$$
\widehat{\operatorname{Var}}(\hat\beta)=
H^{-1}\left(\sum_i s_is_i^T\right)H^{-T}/n^2.
$$

HC1 multiplies by $n/(n-p)$. For NLS the bread is the exact observed Hessian,
including residual curvature, rather than a silently substituted Gauss–Newton
bread. Under a correctly specified conditional mean their asymptotic limits
agree. Finite-sample sandwich estimates still require adequate identification
and a relevant local optimum.

For linear GMM, the effective observation gradient is
$s_i=D^TWz_i(x_i^T\beta-y_i)$. Combining these gradients with $H=D^TWD$ yields
the conventional fixed-weight GMM sandwich. The default 2SLS weight is treated
as the sample instrument-second-moment weight, not an estimated optimal
second-step weight.

For clusters $c=1,\dots,G$, replace the meat with
$\sum_c S_cS_c^T$, where $S_c=\sum_{i\in c}s_i$, and multiply by
$[G/(G-1)][(n-1)/(n-p)]$. Intervals use a normal reference distribution.
No ridge or synthetic-control inference is provided in this prototype.

## References

- [GALAHAD TRU](https://ralna.github.io/galahad_docs/html/Python/tru.html),
  [ARC](https://ralna.github.io/galahad_docs/html/Python/arc.html) and
  [QPB](https://ralna.github.io/galahad_docs/html/Python/qpb.html) interfaces.
- Cartis, Gould and Toint (2011), *Adaptive cubic regularisation methods for
  unconstrained optimization*, Mathematical Programming 127, 245–295.
- White (1980), *A heteroskedasticity-consistent covariance matrix estimator and
  a direct test for heteroskedasticity*, Econometrica 48, 817–838.
- Hansen (1982), *Large sample properties of generalized method of moments
  estimators*, Econometrica 50, 1029–1054.
- [Econometric optimizer comparison](https://lalten.org/quomodoc/docs/econometric-optimizer-comparison):
  seven families and 84 instances; the solver study motivating this prototype.
