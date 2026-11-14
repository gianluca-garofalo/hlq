# The hlq algorithm

This document derives the method that hlq implements. Every claim is derived, nothing beyond standard linear algebra is assumed, and every algorithmic step names the exact operation to perform; the names in `code` are those of `include/hlq`. Section 18 describes how the implementation is verified.

`Every paragraph is kept on one source line so inline math never breaks across a line boundary.`

The idea in one sentence: put every active row of every level into one Gram–Schmidt factorization, $\bar A = K Q$, taken in priority order. The rows of $Q$ are a single orthonormal basis shared by all levels, and $K$ is lower triangular. Then:

- the primal solution is a block forward substitution with $K$;
- all multipliers come from a block backward substitution with $K^\top$;
- adding or removing a row only changes the rows after it, and Givens rotations restore them.

No null-space basis $Z_k$ is ever formed.

### Notation

- $n$: number of variables; $x \in \mathbb R^n$.
- $m$: number of levels, $k = 1, \dots, m$; level 1 has the highest priority.
- $A_k \in \mathbb R^{m_k \times n}$, $b_k \in \mathbb R^{m_k}$: rows and targets of level $k$.
- $\bar A_k$: $A_1, \dots, A_k$ stacked; $\bar A := \bar A_m$, with $M := \sum_k m_k$ rows.
- $a_i \in \mathbb R^{1 \times n}$: row $i$ of $\bar A$ (a row vector), so $a_i x$ is a scalar.
- $\mathcal N(\cdot)$, $\mathcal R(\cdot)$: null space and range; the row space of $A$ is $\mathcal R(A^\top)$.
- $P_k$: orthogonal projector onto $\mathcal N(\bar A_k)$; $P_0 = I$.
- $X^\dagger$: Moore–Penrose pseudo-inverse.
- $\tau$: the rank tolerance, `Settings::rank_tolerance` ($10^{-9}$ in double); $\epsilon_k := \tau \max_{i \in \text{level } k} \lVert a_i \rVert$ is the absolute rank threshold of level $k$, written $\epsilon$ when the level is clear.
- $\mathcal I_k$: indices of the rows of level $k$ in factor order; $\mathcal J_k \subseteq \mathcal I_k$: its independent rows; $r_k := |\mathcal J_k|$, $s_k := m_k - r_k$.
- $\mathcal J := \bigcup_k \mathcal J_k$, $r := |\mathcal J| = \mathrm{rank}\, \bar A$.
- $K$, $Q$: the factors of §3; $\kappa_{ij}$ are the entries of $K$, and $q_i$ the rows of $Q$.
- Sub-blocks:
  - $\hat Q_k := Q[\mathcal J_k, :]$ ($r_k \times n$) and $\hat Q := Q[\mathcal J, :]$ ($r \times n$);
  - $K_{kj} := K[\mathcal I_k, \mathcal J_j]$ ($m_k \times r_j$);
  - $F_k := K_{kk}$.


## Part I — Equality hierarchy

### 1. The problem

The hierarchy is defined lexicographically. Set $S_0 := \mathbb R^n$ and, for $k = 1, \dots, m$,

$$
S_k := \arg\min_{x \in S_{k-1}} \tfrac12 \lVert A_k x - b_k \rVert^2 \tag{1}
$$

The solution is the point of $S_m$ closest to a given reference $x^{(0)}$:

$$
x^\star := \arg\min_{x \in S_m} \tfrac12 \lVert x - x^{(0)} \rVert^2 \tag{2}
$$

Three preprocessing steps bring every problem to this form with $\Lambda_k = I$, the Euclidean metric and $x^{(0)} = 0$:

- **Weights.** A weighted level $\tfrac12 (A_k x - b_k)^\top \Lambda_k (A_k x - b_k)$ with $\Lambda_k = R_k^\top R_k$ becomes (1) after $A_k \leftarrow R_k A_k$, $b_k \leftarrow R_k b_k$.
- **Metric.** A final metric $\lVert \cdot \rVert_H$ with $H = U^\top U$ becomes Euclidean after $x = U^{-1} \tilde x$, $A_k \leftarrow A_k U^{-1}$ (`Solver::set_metric`).
- **Reference.** A nonzero $x^{(0)}$ (`Solver::set_reference`) is removed by $b_k \leftarrow b_k - A_k x^{(0)}$; at the end, $x^{(0)}$ is added back.

From here on, $x^{(0)} = 0$ unless stated otherwise.

### 2. Solution by projections

Define $B_k := A_k P_{k-1}$ (the rows of level $k$ projected onto what the higher levels leave free) and

$$
g_k := b_k - A_k x^{(k-1)}, \qquad x^{(k)} := x^{(k-1)} + B_k^\dagger g_k \tag{3}
$$

**Claim.** $S_k = x^{(k)} + \mathcal N(\bar A_k)$ for every $k$, and $x^\star = x^{(m)}$.

*Proof.* By induction; $S_0 = 0 + \mathcal N(\bar A_0) = \mathbb R^n$.
- **Restrict to $S_{k-1}$.** By the induction hypothesis, $S_{k-1} = \{x^{(k-1)} + \delta : \delta \in \mathcal N(\bar A_{k-1})\}$. For such $\delta$, $A_k \delta = A_k P_{k-1} \delta = B_k \delta$, so (1) is $\min \tfrac12 \lVert B_k \delta - g_k \rVert^2$ over $\delta \in \mathcal N(\bar A_{k-1})$.
- **Minimizers.** Over all of $\mathbb R^n$ the minimizers are $B_k^\dagger g_k + \mathcal N(B_k)$. Moreover $B_k^\dagger g_k \in \mathcal R(B_k^\top) = \mathcal R(P_{k-1} A_k^\top) \subseteq \mathcal N(\bar A_{k-1})$, so it is feasible.
- **Feasible minimizers.** They are $B_k^\dagger g_k + \big(\mathcal N(B_k) \cap \mathcal N(\bar A_{k-1})\big)$, and $\mathcal N(B_k) \cap \mathcal N(\bar A_{k-1}) = \{\delta \in \mathcal N(\bar A_{k-1}) : A_k \delta = 0\} = \mathcal N(\bar A_k)$.
- **Closest to the start.** $x^{(m)} = \sum_k B_k^\dagger g_k$ is a sum of vectors in $\mathcal R(B_k^\top)$. Each of these is orthogonal to $\mathcal N(B_k) \supseteq \mathcal N(\bar A_m)$, so $x^{(m)} \perp \mathcal N(\bar A_m)$. Since $S_m = x^{(m)} + \mathcal N(\bar A_m)$, $x^{(m)}$ is the point of $S_m$ closest to $0$. $\square$

Two consequences are used below:
- On $S_k$, $A_j x = A_j x^{(k)}$ for every $j \le k$, because $\mathcal N(\bar A_k) \subseteq \mathcal N(A_j)$. Each solved level therefore acts on the lower levels as equality constraints at its achieved values.
- The residual of level $k$ is final once level $k$ is solved.

### 3. The priority-ordered Gram–Schmidt factorization

Order the rows by level, level 1 first. Rows inside a level may be reordered freely, since (1) does not depend on their order. Build, for every row $i$, a direction $q_i$ (a unit row vector, or zero) and coefficients $\kappa_{ij}$, level by level:

1. **Start the level.** For each row $l$ of the level, project it on every direction found so far: $\kappa_{lj} = a_l q_j^\top$ and residual $c_l = a_l - \sum_j \kappa_{lj} q_j$.
2. **Pivot.** Take next the remaining row $i$ of the level with the largest $\lVert c_i \rVert$.
3. **Second pass.** If the first pass cancelled most of the row, $\lVert c_i \rVert^2 < \tfrac12 \lVert a_i \rVert^2$, then for every direction found so far $\kappa_{ij} \leftarrow \kappa_{ij} + c_i q_j^\top$ and $c_i \leftarrow c_i - (c_i q_j^\top)\, q_j$. This keeps $Q$ orthonormal to working precision (§15).
4. **Decide the rank.** If $\lVert c_i \rVert > \epsilon$, row $i$ is independent: $\kappa_{ii} = \lVert c_i \rVert$ and $q_i = c_i / \kappa_{ii}$. Otherwise row $i$ is dependent: $\kappa_{ii} = 0$ and $q_i = 0$.
5. **Update the level.** If $q_i \neq 0$, then for every remaining row $l$ of the level, $\kappa_{li} = c_l q_i^\top$ and $c_l \leftarrow c_l - \kappa_{li} q_i$. Go back to step 2 until the level is done.

Since $a_i = \sum_{j \le i} \kappa_{ij} q_j$ by construction,

$$
\bar A = K Q, \qquad K \ \text{lower triangular}, \qquad q_i q_j^\top = \begin{cases} 1 & i = j \in \mathcal J \\ 0 & \text{otherwise} \end{cases} \tag{4}
$$

A dependent row $i$ has a zero row $q_i$, hence a zero column $K[:, i]$, and only its independent columns are stored.

The pivoting has three consequences:
- Within a level, the independent rows come first and the dependent ones last: once the largest residual of a level is $\le \epsilon$, all of them are.
- $\lvert \kappa_{lj} \rvert \le \kappa_{jj}$ for rows $l, j$ of the same level. When $j$ was chosen, its residual was the largest, and $\kappa_{lj} = c_l q_j^\top$ is bounded by $\lVert c_l \rVert$.
- The threshold is absolute, $\epsilon_k = \tau \max_{i \in k} \lVert a_i \rVert$, because rounding noise is absolute (§14).

The pivot search keeps the residual norms of the level and downdates them, $\lVert c_l \rVert^2 \leftarrow \lVert c_l \rVert^2 - \kappa_{li}^2$, recomputing them once they fall below $\sqrt u$ of where they started (`PriorityLQ::factor`).

### 4. What the factors contain

**Lemma 1.** For every $k$, the rows of $\hat Q_1, \dots, \hat Q_k$ are an orthonormal basis of $\mathcal R(\bar A_k^\top)$, and $P_k = I - \sum_{j \le k} \hat Q_j^\top \hat Q_j$.
*Proof.* Gram–Schmidt keeps $\mathrm{span}\{q_j : j \le i\} = \mathrm{span}\{a_j : j \le i\}$; the zero rows add nothing. $\square$

**Lemma 2.** $B_k = A_k P_{k-1} = F_k \hat Q_k$, and $F_k$ has full column rank $r_k$.
*Proof.* $A_k = \sum_{j \le k} K_{kj} \hat Q_j$, $\hat Q_j P_{k-1} = 0$ for $j < k$, and $\hat Q_k P_{k-1} = \hat Q_k$ (Lemma 1). The rows $\mathcal J_k$ of $F_k$ form a lower triangular $r_k \times r_k$ matrix with diagonal $\kappa_{ii} > 0$. $\square$

So the level-$k$ block of $K$ is the level's projected rows written in a basis that all levels share, and $\mathrm{rank}\, B_k = r_k$. The null-space basis $Z_k$ of the classical formulation is any orthonormal basis of the complement of $\hat Q_1, \dots, \hat Q_k$; it is never needed. $\bar A \hat Q^\top = K[:, \mathcal J]$ is block lower triangular with full-column-rank diagonal blocks.

**Gram form.** $G := \bar A \bar A^\top = K K^\top$, and the Schur complement of $G$ after the levels before $k$ is $F_k F_k^\top = B_k B_k^\top$. With $K = L D^{1/2}$, this is the LDL factorization of DAQP's $M = A_W H^{-1} A_W^\top$ (with $H = I$), taken in priority order. §16 explains why $G$ itself should not be formed.

### 5. Primal solution: block forward substitution

By Lemma 2, $B_k = F_k \hat Q_k$ with $F_k$ of full column rank and $\hat Q_k$ of full row rank, so $B_k^\dagger = \hat Q_k^\dagger F_k^\dagger = \hat Q_k^\top F_k^\dagger$. Write $x^{(k)} = \sum_{j \le k} \hat Q_j^\top y_j$. Since $\hat Q_j \hat Q_l^\top = \delta_{jl} I$ and $A_k = \sum_{j \le k} K_{kj} \hat Q_j$, (3) becomes

$$
g_k = b_k - \sum_{j<k} K_{kj}\, y_j, \qquad y_k = F_k^\dagger g_k, \qquad x = \hat Q^\top y \tag{5}
$$

Here $y$ stacks all $y_k$ ($r$ entries). Rows of later levels do not change $A_k x$, because $A_k \hat Q_j^\top = 0$ for $j > k$. Hence the residual of level $k$ in the final solution is exactly the residual of its own least-squares problem:

$$
\rho_k := A_k x - b_k = F_k y_k - g_k \tag{6}
$$

**Computing $y_k = F_k^\dagger g_k$.** Split level $k$ into its independent and dependent rows:

$$
F_k = \begin{bmatrix} K_k^a \\ K_k^b \end{bmatrix}, \quad g_k = \begin{bmatrix} g_k^a \\ g_k^b \end{bmatrix}, \qquad K_k^a \in \mathbb R^{r_k \times r_k} \ \text{lower triangular}, \quad K_k^b \in \mathbb R^{s_k \times r_k}
$$

- **No dependent row ($s_k = 0$):** $y_k = (K_k^a)^{-1} g_k$ by forward substitution, and $\rho_k = 0$.
- **No independent row ($r_k = 0$):** $y_k$ is empty and $\rho_k = -g_k$.
- **Otherwise the level conflicts:** least squares on $F_k$. Take its Householder QR, $F_k = H_k \begin{bmatrix} R_k \\ 0 \end{bmatrix}$ with $H_k$ orthogonal and $R_k$ upper triangular ($r_k \times r_k$), and $c := H_k^\top g_k = \begin{bmatrix} c^a \\ c^b \end{bmatrix}$:

$$
y_k = R_k^{-1} c^a \tag{7}
$$

$$
\rho_k = F_k y_k - g_k = H_k \begin{bmatrix} 0 \\ -c^b \end{bmatrix} \tag{8}
$$

The normal equations of this problem, or the equivalent small system $(I + E_k E_k^\top) w = g^b - E_k g^a$ with $E_k = K_k^b (K_k^a)^{-1}$, would square the conditioning of the level: an ill-conditioned level then loses half of its digits. hlq computes the QR in place, in the diagonal block of the level of a workspace (`detail::householder_qr`), and §6 reuses it.

With a reference, $x = x^{(0)} + \hat Q^\top y$ (in the metric's coordinates, $x = x^{(0)} + U^{-1} \hat Q^\top y$).

$y_k$ depends only on the levels up to $k$. After an update (§12) of a row of level $k$, the levels before $k$ keep their $y_j$, $\rho_j$ and QRs, and their parts $\hat Q_j^\top y_j$ of $x$; only levels $k$ and later are solved again.

### 6. Multipliers: block backward substitution

The inequality case (Part II) needs the multipliers of the higher levels' rows in the subproblem of level $h$:

$$
\min_x \tfrac12 \lVert A_h x - b_h \rVert^2 \quad \text{s.t.} \quad \bar A_{h-1} x = \bar A_{h-1} x^{(h-1)} \tag{9}
$$

With $\lambda_h := \rho_h$, stationarity of its Lagrangian $\tfrac12 \lVert A_h x - b_h \rVert^2 + \sum_{k<h} \lambda_k^\top (A_k x - A_k x^{(h-1)})$ reads

$$
\sum_{k \le h} A_k^\top \lambda_k = 0 \tag{10}
$$

The $\lambda_k$ are not unique when rows are dependent. hlq picks them level by level, from $k = h-1$ down to $1$. Each $\lambda_k$ is the minimum-norm vector such that whatever remains can be absorbed by the levels above $k$:

$$
\tau_{k+1} := \sum_{j=k+1}^{h} A_j^\top \lambda_j, \qquad B_k^\top \lambda_k = -P_{k-1} \tau_{k+1}, \qquad \lambda_k = -(B_k^\top)^\dagger P_{k-1} \tau_{k+1} \tag{11}
$$

**Solvability.** By induction, $\tau_{k+1} \in \mathcal R(\bar A_k^\top)$:
- For $k = h-1$, the normal equations of level $h$ give $B_h^\top \rho_h = 0$, i.e. $P_{h-1} A_h^\top \lambda_h = 0$.
- Then $P_{k-1} \tau_{k+1} \in P_{k-1} \mathcal R(\bar A_k^\top) = \mathcal R(B_k^\top)$, so (11) has an exact solution. That solution gives $P_{k-1} \tau_k = 0$, i.e. $\tau_k \in \mathcal R(\bar A_{k-1}^\top)$.

**Factored form.** Keep the coordinates $\theta := \hat Q \tau \in \mathbb R^r$, in blocks $\theta_j$:
- Since $\tau_{k+1} \in \mathrm{span}(\hat Q_1, \dots, \hat Q_k)$ (Lemma 1), $P_{k-1} \tau_{k+1} = \hat Q_k^\top \theta_k$.
- By Lemma 2, $B_k^\top \lambda_k = \hat Q_k^\top F_k^\top \lambda_k$.
- Adding $A_k^\top \lambda_k = \sum_{j \le k} \hat Q_j^\top K_{kj}^\top \lambda_k$ to $\tau$ adds $K_{kj}^\top \lambda_k$ to $\theta_j$.

Hence:

$$
\theta_j \leftarrow K_{hj}^\top \rho_h \ (j < h); \quad \text{for } k = h-1, \dots, 1: \quad F_k^\top \lambda_k = -\theta_k,\ \lambda_k \in \mathcal R(F_k); \quad \theta_j \leftarrow \theta_j + K_{kj}^\top \lambda_k \ (j < k) \tag{12}
$$

**Solving $F_k^\top \lambda_k = -\theta_k$ with minimum norm.** With the QR of §5, $F_k^\top = \begin{bmatrix} R_k^\top & 0 \end{bmatrix} H_k^\top$, so

$$
\lambda_k = H_k \begin{bmatrix} -R_k^{-\top} \theta_k \\ 0 \end{bmatrix} \tag{13}
$$

which lies in $\mathcal R(F_k)$, spanned by the first $r_k$ columns of $H_k$. When $s_k = 0$, $\lambda_k = -(K_k^a)^{-\top} \theta_k$.

**The final choice among the optima.** After the last level, the solution minimizes $\tfrac12 \lVert \tilde x - \tilde x^{(0)} \rVert^2$ (metric coordinates) over $S_m$. That is one more subproblem, whose gradient is the step $\hat Q^\top y$ itself, and every level is above it: its multipliers are (12) started from $\theta = y$ and run from $k = m$ down (`PriorityLQ::multipliers` with `h` the number of levels factored).

**A last level that is the final choice.** If level $m$ consists of equalities $F x = b$ and $F$ has full column rank, it leaves a single point: on $S_{m-1}$ it minimizes $\lVert F x - b \rVert^2 = (x - r)^\top F^\top F (x - r) + c$, with $r = F^+ b$ and $c$ a constant. That is the final choice among the optima of levels $1, \dots, m-1$ in the metric $W = F^\top F$ with reference $r$, so hlq does not search the level: with $F = Q R$ it takes $C = R^{-1}$ and $r = R^{-1} Q^\top b$ (`Solver::fold`). A regularization such as $x = 0$ then costs nothing. The level is reported as if it had been searched. Its cost is computed from the solution, and its rank is $n$ minus the ranks above. In its subproblem the gradient $F^\top (F x - b) = W (x - r)$ is that of the final choice, so the multipliers of the levels above are the final choice's, and those of its own rows are their residuals. A level conditioned worse than $1 / \sqrt{\epsilon}$ is searched instead, and `Settings::fold_final_level` turns the fold off.

### 7. Example 1 (equalities, a conflict, a dependent row)

$x \in \mathbb R^3$, three levels:

| level | row | target |
|---|---|---|
| 1 | $a_1 = (1, 1, 0)$ | $2$ |
| 2 | $a_2 = (1, 0, 0)$ | $2$ |
| 2 | $a_3 = (0, 1, 0)$ | $1$ |
| 3 | $a_4 = (1, 0, 0)$ | $0$ |
| 3 | $a_5 = (0, 0, 1)$ | $3$ |

**Factorization.**
- Level 1: $q_1 = (1,1,0)/\sqrt2$, $\kappa_{11} = \sqrt2$.
- Level 2: both residuals are $(\pm\tfrac12, \mp\tfrac12, 0)$, a tie, so $a_2$ goes first. $\kappa_{21} = 1/\sqrt2$, $q_2 = (1,-1,0)/\sqrt2$, $\kappa_{22} = 1/\sqrt2$. Then $a_3 = \tfrac1{\sqrt2} q_1 - \tfrac1{\sqrt2} q_2$ has zero residual, so it is dependent.
- Level 3: $a_5$ has the larger residual, so it goes first: $q_5 = (0,0,1)$, $\kappa_{55} = 1$. Then $a_4 = \tfrac1{\sqrt2} q_1 + \tfrac1{\sqrt2} q_2$ is dependent.

In factor order $(a_1, a_2, a_3, a_5, a_4)$, with columns $(q_1, q_2, q_5)$:

$$
K[:, \mathcal J] = \begin{bmatrix} \sqrt2 & 0 & 0 \\ 1/\sqrt2 & 1/\sqrt2 & 0 \\ 1/\sqrt2 & -1/\sqrt2 & 0 \\ 0 & 0 & 1 \\ 1/\sqrt2 & 1/\sqrt2 & 0 \end{bmatrix}
$$

**Forward substitution (5)–(8).**
- Level 1: $y_1 = 2/\sqrt2 = \sqrt2$.
- Level 2: $g_2 = (2,1) - (1/\sqrt2)\sqrt2\,(1,1) = (1, 0)$, and $F_2 = (1/\sqrt2, -1/\sqrt2)^\top$, so the least squares of (7) gives $y_2 = F_2^\top g_2 / \lVert F_2 \rVert^2 = \tfrac1{\sqrt2}$. The residuals are $\rho_2 = (-\tfrac12, -\tfrac12)$: the conflict is split evenly.
- Level 3: $g_3 = (3, 0 - (1 + \tfrac12)) = (3, -\tfrac32)$ and $F_3 = (1, 0)^\top$, so $y_3 = 3$ and $\rho_3 = (0, \tfrac32)$.
- Result: $x = \sqrt2\, q_1 + \tfrac1{\sqrt2} q_2 + 3 q_5 = (\tfrac32, \tfrac12, 3)$.

**Backward substitution (12)–(13), for $h = 3$.**
- $\theta_1 = \theta_2 = (1/\sqrt2)\cdot\tfrac32 = \tfrac3{2\sqrt2}$.
- Level 2: $\lambda_2$ is the minimum-norm solution of $F_2^\top \lambda_2 = -\theta_2$, $\lambda_2 = -F_2 \theta_2 / \lVert F_2 \rVert^2 = (-\tfrac34, \tfrac34)$.
- Then $\theta_1 \leftarrow \theta_1 + \tfrac1{\sqrt2}(-\tfrac34 + \tfrac34) = \tfrac3{2\sqrt2}$.
- Level 1: $\lambda_1 = -\tfrac1{\sqrt2}\cdot\tfrac3{2\sqrt2} = -\tfrac34$.
- Check (10): $-\tfrac34 a_1 - \tfrac34 a_2 + \tfrac34 a_3 + \tfrac32 a_4 + 0\, a_5 = 0$.


## Part II — Inequalities

### 8. The problem

Each row has bounds $l_i \le a_i x \le u_i$ (an equality when $l_i = u_i$). Level $k$ minimizes its violation

$$
f_k(x) = \tfrac12 \sum_{i \in \text{level } k} \mathrm{dist}\big(a_i x, [l_i, u_i]\big)^2 \tag{14}
$$

over the optimal set of the levels above it, as in (1).

**Working set.** $W$ is a set of rows, each with a side $\sigma_i \in \{+1, -1\}$ (upper or lower bound) and target $\beta_i = u_i$ or $l_i$. Equality rows are always in $W$. For fixed $W$, the problem "make the rows of $W$ equal to their targets, level by level" is Part I. Write $\bar x(W)$ for its solution.

**Locked rows.** These are rows of $W$ that may no longer be removed. Equality rows are locked from the start. A row can also be locked for one phase only (§9, step 2).

### 9. The active-set loop

$\hat x$ is the iterate (`Solver::point_`) and $\bar x$ the solution of the current working set (`Solver::primal_`). Tolerances: $\tau_f$ is `feasibility_tolerance` (absolute), $\tau_d$ is `dual_tolerance`, relative: a multiplier of subproblem $h$ counts as nonzero above $z_h := \tau_d (1 + \max_i \lvert \lambda_i \rvert)$, since its rounding noise scales with the largest multiplier of the subproblem.

0. **Start.** $W$ holds the equalities plus the warm start (the active set of the previous solve, when the problem's structure is unchanged). Factor (§3), solve (§5), and set $\hat x \leftarrow \bar x$. The factorization of the previous solve's final $W$ is kept: when neither the rows, the metric nor $W$ changed since (only the bounds or the reference did), it is reused instead of factored again.
1. **For $h = 1, \dots, m + 1$:** phase $m + 1$ is the final choice among the optima (§6).
   1. **Activate violated rows** (levels only). If level $h$ left the factorization in an earlier phase, factor its rows of $W$ after the levels above it (§3, one level). Insert (§12) every inactive row $i$ of level $h$ with $a_i \hat x > u_i + \tau_f$ (side $+1$) or $a_i \hat x < l_i - \tau_f$ (side $-1$).
   2. **Ratio test.** Solve for $\bar x$. Among the inactive rows of levels $\le h$ (every level, in the last phase) with $a_i \bar x$ outside $[l_i - \tau_f, u_i + \tau_f]$, take the smallest step to the violated bound $\beta_i$:
      $$
      t_i = \mathrm{clamp}\Big(\frac{\beta_i - a_i \hat x}{a_i \bar x - a_i \hat x}, 0, 1\Big) \tag{15}
      $$
      The denominator is nonzero because $\hat x$ satisfies these rows. If such a row exists, set $\hat x \leftarrow \hat x + t_i (\bar x - \hat x)$, insert row $i$ at that bound, and repeat step 2. If row $i$ is the row the last change removed and $\hat x$ moved by at most $\tau_f$ in every coordinate, the insertion undoes that removal; the second time this happens to a row in phase $h$, lock it until the end of the phase.
   3. **Drop test.** Set $\hat x \leftarrow \bar x$ and compute the multipliers of subproblem $h$ (§6). For a level's own rows these are its residuals, $\lambda_i = \rho_i$. Let $i$ be the unlocked row of $W$ in levels $\le h$ that maximizes $-\sigma_i \lambda_i$. If $-\sigma_i \lambda_i > z_h$, remove it (§12) and go back to step 2.
   4. **Lock.** Lock every row of $W$ in levels $\le h$ with $\sigma_i \lambda_i > z_h$, then go to the next $h$.
2. **Return $\hat x$**, with `Status::Solved`, or `Status::MaxChanges` once `max_changes` insertions plus removals are spent.

**Why each rule:**
- **Ratio test.** $\hat x$ satisfies every inactive row of the levels already processed, so the levels above $h$ never get worse. Jumping straight to $\bar x$ can violate rows that earlier phases satisfied; that breaks solved levels and cycles.
- **Cycle guard.** A removal that the ratio test undoes at once leaves $W$ and $\hat x$ as they were, so the drop test removes the same row again. Once can be a degenerate step; twice in a phase is a cycle. In the cases seen, the levels above $h$ are ill-conditioned: removing the row lets $\bar x$ improve a level above $h$ by an amount below the tolerances, which the row's own bound forbids. The row's multiplier is then not a usable sign, and the lock stops the search there, as `max_changes` would, but at once.
- **Sign test.** Take a row at its upper bound in subproblem (9), with the inequality $a_i x \le u_i$ in the Lagrangian as $\mu_i (a_i x - u_i)$, $\mu_i \ge 0$. Then $\mu_i = \lambda_i$, so the condition is $\sigma_i \lambda_i \ge 0$; the lower side is symmetric. For a row of level $h$, $\sigma_i \rho_i \ge 0$ says the row is on, or beyond, its bound on the side it is active for.
- **Locking.** If $\sigma_i \lambda_i > z_h$ at level $h$, moving the row off its bound strictly increases $f_h$, so it must stay there for all lower levels. If $\lvert \lambda_i \rvert \le z_h$, level $h$ does not care, and a lower level may still release the row (Example 2). A threshold that does not scale with the multipliers locks rounding noise: in single precision, a multiplier that is $10^{-5}$ only because the others are $10^2$ would lock a bound that a lower level must release.
- **The last phase.** The working set's solution is the point closest to the reference within the affine set of its active rows, but the optimal set may be larger: an inactive-at-every-level constraint (zero multipliers everywhere) may hold the solution away from the reference. Phase $m + 1$ releases those, with the same ratio test.
- **Which levels take part.** Rows of levels $> h$ in $W$ (for example lower-level equalities) take part in the ratio and drop tests only from their own phase on. They stay in the factorization and in $\bar x$ until phase $h$ changes $W$, which keeps a warm start from the optimal active set at zero changes. At the first change they leave the factorization (dropping the positions after level $h$ costs nothing), so that the updates of §12 no longer rotate them and §5 no longer solves them, and each level comes back in its own phase. Phase $h$ needs nothing from them: its ratio test, its drop test and the multipliers of its subproblem involve only levels $\le h$, and its $\bar x$, the solution of levels $\le h$ closest to the reference, is as good a target as any other point of their optimal set.

### 10. Example 2 (a drop decided two levels below)

$x \in \mathbb R^2$:
- level 1: $x_1 \le 1$, row $a_1 = (1, 0)$;
- level 2: $x_1 + x_2 = 4$, row $a_2 = (1, 1)$;
- level 3: $x_2 = 4$, row $a_3 = (0, 1)$.

The optimum is $(0, 4)$, and all three levels are satisfied there. Start warm, with $a_1$ active at its upper bound from a previous solve (with level 3 being $x_2 = 2$, say).

**Initial solve.** $q_1 = e_1$, $\kappa_{11} = 1$; $\kappa_{21} = 1$, $q_2 = e_2$, $\kappa_{22} = 1$; $a_3 = q_2$ is dependent ($\kappa_{31} = 0$, $\kappa_{32} = 1$).
- $y_1 = 1$ and $y_2 = 4 - 1 = 3$, so $\bar x = (1, 3)$.
- Level 3 has no independent row, so $\rho_3 = -g_3 = -(4 - 3) = -1$.

**Phases 1 and 2.** Nothing is inactive. In phase 1 the multiplier of $a_1$ is $\rho_1 = a_1 \bar x - 1 = 0$, so it is neither dropped nor locked; locking it here would make the optimum unreachable. In phase 2, $\rho_2 = 0$ makes every multiplier zero.

**Phase 3.** $\lambda_3 = -1$, so $\theta_1 = 0 \cdot (-1) = 0$ and $\theta_2 = 1 \cdot (-1) = -1$.
- Level 2: $\lambda_2 = 1$, then $\theta_1 = 0 + 1 \cdot 1 = 1$.
- Level 1: $\lambda_1 = -1$, so $\sigma_1 \lambda_1 = -1 < 0$ and $a_1$ is dropped.
- Re-solving (see the update below) gives $\bar x = (0, 4)$. The ratio test finds nothing ($x_1 = 0 \le 1$), and the residuals are zero, so the loop ends.

A rule that drops a row only when its own level's residual has the wrong sign would stay at $(1, 3)$: that residual is zero here. The multiplier that releases $a_1$ belongs to the subproblem of level 3.

**The removal as an update (§11).** Removing $a_1$ leaves the tail $a_2, a_3$ with coefficients $(1, 1)$ and $(0, 1)$ on $(e_1, e_2)$. One rotation of the directions, $d_1 = (e_1 + e_2)/\sqrt2$ and $d_2 = (-e_1 + e_2)/\sqrt2$, makes it triangular again: $a_2 = \sqrt2\, d_1$ and $a_3 = \tfrac1{\sqrt2} d_1 + \tfrac1{\sqrt2} d_2$.
- Row $a_3$ is now independent: the direction freed at level 1 passes through level 2 to level 3.
- In Gram form, the tail's Gram matrix $\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$ gains $k k^\top$ with $k = (1, 0)$, its coefficients on the removed direction: $\begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}$.
- Then $y = (2\sqrt2, 2\sqrt2)$ and $\bar x = (0, 4)$.


## Part III — Updating the factorization

### 11. What changes when a row is removed or inserted

Let row $p$ change, and split the rows into the **lead** (before $p$) and the **tail** (after $p$).
- **The lead does not change.** Gram–Schmidt of a row only sees the rows before it.
- **The tail changes only through its coefficients on the direction of row $p$.** Let $C_t$ be the tail's residuals after the lead and row $p$, $S_t := C_t C_t^\top$, and $k := $ the tail's coefficients on $q_p$.

$$
\text{removal:}\ \ S_t' = S_t + k k^\top, \qquad \text{insertion:}\ \ S_t' = S_t - k k^\top \tag{16}
$$

Removal, because the tail's residuals gain back $k\, q_p$, and $q_p$ is orthogonal to the rows of $C_t$. Insertion, because they lose it: $C_t' = C_t - k q_p^\top$ with $k = C_t q_p^\top$.

Bookkeeping of the rank:
- **$k = 0$ or $q_p = 0$:** the tail does not change.
- **Removal** can only raise the tail's rank: a dependent tail row may become independent, as $a_3$ does in Example 2. If the tail's rank does not grow, the freed direction leaves the factorization, and the rank of $\bar A$ drops by one.
- **Insertion** of an independent row can only lower the tail's rank, by at most one. It drops exactly when $q_p$ lies in the span of the tail's directions: the new row takes a direction that a lower level was using. Otherwise $q_p$ has a component in $\mathcal N(\bar A)$, and the rank of $\bar A$ grows by one.

Nothing else is affected. With a basis per level (nested null spaces $Z_j$), a change at level $k$ forces every $Z_j$, $j \ge k$, and every factorization built on them to be rebuilt. Here all levels share the basis $Q$, so a change is a rotation of some rows of $Q$ together with the matching columns of $K$.

### 12. The updates

Both updates express the rows after the change, the **tail**, in coordinates: on an orthonormal set of rows $B$ (`tail_basis_`) that spans what they have outside the directions before $p$, with coefficients $C$ (`rows_`). Their coefficients on the directions before $p$ do not change; only their positions shift.

**Removal of row $p$** (`PriorityLQ::remove`). $B$ holds $q_p$ (if row $p$ was independent) and the directions of the independent tail positions; $C$ is the tail's columns of $K$ on them. $C$ is in echelon form except for its first column, the removed direction.

**Insertion of row $a$** (`PriorityLQ::insert`) at $p$, after the independent rows of its level. One projection on every direction gives its lead coefficients, its coefficients $w$ on the tail's directions, and its residual $c$; a second projection follows when the first cancelled most of $a$. $B$ holds the tail's directions plus $c / \lVert c \rVert$ when $\lVert c \rVert > \epsilon$; $C$ has the new row $(w, \lVert c \rVert)$ on top of the tail's columns of $K$ (and a zero column). $C$ is in echelon form except for its first row.

**Echelon form by rotations** (`PriorityLQ::rotate_tail`). Process the tail's positions in order, level by level, with a counter `next` of the first coordinate not yet taken:
1. **Pivot** inside the level on the row with the most left on the free coordinates (downdated squared norms, recomputed when they fall below $\sqrt u$ of their start).
2. **Compress** the row's entries on coordinates `next`, ..., last nonzero into coordinate `next`: Givens rotations of adjacent coordinates, from the right, each applied to the columns of $C$ (for the rows from this one on) and to the rows of $B$.
3. **Decide.** If the entry on `next` exceeds $\epsilon_k$ in magnitude, the row is independent: it takes coordinate `next` (sign made positive), its direction is that row of $B$, and the rows below get their coefficients on it from that column of $C$. Otherwise it is dependent, and what it has left is dropped.

The structure keeps this cheap: after a removal each row has one coordinate to absorb (one rotation); after an insertion the new row needs up to $\dim B - 1$ rotations and every later row one or two. Each rotation costs $O(t + n)$ for $t$ tail rows, so an update costs $O(n\, r_{\text{lead}})$ for the projection plus $O(\dim B\,(t + n))$ for the rotations. Coordinates that no row takes are directions the tail no longer needs. A tail row whose new entry exceeds $\epsilon$ becomes independent; when an insertion takes a direction a lower level was using, the step-3 test decides which tail row loses it.

**Pivoting and drift.** Rows keep their order inside a level except as step 1 reorders the tail. A factorization reused across solves (step 0 of §9) is only ever updated, by orthogonal rotations and re-orthogonalized insertions: after 20,000 warm solves in a row, with the bounds moving, the solutions agree with fresh factorizations to $10^{-14}$, and `tests/test_reuse.cpp` checks 2,000.

## Part IV — Cost and numerics

### 13. Cost

For $M$ active rows of rank $r$ in $n$ variables, a change at position $p$ with $t$ rows after it (in phase $h$, only rows of levels $\le h$ are in the factorization):

| operation | cost |
|---|---|
| factorization from scratch (§3) | $O(M n r)$ |
| insertion or removal (§12) | $O(n\, r + \dim B\,(t + n))$ |
| solution of the working set (§5) | $O(M r + n r)$, plus a QR of each conflicting level, from the level of the change on |
| multipliers of a subproblem (§6) | $O(M r)$, reusing those QRs |
| ratio test (15) | $O(n)$ per inactive row of the levels involved, $O(1)$ for a bound on one variable |
| storage | $O(M^2 + M n)$, allocated once by the constructor |

Measured on the families of §18 (see the README for the full table), cold solves from data to solution take about a quarter of the time of HierarchicalQP and LexLS on the large family (20–40 variables), and about the same on small problems. DAQP, a dual method, is faster still, at an accuracy of about $10^{-6}$: hlq makes fewer active-set changes, but each one costs more, since it keeps $\hat Q$ (vectors of length $n$) where DAQP keeps only the LDL factors of the Gram matrix (§4).

### 14. Why the rank threshold is absolute

The rounding error in $\lVert c_i \rVert$ is about $u \max_j \lVert a_j \rVert$ ($u$ = machine precision), whatever the size of $a_i$. A test relative to the residuals themselves lets that noise through: a level whose rows all lie in the span of the levels above projects to pure rounding noise, and a relative test gives it rank. Hence $\epsilon_k = \tau \max_{i \in k} \lVert a_i \rVert$.

### 15. Two passes

One pass of Gram–Schmidt loses orthogonality in proportion to the cancellation in each row. A second pass, done when the first cancelled more than half of the row's squared norm, restores it to working precision (Daniel et al. 1976). This is what lets (5) use $\hat Q_j \hat Q_l^\top = \delta_{jl} I$.

### 16. Do not form $\bar A \bar A^\top$

The Gram form of §4 is equivalent in exact arithmetic, but computing its LDL from $G$ squares the conditioning:
- The pivot of a dependent row comes out as $u \lVert G \rVert \kappa^2$ instead of 0, where $\kappa$ is the conditioning of the earlier levels.
- No threshold below about $\sqrt u$ can tell rank from noise.
- With one level whose smallest singular value is $10^{-1}$ to $10^{-6}$ of its largest, forming $G$ gave wrong level costs on 2102 of 20000 problems in a prototype; the Gram–Schmidt form got 2 wrong, both right at the threshold.

The form of (4) computes $d_i = \lVert c_i \rVert^2$ from the vector itself. Its cost equals that of forming the Gram row ($\bar A a^\top$ is $M n$ either way).

**Gram mode** (`Settings::gram_mode`, off by default) takes that trade on purpose. It keeps $K$ and not $Q$: where $Q x$ is needed, it is $K_I^{-1} (\bar A_I x)$ over the independent positions $I$, and the updates of §12 rotate $K$ alone. Two safeguards keep it usable:
- the rank tolerance is raised to $10 \sqrt u$;
- the rank stops at $n$, because past it what a row has outside the others is pure noise of order $\sqrt u\, \kappa$.

Measured against the default:
- cold solves are 6–13% faster, and DAQP still takes 0.5–0.96 of the time;
- on the random, integer, degenerate and large families it finds the same ranks and solutions to $3 \cdot 10^{-8}$;
- on their ill-conditioned versions it fails the certificate of §18 on 3–12% of problems, against at most 0.05% by default.

DAQP gets away with the Gram form because it never decides rank among soft constraints: each has its own slack, so its rows stay independent. A hierarchy has dependent rows all the time.

### 17. Relation to other methods

- **HCOD.** The factorization is the HCOD of Escande, Mansard and Wieber (IJRR 2014), $\bar A = W H Y^\top$, written without $W$: here $H = K$, and $Y$ is $Q$ completed to an orthogonal matrix. hlq updates it by rotations instead of recomputing it from the changed level down.
- **HierarchicalQP** (Weighted-Hierarchical-QP) computes a complete orthogonal decomposition per level on nested null spaces, recomputed from the changed level down, with the same active-set loop as §9 without its last phase.
- **LexLS** (Dimitrov, Sherikov and Wieber) solves the same problem with a lexicographic QR and its own regularization and tolerances.
- **DAQP** factors $M = A_W H^{-1} A_W^\top$ by LDL with row appends and removals. Its hierarchical mode solves one softened problem per level, with proximal iterations so the slacks are exact, and between levels eliminates frozen rows with Householder reflections. hlq keeps the update idea but orders all levels in one factor: each level's least squares is exact (7), and $M$ is never formed (§16).

### 18. Verification

The tests (`tests/`) check the implementation in three independent ways:

- **Against an SVD reference** (`tests/support/reference.hpp`): nested null spaces with SVD. The factorization's ranks, solution and multipliers must match on random equality hierarchies with rank-deficient levels, rows dependent on the levels above, and ill-conditioned levels; after random insertions and removals, every update must match a fresh factorization.
- **With an optimality certificate** (`tests/support/certificate.hpp`), independent of the solver: at every level, the gradient of its squared violation must be balanced by multipliers of the bounds tight above it with the sign their side allows, found by non-negative least squares (Lawson and Hanson); and the final choice must be balanced the same way. Random, integer, degenerate (many parallel rows) and large problems, cold, warm, from another problem's active set, in a metric, and without a final regularization level. A test checks that the certificate rejects perturbed solutions.
- **Against other solvers** (`benchmarks/compare.cpp`): level by level against LexLS, HierarchicalQP and DAQP, pinned; hlq is never worse on well-conditioned families, and on ill-conditioned ones it is better more often than not.

The no-allocation guarantee is checked by a test that counts the calls to a replaced global `operator new`. CI runs everything with gcc, clang and MSVC, and under the address and undefined-behaviour sanitizers.

**Known limits.**
- On ill-conditioned problems, a level can be decided by differences below the tolerances, and the levels below it then end short of the certificate's optimum: 8 of 40,000 problems of the integer and degenerate ill-conditioned families. The cycle guard of §9 ends the search on such problems, which would otherwise cycle until `max_changes`.
- In single precision, decisions within `feasibility_tolerance` ($10^{-5}$) of a bound can change lower levels of badly balanced problems.

### References

- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., §6.5 (updating matrix factorizations).
- J. W. Daniel, W. B. Gragg, L. Kaufman and G. W. Stewart, "Reorthogonalization and stable algorithms for updating the Gram–Schmidt QR factorization", *Math. Comp.* 30 (1976).
- A. Escande, N. Mansard and P.-B. Wieber, "Hierarchical quadratic programming: Fast online humanoid-robot motion generation", *IJRR* 33(7) (2014).
- D. Arnström, A. Bemporad and D. Axehill, "A dual active-set solver for embedded quadratic programming using recursive LDLᵀ updates", *IEEE TAC* (2022).
- C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*, Prentice-Hall (1974), chapter 23 (non-negative least squares).
