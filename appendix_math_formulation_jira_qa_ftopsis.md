# Appendix: Mathematical Formulation

## A1. Notation

Let:

- \(x\): an incoming Jira issue / ticket
- \(t\): the current simulation slot
- \(j \in \{1,\dots,M\}\): a candidate queue
- \(M\): number of queues
- \(Q_j(t)\): backlog of queue \(j\) at slot \(t\)
- \(\mu_j\): service capacity of queue \(j\)
- \(p_j(x)\): calibrated classifier probability that ticket \(x\) belongs to queue \(j\)
- \(p_{\max}(x)=\max_j p_j(x)\)
- \(c(x)\): ticket complexity score
- \(W_i\): waiting time of completed ticket \(i\)
- \(y_i\): true queue label of ticket \(i\)
- \(\hat y_i\): chosen queue of ticket \(i\)

The routing task is to select a queue

\[
j^*(x,t)=\arg\min_j \mathrm{OperationalRisk}_j(x,t)
\]

for each incoming ticket.

Plain-language interpretation:
- each arriving issue must be routed to one queue
- the best queue is not defined only by classification accuracy
- it is defined by a combination of queue congestion, delay, and misroute risk

## A2. Text Classifier

Each ticket is represented as text

\[
\mathrm{text}(x)=\mathrm{summary}(x)\; || \; \mathrm{description}(x)
\]

and converted into a sparse feature vector

\[
\phi(x)=\big[\mathrm{TFIDF}_{word}(x),\ \mathrm{TFIDF}_{char}(x)\big].
\]

For each queue \(j\), multinomial logistic regression computes a score

\[
z_j(x)=w_j^\top \phi(x)+b_j
\]

and calibrated probabilities

\[
p_j(x)=\frac{e^{z_j(x)}}{\sum_{m=1}^{M} e^{z_m(x)}}.
\]

The classifier-only decision is

\[
\hat y_{\mathrm{clf}}(x)=\arg\max_j p_j(x).
\]

Plain-language interpretation:
- the classifier gives one probability per queue
- the classifier-only baseline simply picks the largest one

## A3. Confidence and Uncertainty

Classifier uncertainty is measured via normalized entropy:

\[
H(x)= -\frac{1}{\log M}\sum_{j=1}^{M} p_j(x)\log p_j(x).
\]

We also retain the maximum class confidence

\[
p_{\max}(x)=\max_j p_j(x).
\]

Plain-language interpretation:
- low entropy means the classifier is confident
- high entropy means the ticket is ambiguous
- \(p_{\max}\) is used as a direct confidence signal and in gating logic

## A4. Complexity Score

We define a bounded complexity proxy

\[
c(x)=0.50 \cdot \mathrm{normTokenCount}(x)
+ 0.30 \cdot \mathrm{normEntropy}(x)
+ 0.20 \cdot \mathrm{normTechnicalTokenCount}(x),
\]

with \(c(x)\in[0,1]\).

Plain-language interpretation:
- longer tickets tend to be harder
- tickets with uncertain classifier outputs tend to be harder
- tickets with technical tokens such as versions, IDs, or error codes tend to be harder

## A5. Jira-Derived Service Units

Let \(T_i\) be time-to-resolution in hours for issue \(i\). With slot size \(h_{slot}\), raw service demand is

\[
s_i^{raw}=\left\lceil \frac{T_i}{h_{slot}} \right\rceil.
\]

For the main benchmark, clipped linear service units are

\[
s_i=\min\big(\max(s_i^{raw},1),S_{\max}\big).
\]

For the tail-sensitive benchmark, quantile-spread service units are

\[
r_i=\mathrm{percentileRank}(s_i^{raw}),
\]

\[
s_i=\left\lceil r_i \cdot S_{\max}\right\rceil.
\]

Plain-language interpretation:
- real Jira resolution times are turned into discrete service demand
- the main benchmark uses a clipped version
- the tail benchmark stretches the upper tail to create stronger congestion

## A6. Queue Pressure and Capacity

Normalized queue pressure is

\[
q_j(t)=\frac{Q_j(t)}{\max(\mu_j,1)}.
\]

For empirical service-demand capacity mode, capacity is

\[
\mu_j=
\max\left(
1,\ 
\mathrm{round}\left(\frac{\lambda \cdot p_j^{train}\cdot \bar s_j}{\rho}\right)
\right),
\]

where

- \(\lambda\): scenario base arrival rate
- \(p_j^{train}\): train-set queue share
- \(\bar s_j\): average service units for queue \(j\)
- \(\rho\): target utilization

Plain-language interpretation:
- a queue with more traffic and heavier tickets gets more capacity
- pressure is not just backlog; it is backlog relative to capacity

## A7. Arrival Process

Arrival counts per slot follow a Poisson process:

\[
A_t \sim \mathrm{Poisson}(\lambda_t).
\]

In bursty scenarios,

\[
\lambda_t=
\begin{cases}
\lambda_{burst}, & t \ge o \text{ and } (t-o)\bmod B = 0\\
\lambda_{base}, & \text{otherwise}
\end{cases}
\]

where:
- \(o\): burst start offset
- \(B\): burst interval

Plain-language interpretation:
- normal slots receive arrivals around a base rate
- burst slots receive many more tickets

## A8. Queue Families and Macro-Groups

To expose meaningful queue structure, the benchmark constructs queue-to-queue relations from Jira histories and component overlap.

Let:

- \(C^{trans}_{ij}\): number of direct transitions from queue \(i\) to queue \(j\)
- \(C^{overlap}_{ij}\): number of issues in queue \(i\) whose tokenized component set overlaps with queue \(j\)

The total directed relation weight is

\[
C_{ij}=C^{trans}_{ij}+C^{overlap}_{ij}.
\]

The outgoing probability from queue \(i\) to queue \(j\) is

\[
P_{ij}=\frac{C_{ij}}{\sum_k C_{ik}}.
\]

Symmetric support is

\[
S_{ij}=C_{ij}+C_{ji}.
\]

Queues are merged into the same macro-group if they have token overlap or sufficiently strong support:

\[
\mathrm{merge}(i,j)
\iff
\big(\mathrm{tokenOverlap}(i,j)>0\big)
\ \lor\
\big(S_{ij}\ge \tau_{support}\ \land\ P_{ij}\ge 0.05\big).
\]

Plain-language interpretation:
- if two queues look semantically related or historically interact a lot, we treat them as belonging to the same family
- the hierarchical policy is then allowed to route within that family

## A9. Queue Similarity and Skill Factor

Let \(e(x)\) be the embedding of ticket \(x\), and let \(c_j\) be the centroid embedding of queue \(j\), computed from the train split only.

Cosine similarity is

\[
\mathrm{sim}_j(x)=\cos(e(x),c_j).
\]

Rescaled similarity is

\[
\mathrm{sim01}_j(x)=\frac{\mathrm{sim}_j(x)+1}{2}.
\]

The queue-specific skill factor is

\[
\kappa_j(x)=\mathrm{clip}\big(1.4-0.6\cdot \mathrm{sim01}_j(x),\ \kappa_{\min},\ \kappa_{\max}\big).
\]

Plain-language interpretation:
- if a ticket is very similar to a queue centroid, that queue is treated as more suitable
- the suitability enters the delay-risk calculation through \(\kappa_j(x)\)

## A10. Learned Jira Delay Model

For each queue \(j\), we build a queue-specific feature vector

\[
\psi_j(x)=
[
c(x),\ H(x),\ p_{\max}(x),\ p_j(x),\ \mathrm{sim}_j(x),\ \kappa_j(x),\ \mathrm{priorityCode}(x),\ \mathrm{issueTypeCode}(x),\ j,\ \mathrm{queueShare}_j,\ \overline{s}_j,\ \overline{T}_j,\ \overline{\Delta q}_j,\ \mathrm{queueChangeRate}_j
].
\]

A random forest regressor predicts queue-specific delay/service demand:

\[
\hat d_j(x)=\max\big(1,\ f_{\mathrm{RF}}(\psi_j(x))\big).
\]

Plain-language interpretation:
- instead of a single service estimate per ticket, we estimate how hard the ticket may be for each candidate queue
- this is what makes the delay criterion queue-dependent in the learned-delay environment

## A11. Delay Risk

The generic backlog-scaled base delay term is

\[
d^{base}_j(x,t)=
\frac{Q_j(t)+1}{\max(\mu_j,1)}
\cdot
\big(0.5+0.5\,c(x)\big).
\]

Three delay modes are used:

Redundant baseline:

\[
d_j(x,t)=d^{base}_j(x,t).
\]

Embedding-kappa mode:

\[
d_j(x,t)=d^{base}_j(x,t)\cdot \kappa_j(x).
\]

Learned Jira delay mode:

\[
d_j(x,t)=
\frac{Q_j(t)+1}{\max(\mu_j,1)}
\cdot
\max(\hat d_j(x),1).
\]

Plain-language interpretation:
- delay risk always increases with queue congestion
- depending on the environment, it also depends on semantic fit or learned queue-specific service demand

## A12. Misroute Risk

Misroute risk is defined as

\[
r_j(x)=-\log\big(\mathrm{clip}(p_j(x),10^{-6},1)\big).
\]

Plain-language interpretation:
- if the classifier thinks queue \(j\) is likely, misroute risk is small
- if the classifier thinks queue \(j\) is very unlikely, misroute risk becomes large

## A13. Baseline Policies

### A13.1 Classifier-only

\[
j^*=\arg\max_j p_j(x).
\]

### A13.2 Join-Shortest-Queue (JSQ)

\[
j^*=\arg\min_j q_j(t).
\]

### A13.3 MaxWeight-Delay

\[
j^*=\arg\min_j d_j(x,t).
\]

### A13.4 MaxWeight-Prob

\[
U_j(x,t)=\frac{Q_j(t)+1}{\max(\mu_j,1)}+\alpha\big(1-p_j(x)\big),
\]

\[
j^*=\arg\min_j U_j(x,t).
\]

Plain-language interpretation:
- Classifier-only trusts the text model
- JSQ trusts the shortest normalized backlog
- MaxWeight-Delay trusts the smallest delay-risk estimate
- MaxWeight-Prob balances queue pressure and classifier confidence

## A14. QA-FTOPSIS Criteria

For each candidate queue \(j\), we construct a three-dimensional cost vector

\[
X_j(x,t)=
\begin{bmatrix}
q_j(t)\\
d_j(x,t)\\
r_j(x)
\end{bmatrix}.
\]

The three criteria are:
- backlog pressure
- predicted delay risk
- misroute risk

Plain-language interpretation:
- each queue is scored on congestion, expected delay, and routing plausibility

## A15. Fuzzification

Each scalar criterion value \(z\) is converted into a triangular fuzzy number

\[
\tilde z=(l,m,u)=\big(\max(0,z(1-\delta)),\ z,\ z(1+\delta)\big).
\]

Default criterion-specific fuzziness values are

\[
\delta=[0.10,\ 0.20,\ 0.15].
\]

Plain-language interpretation:
- we do not treat each criterion value as exact
- we give it a lower bound, center, and upper bound

## A16. Cost Normalization

For one cost-criterion column with triangular values \((l_a,m_a,u_a)\), define

\[
L_{\min}=\min_a l_a,\qquad
U_{\max}=\max_a u_a,\qquad
D=U_{\max}-L_{\min}.
\]

The normalized fuzzy value is

\[
\tilde r_a=
\left(
\frac{U_{\max}-u_a}{D},
\frac{U_{\max}-m_a}{D},
\frac{U_{\max}-l_a}{D}
\right).
\]

If all alternatives have the same fuzzy value, the normalized column is set to \((1,1,1)\) for all alternatives.

Plain-language interpretation:
- all three criteria are costs, so lower raw values should become better normalized scores
- equal columns are made neutral so they do not distort the ranking

## A17. Weighted Fuzzy Matrix

Let \(w_k\) be the weight of criterion \(k\). Then

\[
\tilde v_{jk}=w_k\tilde r_{jk}.
\]

Plain-language interpretation:
- after normalization, each criterion is scaled by its importance

## A18. Fuzzy Ideals

The fuzzy ideal best and worst are

\[
\tilde A_k^+=\max_j \tilde v_{jk},
\qquad
\tilde A_k^-=\min_j \tilde v_{jk}.
\]

Plain-language interpretation:
- \(\tilde A^+\) is the best achievable fuzzy point
- \(\tilde A^-\) is the worst achievable fuzzy point

## A19. Distances and Closeness

For two triangular fuzzy numbers \((l_1,m_1,u_1)\) and \((l_2,m_2,u_2)\), the vertex distance is

\[
d_v=
\sqrt{
\frac{(l_1-l_2)^2+(m_1-m_2)^2+(u_1-u_2)^2}{3}
}.
\]

For alternative \(j\), distance to a fuzzy ideal point over all criteria is

\[
D_j=
\sqrt{
\sum_k d_v(\tilde v_{jk}, \tilde A_k)^2
}.
\]

Hence

\[
D_j^+=\mathrm{dist}(\tilde V_j,\tilde A^+),
\qquad
D_j^-=\mathrm{dist}(\tilde V_j,\tilde A^-).
\]

The TOPSIS closeness coefficient is

\[
CC_j=\frac{D_j^-}{D_j^+ + D_j^-}.
\]

The QA-FTOPSIS decision is

\[
j^*=\arg\max_j CC_j.
\]

Plain-language interpretation:
- a good queue is close to the ideal best and far from the ideal worst
- the closeness coefficient summarizes that trade-off into one number

## A20. Confidence Gate, Top-K, and Hierarchy

### Confidence gate

If classifier confidence exceeds a threshold \(\tau\),

\[
\text{if } p_{\max}(x)\ge \tau,\quad
j^*=\arg\max_{j\in\mathcal{C}(x)} p_j(x).
\]

### Top-K gating

Define

\[
\mathrm{TopK}(x)=\text{the }K\text{ queues with largest }p_j(x).
\]

Then candidate search is restricted to

\[
\mathcal{C}(x)=\mathrm{TopK}(x).
\]

### Hierarchical routing

If the predicted queue belongs to macro-group \(g\), then

\[
\mathcal{C}(x)=\mathcal{H}(x)=\{j:\mathrm{macroGroup}(j)=g\}.
\]

Plain-language interpretation:
- confidence gate says “if the classifier is very sure, trust it”
- Top-K says “only rerank among plausible queues”
- hierarchy says “only rerank inside the correct queue family”

## A21. Simulation Dynamics

### Service update

If a ticket is served during a slot,

\[
s_i(t+1)=s_i(t)-1.
\]

The ticket completes when

\[
s_i(t)=0.
\]

### Waiting time

For completed ticket \(i\),

\[
W_i=\mathrm{completionSlot}_i-\mathrm{arrivalSlot}_i+1.
\]

### SLA violation

\[
\mathrm{SLA}_i=\mathbf{1}[W_i>d(\mathrm{priority}_i)].
\]

### Bounce for incompatible misroutes

If a ticket is routed to an incompatible wrong queue, it does not receive service there. It is transferred after one slot to the true queue.

Plain-language interpretation:
- waiting time includes all queueing and bounce delay
- incompatible misroutes are explicitly penalized in time

## A22. Compatible Routes

If a chosen queue is not the true queue but belongs to the compatible neighborhood of the true queue, then it is treated as a compatible route:

\[
\mathrm{compatibleRoute}_i=1
\quad\text{if}\quad
\hat y_i \in \mathcal{N}(y_i)
\ \land\
\hat y_i \neq y_i.
\]

In that case, the ticket is not counted as a hard misroute.

Plain-language interpretation:
- not every non-identical route is equally bad
- routing inside a closely related family is treated differently from routing to an unrelated queue

## A23. Evaluation Metrics

For \(N\) completed tickets and \(T\) slots:

Mean waiting time:

\[
\mathrm{MeanWait}=\frac{1}{N}\sum_{i=1}^{N} W_i
\]

95th-percentile waiting time:

\[
\mathrm{p95}=\mathrm{Percentile}_{95}(\{W_i\})
\]

99th-percentile waiting time:

\[
\mathrm{p99}=\mathrm{Percentile}_{99}(\{W_i\})
\]

Mean backlog:

\[
\mathrm{MeanBacklog}=\frac{1}{T}\sum_{t=1}^{T}\sum_j Q_j(t)
\]

Accuracy:

\[
\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat y_i=y_i]
\]

Misroute rate:

\[
\mathrm{MisrouteRate}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{misrouted}_i=1]
\]

Average cost:

\[
\mathrm{AvgCost}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
W_i + 5\cdot \mathrm{SLA}_i + 2\cdot \mathrm{Misroute}_i
\right)
\]

Macro-F1 is computed as the unweighted mean of the per-class F1 scores.

Plain-language interpretation:
- average cost is the main operational objective
- it combines delay, SLA misses, and hard misroutes
- macro-F1 ensures that improvements are not obtained by collapsing minority queues

## A24. Core Mathematical Claim of the Paper

The positive result does not come from fuzzy arithmetic alone. It comes from the composition

\[
\text{classifier probabilities}
\;+\;
\text{macro-group construction}
\;+\;
\text{queue pressure}
\;+\;
\text{learned queue-specific delay}
\;+\;
\text{fuzzy reranking within a queue family}.
\]

More concretely, the strongest positive finding is supported by:

\[
j^*(x,t)
=
\arg\max_{j\in \mathcal{H}(x)}
CC_j
\]

where the criteria behind \(CC_j\) are

\[
\big(q_j(t),\ d_j(x,t),\ r_j(x)\big)
\]

and \(d_j(x,t)\) is the learned Jira delay term.

Plain-language interpretation:
- the classifier finds the right neighborhood
- hierarchy restricts the decision to the right queue family
- fuzzy TOPSIS selects the best queue inside that family
- this is the mechanism behind the positive result in the paper
