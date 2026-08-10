# Stage-1 Gap Review

> Historical gap snapshot. Its rewiring status predates the Planetoid
> normalization correction and repeated fixed-ensemble counterfactual analysis;
> use `README.md` and the current manuscript for active claims.

This note records the current status of the stage-1 GNN Rashomon pipeline so
the paper text, code, and remaining experiments stay aligned.

## Complete

- GCN seed and hyperparameter Rashomon construction for Cora, CiteSeer, PubMed,
  Adult, and German Credit.
- Node-level multiplicity metrics, structural diagnostics, citation-graph
  observational correlations, feature-only MLP ablations, and topology-only
  ablations.
- Degree-preserving random rewiring, homophily-targeted rewiring, and local
  entanglement treatment/control analysis on the citation graphs.
- Bounded GNNExplainer analysis and explanation-stability Pareto selection for
  Cora GCN.
- Fairness dispersion and fairness Pareto selection for Adult and German
  Credit.
- Architecture-validation seed Rashomon sets for GAT, GraphSAGE, and APPNP on
  Cora and PubMed, plus bounded explanation/Pareto checks for GAT and APPNP.
- Amazon-Photo dataset loading/configuration and Slurm smoke/seed entrypoints
  for stage-2 generalization.
- Reproducibility manifest generation for final artifacts, package versions,
  git state, and Slurm logs.

## Partial Or Diagnostic

- CiteSeer global rewiring is partially rescued by the broader relative-2.0
  tolerance, but the strongest homophily target still retains only one model and
  should be treated as diagnostic rather than paper-grade evidence.
- Cora GraphSAGE architecture validation retained only five models under the
  broader tolerance, so it should be interpreted cautiously.
- Explanation instability is not a simple monotone consequence of predictive
  multiplicity in the completed bounded experiments; degree, confidence, and
  architecture should be treated as controls or mediators in the write-up.
- Pokec is deferred because the required raw files and sensitive-attribute
  documentation are not present in the workspace. COMPAS recidivism was explored
  but is not used in the first paper evidence because it requires preprocessing
  choices that would distract from the methodological comparison.

## Remaining Gaps

- Stage-2 Amazon-Photo experiments still need to be run and summarized; NBA
  remains a dataset-ingestion task because no raw file/schema is present in the
  workspace yet.
- FairGNN or another fairness-intervention baseline for comparison with
  Rashomon-aware selection.
- PGExplainer and GAT-attention explanation variants from the original proposal.
- Full wall-clock/GPU-CPU accounting for each final table, beyond the current
  manifest's git/package/artifact/log hashes.
- Final paper-quality figure generation and a single command that rebuilds the
  exact tables used in `../rashomon.tex`.
