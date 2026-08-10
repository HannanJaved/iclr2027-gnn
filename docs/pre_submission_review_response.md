# Pre-Submission Review Response

> Historical review snapshot. Its rewiring and explanation-inference statements
> are superseded by the normalization-corrected, repeated fixed-ensemble analysis
> documented in `README.md` and the current manuscript.

This note summarizes the main risks raised in the external review and the
current response before launching additional jobs.

## Highest Priority

- The original `Multiplicity -> Explanation Instability` framing is too strong.
  Existing controlled regressions show no independent capacity effect for Cora
  GCN, a positive capacity-to-Jaccard association for Cora GAT, and no direct
  capacity effect for Cora APPNP, PubMed GAT, or PubMed APPNP. The paper draft
  should frame XAI as a negative/boundary-condition result: in the current
  bounded experiments, multiplicity is not supported as a direct driver of
  explanation instability.
- Do not submit more explanation jobs until the existing controlled-regression
  result is reported plainly. More XAI compute is useful only after the claim is
  reframed.

## Methodological Caveats To Keep Visible

- Relative-2.0 Rashomon sets are sensitivity conditions, not silently pooled
  with stricter relative-0.1 seed sets. Final reporting should include absolute
  loss and validation-accuracy ranges for every tolerance condition.
- Small retained sets need explicit caveats. Cora GraphSAGE retained only five
  models; the strongest CiteSeer homophily-targeted rewiring retained only one.
  Any frontier or multiplicity summary with fewer than ten retained models
  should be labeled diagnostic.
- The current fairness-dispersion evidence is descriptive. The model-selection
  Pareto result is strong, but the group-concentration-to-dispersion claim needs
  a formal test before it can be stated as a predictive result.
- Local-entanglement rewiring is intervention-backed association, not a fully
  general causal proof. This is still strong evidence, but the causal language
  should remain careful.
- Reported p-values are currently uncorrected. Final tables should add
  confidence intervals and either FDR/Bonferroni correction or an explicit note
  defining exploratory test families.
- The topology-only ablation is a degeneracy diagnostic. Universal disagreement
  means the topology-only objective is weakly constrained, not that topology
  alone explains labels.

## Framing Updates Already Made

- The abstract now says Rashomon-aware selection improves fairness tradeoffs,
  while explanation stability is not independently explained by multiplicity
  after degree and confidence controls.
- The pipeline language now says analysis pipeline rather than causal pipeline.
- Step 4 now asks whether and when multiplicity implies explanation instability
  and names degree, confidence, and correctness as controls.
- The XAI contribution is now framed as boundary conditions for XAI
  trustworthiness.
- The fairness section now explicitly states the policy-relevant implication:
  aggregate fairness metrics can mask localized arbitrariness.
- The abstract and Step 5 now separate the tested fairness Pareto result from
  the still-descriptive fairness-dispersion/concentration evidence.
- The Cora GraphSAGE architecture-validation cell is explicitly marked
  diagnostic because it retained fewer than ten models.

## Before New Jobs

1. Generate a tolerance-sensitivity summary from existing runs if possible.
2. Add confidence intervals and multiple-comparison notes for the current
   structural and intervention results.
3. Add or report a formal fairness concentration-to-dispersion test if that
   claim remains in the paper.
4. Only then decide whether new jobs should target Amazon-Photo, PGExplainer,
   FairGNN-style baselines, or tolerance sweeps.
