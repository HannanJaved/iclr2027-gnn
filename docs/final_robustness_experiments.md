> Note: cluster Slurm launch scripts are omitted from this anonymous > release. The CLI entrypoints below remain the source of truth for > reproducing analyses locally.

# Final robustness experiments

These jobs address three remaining sensitivity questions without changing the formal
train-loss-defined Rashomon sets.

## Submit

From the project root, submit the complete Romeo batch with:

```bash
bash scripts/ [omitted from anonymous release]/submit_final_robustness_romeo.sh
```

The helper submits independent set-size and fairness jobs and reruns the existing GCN rewiring
summary with sign tests. It also chains a clean 50-seed PubMed GAT training batch, Rashomon-set
preparation, fixed-ensemble evaluation, and summary jobs with `afterok` dependencies. It does not
submit the final release manifest.

## Outputs

- `outputs/set_size_robustness/`: 500 common-`K=10` model subsamples for 22 substantive
  Rashomon sets. Each draw reports disagreement, entropy, variation ratio, maximum pairwise TV,
  mean pairwise TV, and node-level 95th-percentile pairwise TV. The dependent summary job writes
  `set_size_robustness_summary.csv` across all sets.
- `outputs/budget_matched_fairness/`: 1,000 validation-only NBA selections after downsampling
  each retained GCN pool to five candidates, compared separately with FairGNN and FairSIN.
- `outputs/repeated_rewiring/repeated_rewiring_inference.csv`: existing GCN inference augmented
  with two-sided and directional sign tests, including BH and Bonferroni corrections.
- `outputs/canonical_pubmed_gat/`: newly trained PubMed GAT runs using the corrected canonical
  Planetoid normalization path. The preparation job constructs the train-loss set and applies the
  0.02 validation-accuracy sensitivity constraint; its retained count is determined at runtime.
- `outputs/non_gcn_rewiring/pubmed_gat/`: that validation-constrained canonical PubMed GAT set
  evaluated on the same 140 counterfactual graphs used for GCN.

The historical PubMed GAT checkpoints reproduce their archived predictions on raw cached
Planetoid features but not on the corrected normalized representation. They are therefore not used
for the rewiring extension. The canonical retraining stage is required, not optional setup.

The fairness job estimates retained-pool downsampling sensitivity. It standardizes the number of
models available to the deployment selector but does not claim equal training compute across
ordinary GNN, FairGNN, and FairSIN procedures.

After inspecting and incorporating the results, commit the code and manuscript changes, ensure
the worktree is clean, and submit `scripts/ [omitted from anonymous release]/release_manifest_romeo_cpu.sbatch`.
