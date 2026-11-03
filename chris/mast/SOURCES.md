# Sources of the files in `chris/mast`

Every file here is a verbatim copy of a committed file in one of the two frozen trees, taken with `git show HEAD:<path>`
and checked byte for byte against the working tree (0 differences). Paths inside each folder match the paths in the source
repository, so imports, relative paths and the absolute `/fsx` paths in the code behave as they did.

| Folder | Source tree (branch) | Commit | What ran from it |
|---|---|---|---|
| `multimachine/` | `/fsx/chris.beckham/github/fec-multimachine` (`multimachine`) | `58be17785fb9fd77d7345342ee7e7d9a676d0cca` | training (`train_mm.py`), prediction (`predict.py`), synthetic cache building |
| `starter/` | `/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter` (`nn-coil-image`) | `46527a2e0f2b22be1e21f1232b23b9c1dfea7edf` | candidate build scripts, post-processing (q95 rule, βN V2), checkpoint selection during training (`synth_S/`) |

Files were chosen by following imports, `sys.path` additions and subprocess calls from the entry points below.

- `multimachine/`: `train_mm.py`, `predict.py`, `score.py`, `lindec_build.py`, `synth_cache_seq.py`, `synth_merge.py`,
  `synth_shape_targets.py`, `build_cache.py`, `build_features.py`, `build_mast_inputs.py`, `build_mast_sol_half.py`,
  `build_thomson_raw.py`.
- `starter/`: `build_candidate.sh`, `build_ensemble.sh`, `build_qb.sh`, `build_Z9.sh`, the `run_*.sh` launchers,
  `postproc_qb.py`, `splice_q95.py`, `q95_study/H_v4/apply_cc.py`, `q95_study/F_betaN/apply_bn_variant.py`,
  `synth_S/ensemble_synth.py` (with `predict_synth.py` and `postproc_synth.py`).

## File list

```
multimachine/data_fixes.py
multimachine/fusion_scoring/common.py
multimachine/fusion_scoring/contour.py
multimachine/fusion_scoring/derive.py
multimachine/fusion_scoring/lcfs.py
multimachine/fusion_scoring/masks/d3d_envelope.npz
multimachine/fusion_scoring/masks/mast_envelope.npz
multimachine/fusion_scoring/metrics.py
multimachine/fusion_scoring/o_point.py
multimachine/local_score.py
multimachine/my_experiments/data_stats.py
multimachine/my_experiments/infer_mirror_test.py
multimachine/my_experiments/nn/baseline_pictures.npz
multimachine/my_experiments/nn/baseline_pictures_ecoil135.npz
multimachine/my_experiments/nn/baseline_pictures_fitgain2.npz
multimachine/my_experiments/nn/build_cache.py
multimachine/my_experiments/nn/build_features.py
multimachine/my_experiments/nn/build_mast_inputs.py
multimachine/my_experiments/nn/build_mast_sol_half.py
multimachine/my_experiments/nn/build_thomson_raw.py
multimachine/my_experiments/nn/common.py
multimachine/my_experiments/nn/features_v2.py
multimachine/my_experiments/nn/features_v3.py
multimachine/my_experiments/nn/features_v4.py
multimachine/my_experiments/nn/features_v5.py
multimachine/my_experiments/nn/fluxfn_loss.py
multimachine/my_experiments/nn/geom_mask/mast_mask_limiter.npz
multimachine/my_experiments/nn/greens.py
multimachine/my_experiments/nn/gs_solver/gs.py
multimachine/my_experiments/nn/lindec.py
multimachine/my_experiments/nn/lindec_build.py
multimachine/my_experiments/nn/mast_monitor.py
multimachine/my_experiments/nn/mast_region.py
multimachine/my_experiments/nn/model.py
multimachine/my_experiments/nn/physics_loss.py
multimachine/my_experiments/nn/predict.py
multimachine/my_experiments/nn/score.py
multimachine/my_experiments/nn/shape_loss.py
multimachine/my_experiments/nn/shift_report.py
multimachine/my_experiments/nn/synth_cache.py
multimachine/my_experiments/nn/synth_cache_seq.py
multimachine/my_experiments/nn/synth_data.py
multimachine/my_experiments/nn/synth_merge.py
multimachine/my_experiments/nn/synth_shape_targets.py
multimachine/my_experiments/nn/train.py
multimachine/my_experiments/nn/train_mm.py
multimachine/my_experiments/nn/zoom.py
multimachine/push_predictions.py
multimachine/submission_skeleton.py
multimachine/validate_submission.py
starter/data_fixes.py
starter/fusion_scoring/common.py
starter/fusion_scoring/contour.py
starter/fusion_scoring/derive.py
starter/fusion_scoring/lcfs.py
starter/fusion_scoring/masks/d3d_envelope.npz
starter/fusion_scoring/masks/mast_envelope.npz
starter/fusion_scoring/metrics.py
starter/fusion_scoring/o_point.py
starter/local_score.py
starter/my_experiments/nn/build_Z9.sh
starter/my_experiments/nn/build_candidate.sh
starter/my_experiments/nn/build_ensemble.sh
starter/my_experiments/nn/build_qb.sh
starter/my_experiments/nn/common.py
starter/my_experiments/nn/features_v2.py
starter/my_experiments/nn/features_v3.py
starter/my_experiments/nn/features_v4.py
starter/my_experiments/nn/features_v5.py
starter/my_experiments/nn/greens.py
starter/my_experiments/nn/mast_diag/physics_suite.py
starter/my_experiments/nn/mast_region.py
starter/my_experiments/nn/model.py
starter/my_experiments/nn/physics_loss.py
starter/my_experiments/nn/postproc_qb.py
starter/my_experiments/nn/q95_study/F_betaN/apply_bn_variant.py
starter/my_experiments/nn/q95_study/F_betaN/bn_lib.py
starter/my_experiments/nn/q95_study/F_betaN/out/dd_constants.json
starter/my_experiments/nn/q95_study/H_v4/apply_cc.py
starter/my_experiments/nn/q95_study/H_v4/qlib.py
starter/my_experiments/nn/run_cond_train.sh
starter/my_experiments/nn/run_overnight_1016.sh
starter/my_experiments/nn/run_tune3_train.sh
starter/my_experiments/nn/run_v6_extra.sh
starter/my_experiments/nn/run_v7.sh
starter/my_experiments/nn/shift_report.py
starter/my_experiments/nn/splice_q95.py
starter/my_experiments/nn/synth_S/ensemble_synth.py
starter/my_experiments/nn/synth_S/extra_sets.json
starter/my_experiments/nn/synth_S/paths.py
starter/my_experiments/nn/synth_S/postproc_synth.py
starter/my_experiments/nn/synth_S/predict_synth.py
starter/my_experiments/nn/synth_S/run_all.py
starter/my_experiments/nn/synth_S/scorer.py
starter/my_experiments/nn/upload_candidate.sh
starter/push_predictions.py
starter/submission_skeleton.py
starter/validate_submission.py
```
