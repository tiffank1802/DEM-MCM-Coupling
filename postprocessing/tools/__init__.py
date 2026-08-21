"""One-off maintenance scripts of the post-processing folder.

These scripts were used to build, fix or move the repository data and are
kept for reference. They are **not** part of the regular post-processing
pipeline:

* :mod:`postprocessing.tools.calibrage` — calibration of the Markov time
  step (tau) against the DEM reference;
* :mod:`postprocessing.tools.create_hf_dir` — creation of a folder in the
  bucket with the particle states;
* :mod:`postprocessing.tools.directory` — ad-hoc directory exploration;
* :mod:`postprocessing.tools.fix_ts`, ``fix_test_annotations``,
  ``fix_test_annotations2``, ``fix_test_session_sync`` — one-shot
  code-fixing scripts (legacy);
* :mod:`postprocessing.tools.mv_recursive` — one-shot bucket reorganisation.
"""
