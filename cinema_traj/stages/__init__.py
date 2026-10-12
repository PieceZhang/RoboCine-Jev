"""cinema_traj.stages — pipeline stages, each dual-runnable.

Contract (mirrors vlm_intention.stages): every stage module exposes
``run(cfg=None, **stage_flags)`` and has a ``__main__`` argparse block, so it
works both via ``python run_traj.py --stage <name>`` and
``python -m cinema_traj.stages.<name>``.
"""
