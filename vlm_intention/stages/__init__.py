"""Pipeline stages: filter → annotate_vlm + annotate_depth → [calibrate] → mine → balance.

Each stage exposes run(cfg=None) and is runnable standalone
(`python -m vlm_intention.stages.<name>`) or via run_pipeline.py --stage <name>.
"""
