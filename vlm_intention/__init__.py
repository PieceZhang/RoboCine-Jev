"""VLMIntention — obstacle-direction training-data pipeline.

Stages: raw → filter → annotate (VLM 5-tag + depth) → mine + rebalance → train set.
Import stage modules explicitly: `from vlm_intention.stages import balance`.
"""
__version__ = "1.0.0"
