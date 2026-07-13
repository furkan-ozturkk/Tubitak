"""Implements the Fixed-14B and Fixed-32B baselines.

Both baselines bypass the Level-2 router and route every query to a single
fixed-size generator, isolating the effect of dynamic model-size selection.

Reference:
    LogRouter paper, Section IV-C (Baselines and Ablation Conditions) and
    Table IV.
"""
