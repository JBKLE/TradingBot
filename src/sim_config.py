"""Configuration constants for the simulation engine."""
from .sim_models import SimDirection, SlTpConfig, SlVariant

SL_TP_VARIANTS: list[SlTpConfig] = [
    SlTpConfig(variant=SlVariant.TIGHT, sl_pct=0.003, tp_pct=0.005),
    SlTpConfig(variant=SlVariant.MEDIUM, sl_pct=0.006, tp_pct=0.012),
    SlTpConfig(variant=SlVariant.WIDE, sl_pct=0.010, tp_pct=0.025),
]

DIRECTIONS: list[SimDirection] = [SimDirection.BUY, SimDirection.SELL]
