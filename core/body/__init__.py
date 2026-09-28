"""
core.body - Layer 0 身体-神经模拟系统
"""
from .parts import (
    SubPart, BodyRegion, build_default_body,
    compute_global_arousal, compute_readiness, compute_total_wetness,
    PartnerState
)
from .neural import NeuralNode, build_neural_nodes, connect_nodes, stimulate_node, propagate
