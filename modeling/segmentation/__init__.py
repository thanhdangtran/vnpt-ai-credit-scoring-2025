"""
Segmentation module for credit scoring.

This module provides customer segmentation tools for credit risk modeling,
including CHAID (Chi-square Automatic Interaction Detection) algorithm.

Segmentation is used to:
- Identify distinct customer groups with different risk profiles
- Build segment-specific scorecards
- Improve model performance through population stratification
"""

from .chaid_segmenter import (
    # Dataclasses
    CHAIDNode,
    SegmentProfile,
    # Classes
    CHAIDSegmenter,
    VietnameseCreditSegmenter,
    # Constants
    DEFAULT_ALPHA,
    VIETNAMESE_SEGMENT_NAMES,
)

__all__ = [
    # Dataclasses
    "CHAIDNode",
    "SegmentProfile",
    # Classes
    "CHAIDSegmenter",
    "VietnameseCreditSegmenter",
    # Constants
    "DEFAULT_ALPHA",
    "VIETNAMESE_SEGMENT_NAMES",
]
