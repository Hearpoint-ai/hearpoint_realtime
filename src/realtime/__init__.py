"""
Real-time TFGridNet inference module for target speech extraction.

This module provides utilities for running the TFGridNet model in real-time,
capturing audio from a microphone and outputting enhanced audio to headphones.
"""

from .realtime_inference import RealtimeInference, FileBasedTest

__all__ = ["RealtimeInference", "FileBasedTest"]
