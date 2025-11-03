"""
File containing custom errors related strictly to the machine learning pipeline.
"""

class DataLeakageError(Exception):
    """Raised when data leakage is detected in the dataset or pipeline."""
    pass

class PipelineStateError(Exception):
    """Raised when the pipeline is a state which has not initialized the attributes necessary for running a specific method"""
    pass

class ModelStateError(Exception):
    pass