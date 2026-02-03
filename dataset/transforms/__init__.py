from .VideoTransforms import VideoTransforms
from .VideoAugmentation import VideoAugmentation, VideoAugmentationPipeline

def get_transform_sequence(transform_names, **kwargs):
    """
    Given a list of transform names, return a forwardable class of corresponding transform instances.
    
    Args:
        transform_names: List of strings representing transform names.
        kwargs: Additional keyword arguments to pass to transform constructors.

    Returns:
        List of transform instances.
    """

