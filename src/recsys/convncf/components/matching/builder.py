from .matching import ConvoluationalCollaborativeFilteringLayer


def matching_fn_builder(**kwargs):
    cls = ConvoluationalCollaborativeFilteringLayer
    return cls(**kwargs)