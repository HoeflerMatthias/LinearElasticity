import itertools


def grid_product(grid_dict):
    """Expand a dict of lists into a list of param dicts."""
    keys = list(grid_dict.keys())
    for vals in itertools.product(*(grid_dict[k] for k in keys)):
        yield dict(zip(keys, vals))