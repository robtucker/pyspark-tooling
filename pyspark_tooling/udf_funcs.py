def set_intersection(a: list, b: list):
    """Calculate the shared elements of two array columns"""

    # if neither of the functions are lists then the result
    # should be null rather than 0. This is because 0 implies
    # a known score whereas null implies an unknown score
    if (not isinstance(a, list)) or (not isinstance(b, list)):
        return None
    set_a = set(a)
    set_b = set(b)
    # ensure nulls and empty strings are not counted as shared members
    set_a.discard("")
    set_a.discard(None)
    set_b.discard("")
    set_b.discard(None)

    # valid intersection can only be calculated if both sets have members
    # if one of the sets has no members then the intersection is null
    if len(set_a) == 0 or len(set_b) == 0:
        return None

    return list(set_a.intersection(set_b))


def set_intersection_over_union(a: list, b: list):
    """Calculate the intersection and union of two array columns as a single step
    This is to prevent converting the data to and from java more than once"""

    set_a = set()
    set_b = set()

    if isinstance(a, list):
        set_a.update(a)

    if isinstance(b, list):
        set_b.update(b)

    set_a.discard("")
    set_a.discard(None)
    set_b.discard("")
    set_b.discard(None)

    # if either of the sets has no members
    # then return null rather than a score
    if len(set_a) == 0 or len(set_b) == 0:
        return None

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    return intersection / union
