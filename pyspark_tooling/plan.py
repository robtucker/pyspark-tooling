from pyspark.sql import DataFrame


def to_plan(df: DataFrame):
    """Retrieve the plan as a dictionary"""
    return parse_plan(df._jdf.queryExecution().toString())


def parse_plan(plan: str):
    plan_delimiters = [
        ("physical", "== Physical Plan =="),
        ("optimized", "== Optimized Logical Plan =="),
        ("analyzed", "== Analyzed Logical Plan =="),
        ("logical", "== Parsed Logical Plan =="),
    ]

    current = plan
    res = {}
    for i in plan_delimiters:
        if not current:
            return res
        parts = current.split(i[1])
        if len(parts) != 2:
            # error occured - return an empty object
            return res
        elements = parts[1].split("+-")
        res[i[0]] = [e.strip() for e in elements]
        current = parts[0]
    return res
