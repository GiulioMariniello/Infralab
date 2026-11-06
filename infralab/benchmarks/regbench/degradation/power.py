# Power law of condition against effective age: C = 1 - s (A/L)^k.
# Each quarter the condition loses the increment of the curve.


def degrade(ctx):
    L = ctx["vita_utile_anni"]                      # service life, years
    if L <= 0:
        return ctx["condition"]
    r0 = max(ctx["previous_age_years"], 0.0) / L    # effective age before the quarter
    r1 = max(ctx["effective_age_years"], 0.0) / L   # and after it
    return ctx["condition"] - ctx["scale"] * (r1 ** ctx["shape"] - r0 ** ctx["shape"])
