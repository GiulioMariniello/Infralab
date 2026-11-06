# Markov chain on the 1-9 ratings of the US National Bridge Inventory.
# P: yearly probability that the rating of an item drops, by starting rating.
# D: mean size of the drop, in ratings.
# Estimated on Iowa, Ohio, Pennsylvania and Texas, 2015-2023, 2.3 million item-year pairs.
P = {9: 0.1483, 8: 0.0672, 7: 0.0334, 6: 0.0321, 5: 0.0355, 4: 0.0421, 3: 0.0428, 2: 0.0852, 1: 0.0193}
D = {9: 1.12, 8: 1.06, 7: 1.08, 6: 1.14, 5: 1.27, 4: 1.35, 3: 2.22, 2: 1.80, 1: 1.00}


def degrade(ctx):
    c = ctx["condition"]
    if ctx["age_increment_years"] <= 0:          # clock stopped (works under way): no degradation
        return c
    r = min(9, max(1, round(9 * c)))             # rating 1-9 of a condition between 0 and 1
    if ctx["rand"]() < 1 - (1 - P[r]) ** 0.25:   # quarterly probability from the yearly one
        c -= D[r] / 9
    return c
