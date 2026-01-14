import pandas as pd
import itertools

def grayish_sort(df: pd.DataFrame, factors: list):
    """
    Stable mixed-radix Gray-code–like sort over factors.
    Works on a copy of df: converts columns to ordered categoricals internally if needed.
    Returns the original df sorted by Grayish order.
    """
    # Work on a copy
    df_copy = df[factors].copy()
    
    # Convert each factor to ordered categorical (internally)
    schema = {}
    for f in factors:
        if pd.api.types.is_categorical_dtype(df_copy[f]) and df_copy[f].cat.ordered:
            cats = list(df_copy[f].cat.categories)
        else:
            cats = pd.unique(df_copy[f])
            # Sort appropriately based on dtype
            try:
                # Try to sort as numbers
                cats = sorted(cats, key=lambda x: float(x))
            except (ValueError, TypeError):
                # Fall back to string sort
                cats = sorted(cats, key=lambda x: str(x))
        
        df_copy[f] = pd.Categorical(df_copy[f], categories=cats, ordered=True)
        schema[f] = cats
    
    # Build categories and radix sizes
    categories = [schema[f] for f in factors]
    radices = [len(c) for c in categories]
    
    # Generate Gray order in index space
    gray_indices = list(mixed_radix_gray_gen(radices))
    
    # Convert index tuples → label tuples
    gray_labels = [
        tuple(categories[i][idx[i]] for i in range(len(factors)))
        for idx in gray_indices
    ]
    
    # Lookup table
    order_lookup = {combo: i for i, combo in enumerate(gray_labels)}
    
    # Assign Gray order to rows
    df_copy['_gray_order'] = df_copy[factors].apply(lambda r: order_lookup[tuple(r)], axis=1)
    
    # Sort original df using the Gray order from the copy
    sorted_df = df.iloc[df_copy['_gray_order'].argsort(kind='stable')]
    
    return sorted_df

def mixed_radix_gray_gen(levels):
    if not levels:
        yield []
        return
    first, rest = levels[0], levels[1:]
    tail = list(mixed_radix_gray_gen(rest))
    for i in range(first):
        seq = tail if i % 2 == 0 else reversed(tail)
        for code in seq:
            yield [i] + code


# ---------------- Example ----------------
if __name__ == "__main__":
    A = ['low', 'medium', 'high']
    B = ['blue', 'red']
    C = [0, 1]
    D = ['small', 'medium', 'large']

    FACTORS = ['A', 'B', 'C', 'D']

    data = list(itertools.product(A, B, C, D))
    df = pd.DataFrame(data + data[:3], columns=FACTORS)

    # Intentionally leave all columns as object/string or int
    df_sorted = grayish_sort(df, FACTORS)
    print(df_sorted)
