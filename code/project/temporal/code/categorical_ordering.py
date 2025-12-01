import pandas as pd

# # Define the levels for each category
# A = ['low', 'medium', 'large']  # Most significant
# B = ['blue', 'red']
# C = ['small', 'medium', 'large']  # Least significant

# data = list()
# for i in range(len(A)):
#     for j in range(len(B)):
#         for k in range(len(C)):
#             data.append((A[i], B[j], C[k]))
#         C.reverse()
#     B.reverse()

# df = pd.DataFrame(data)
# print(df)

def generate_grayish_combinations_from_df_recursive(
    categories_df, index=0, current_combination=None, data=None
):
    # gray as in gray code
    if current_combination is None:
        current_combination = []
    if data is None:
        data = []
    
    if index < len(categories_df.columns):
        category_values = categories_df.iloc[:, index].unique()
        for item in category_values:
            new_combination = current_combination + [item]
            generate_grayish_combinations_from_df_recursive(
                categories_df, index + 1, new_combination, data
            )
        categories_df.iloc[:, index] = categories_df.iloc[:, index][::-1]
    else:
        data.append(tuple(current_combination))
    
    return data


def grayish_sort(combo_series):
    # Create a mapping from each unique combo to all its original indices
    combo_to_indices = {}
    for idx, combo in enumerate(combo_series):
        combo_tuple = tuple(combo)
        if combo_tuple not in combo_to_indices:
            combo_to_indices[combo_tuple] = []
        combo_to_indices[combo_tuple].append(idx)
    
    # Create deduplicated series for processing
    unique_combos = pd.Series(list(combo_to_indices.keys()))
    
    # Split the unique combos into individual columns
    df = unique_combos.apply(pd.Series)
    
    # Generate combinations based on individual columns
    data = generate_grayish_combinations_from_df_recursive(df)
    
    # Create a mapping from combinations to their position in unique_combos
    combo_to_unique_idx = {tuple(combo): idx for idx, combo in enumerate(unique_combos)}
    
    # Get the sorted order of unique combinations
    sorted_unique_indices = [combo_to_unique_idx[combo] for combo in data]
    
    # Expand to include all original indices (including duplicates)
    sorted_indices = []
    for unique_idx in sorted_unique_indices:
        unique_combo = unique_combos[unique_idx]
        sorted_indices.extend(combo_to_indices[tuple(unique_combo)])
    
    return sorted_indices


if __name__ == '__main__':
    # Define the levels for each category
    A = ['low', 'medium', 'large']  # Most significant
    B = ['blue', 'red']
    C = ['small', 'medium', 'large']  # Least significant
    
    # Generate Cartesian product to form input DataFrame with one 'combo' column
    data = []
    for a in A:
        for b in B:
            for c in C:
                data.append([(a, b, c)])
    
    df = pd.DataFrame(data, columns=['combo'])
    
if __name__ == '__main__':
    # Define the levels for each category
    A = ['low', 'medium', 'large']  # Most significant
    B = ['blue', 'red']
    C = ['small', 'medium', 'large']  # Least significant
    
    # Generate Cartesian product to form input DataFrame with one 'combo' column
    data = []
    for a in A:
        for b in B:
            for c in C:
                data.append([(a, b, c)])
    
if __name__ == '__main__':
    # Define the levels for each category
    A = ['low', 'medium', 'large']  # Most significant
    B = ['blue', 'red']
    C = ['small', 'medium', 'large']  # Least significant
    
    # Generate Cartesian product to form input DataFrame with one 'combo' column
    data = []
    for a in A:
        for b in B:
            for c in C:
                data.append([(a, b, c)])
    
    # Create df with some duplicates to simulate real scenario
    df_with_duplicates = pd.DataFrame(data + data[:5], columns=['combo'])  # Add some duplicates
    print(f"Original df with duplicates has {len(df_with_duplicates)} rows")
    
    # Apply the grayish sort function directly to the original df with duplicates
    sorted_indices = grayish_sort(df_with_duplicates['combo'])
    
    # Verify by showing the reordered dataframe
    df_sorted = df_with_duplicates.iloc[sorted_indices].reset_index(drop=True)
    print("\nFirst 10 rows of sorted df:")
    print(df_sorted.head(10))