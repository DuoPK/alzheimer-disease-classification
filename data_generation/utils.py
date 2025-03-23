def print_changed_values(original_df, df):
    changed_cells = []
    for column in df.columns:
        null_mask = original_df[column].isnull()
        changed_indices = df.index[null_mask]
        for idx in changed_indices:
            diagnosis = df.at[idx, 'Diagnosis'] if 'Diagnosis' in df.columns else 'Brak diagnozy'
            changed_cells.append((idx, column, df.at[idx, column], diagnosis))

    print("Lista imputowanych wartości:")
    for idx, column, value, diagnosis in changed_cells:
        print(f"Indeks: {idx}, Cecha: {column}, Nowa wartość: {value}, Diagnoza: {diagnosis}")
