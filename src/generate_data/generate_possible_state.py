from itertools import product
import pandas as pd
import os

# ============================================
# 1️⃣ CREATE THE COMPLETE LIST OF VALID PREFIXES
# ============================================

NUM_ACTIVITIES = 21
all_valid_prefixes = []

# --- (1) The first 6 sequential prefixes (0–4) ---
def create_initial_6_prefixes(num_activities):
    """Generate 6 initial prefixes by sequentially enabling indices 0 to 4."""
    initial_prefixes = []
    prefix = [0] * num_activities
    initial_prefixes.append(prefix.copy())
    
    for i in range(5):
        prefix[i] = 1
        initial_prefixes.append(prefix.copy())
    
    return initial_prefixes

initial_6_prefixes = create_initial_6_prefixes(NUM_ACTIVITIES)
all_valid_prefixes.extend(initial_6_prefixes)


# --- (2) 31 variable prefixes (varying indices 5–9) ---
def create_31_variable_prefixes(num_activities):
    """Generate 2^5 - 1 = 31 prefixes varying indices 5–9."""
    fixed_prefix = [1] * 5
    fixed_suffix = [0] * (num_activities - 10)
    
    variations = product([0, 1], repeat=5)
    zero_variation = (0, 0, 0, 0, 0)
    
    variable_prefixes = []
    for var in variations:
        if var != zero_variation:
            new_prefix = fixed_prefix + list(var) + fixed_suffix
            variable_prefixes.append(new_prefix)
    return variable_prefixes

variable_31_prefixes = create_31_variable_prefixes(NUM_ACTIVITIES)
all_valid_prefixes.extend(variable_31_prefixes)


# --- (3) 1023 complex prefixes (varying indices 10–19) ---
def create_1023_complex_prefixes(num_activities):
    """Generate 2^10 - 1 = 1023 prefixes varying indices 10–19."""
    fixed_prefix = [1] * 10
    fixed_suffix = [0]
    
    variations = product([0, 1], repeat=10)
    zero_variation = tuple([0] * 10)
    
    complex_prefixes = []
    for var in variations:
        if var != zero_variation:
            new_prefix = fixed_prefix + list(var) + fixed_suffix
            complex_prefixes.append(new_prefix)
    return complex_prefixes

complex_1023_prefixes = create_1023_complex_prefixes(NUM_ACTIVITIES)
all_valid_prefixes.extend(complex_1023_prefixes)

print(f"✅ Created {len(all_valid_prefixes)} valid prefixes (expected 1060).")


# ============================================
# 2️⃣ EXTEND WITH GENDER & MARITAL STATUS RULES
# ============================================

valid_sequences = []  # [Gender, Marital_Status, prefix...]

for gender, marital_status in product([0, 1], repeat=2):
    adjusted_prefixes = []

    for prefix in all_valid_prefixes:
        current_prefix = prefix[:]

        # index 8  -> Gynecological Examination
        # index 9  -> Breast Examination
        
        if gender == 0 and marital_status == 0:
            # Female & Single → No gynecological exam
            current_prefix[8] = 0

        elif gender == 1:
            # Male → No gynecological exam and no breast exam
            current_prefix[8] = 0
            current_prefix[9] = 0

        # (Female & Married) → keep prefix unchanged
        adjusted_prefixes.append(current_prefix)

    # Remove duplicate prefixes after adjustment
    unique_adjusted_prefixes = set(tuple(p) for p in adjusted_prefixes)
    
    # Combine Gender + Marital_Status + Prefix into final list
    for prefix_tuple in unique_adjusted_prefixes:
        sequence = [gender, marital_status] + list(prefix_tuple)
        valid_sequences.append(sequence)

print("✅ Expanded valid_sequences with gender & marital status.")
print(f"Total sequences generated: {len(valid_sequences)}")


# ============================================
# 3️⃣ CONSISTENCY CHECKS & SAMPLE OUTPUT
# ============================================

# Assertions to verify rule consistency
assert all(s[10] == 0 for s in valid_sequences if s[0] == 1), "Male must have Gyne=0"
assert all(s[11] == 0 for s in valid_sequences if s[0] == 1), "Male must have Breast=0"
assert all(s[10] == 0 for s in valid_sequences if s[0] == 0 and s[1] == 0), "Single female must have Gyne=0"

# Print sample sequences
print("\n--- Sample 3 sequences ---")
for s in valid_sequences[:3]:
    print(s)

print("\n--- Format: [Gender, Marital, Index8(Gyne), Index9(Breast)] ---")
male_sequence = next(s for s in valid_sequences if s[0] == 1)
print(f"Male sample: {[male_sequence[0], male_sequence[1], male_sequence[10], male_sequence[11]]}")


# ============================================
# 4️⃣ STATISTICS & SAVE RESULTS
# ============================================

count_female_single = sum(1 for s in valid_sequences if s[0] == 0 and s[1] == 0)
count_female_married = sum(1 for s in valid_sequences if s[0] == 0 and s[1] == 1)
count_male = sum(1 for s in valid_sequences if s[0] == 1)

print("\n--- Statistics ---")
print(f"Female single: {count_female_single}")
print(f"Female married: {count_female_married}")
print(f"Male (both marital states): {count_male}")

# Save all sequences to CSV for later use
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, '../../data/raw_data/')
file_path = os.path.join(output_dir, "possible_state.csv")

df = pd.DataFrame(valid_sequences)
df.to_csv(file_path, index=False, header=False)
print("\n✅ Saved possible_state.csv successfully.")

# ============================================
# 5️⃣ CREATE END STATES (CONCLUSION = 1)
# ============================================

print("\n--- Generating end states ---")

# The 11 last activities (indices 10–20 in prefix part)
# Since gender and marital status take positions 0 and 1, 
# the 11 last prefix activities correspond to columns 11 → 21 (inclusive) in the DataFrame.
activity_cols = list(range(12, 23))  # total 11 columns

# Filter rows where the last 11 prefix bits = [1,...,1,0]
def is_end_state(row):
    """Return True if last 10 prefix activities = 1 and the last (Conclusion) = 0."""
    last_11 = row[activity_cols].tolist()
    return last_11[:-1] == [1]*10 and last_11[-1] == 0

# Apply the filter
df_end = df[df.apply(is_end_state, axis=1)].copy()

# Set the last column (Conclusion) = 1
df_end.iloc[:, -1] = 1

# Remove duplicates (if any)
df_end = df_end.drop_duplicates()

print(f"✅ Found {len(df_end)} end states.")

# Save to CSV file
end_file_path = os.path.join(output_dir, "end_state.csv")
df_end.to_csv(end_file_path, index=False, header=False)

print(f"✅ Saved end_state.csv successfully at {end_file_path}")
