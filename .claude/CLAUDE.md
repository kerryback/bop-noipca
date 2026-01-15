# Claude Code Guidelines for This Project

## Platform: Windows

This project runs on **Windows**, which has specific requirements and limitations that must be considered when writing code.

## 🚨 CRITICAL: NO PLACEHOLDERS OR SIMPLIFIED IMPLEMENTATIONS 🚨

**ABSOLUTE REQUIREMENT - THIS IS NON-NEGOTIABLE:**

When implementing any function, method, or feature:

### ❌ NEVER DO THIS:
- Return `np.nan` for values that should be computed
- Use "simplified" or "placeholder" implementations
- Add comments like "TODO", "FIXME", "Would need X to implement properly"
- Implement partial functionality without full statistics
- Create functions that silently return incomplete results

### ✅ ALWAYS DO THIS INSTEAD:
1. **Ask first**: "I notice this function requires X, Y, Z to implement properly. Should I:
   - Implement it fully (requires adding parameters A, B, C)
   - Not implement this feature at all
   - Implement with explicit limitations documented"

2. **Implement fully**: If implementing, do it completely or not at all
   - All statistics must be computed properly
   - No `np.nan` placeholders for computable values
   - All required data must be available or requested

3. **Raise errors for incomplete functionality**:
   ```python
   # If something can't be implemented properly
   raise NotImplementedError("Feature X requires Y which is not available")
   ```

4. **Document limitations explicitly**: If there's a valid reason for partial implementation, document it clearly in:
   - Function docstring with **WARNING** section
   - Return value documentation
   - User-facing documentation

### Examples of Unacceptable Code:

❌ **WRONG:**
```python
def compute_stats(...):
    # Would need loadings to compute this properly
    return {
        'stdev': np.nan,  # TODO: implement
        'mean': simplified_value,  # Simplified
        'hjd': np.nan  # Would need full data
    }
```

✅ **CORRECT Option 1 - Ask First:**
```python
# Before implementing, ask:
# "This requires ipca_weights to compute stdev and hjd properly.
#  Should I add ipca_weights as a parameter and implement fully?"
```

✅ **CORRECT Option 2 - Raise Error:**
```python
def compute_stats(...):
    raise NotImplementedError(
        "compute_ipca_portfolio_stats requires ipca_weights parameter "
        "to compute stdev and hjd. Currently not implemented."
    )
```

✅ **CORRECT Option 3 - Explicit Documentation:**
```python
def compute_stats(...):
    """
    WARNING: This function only computes mean and xret.
    stdev and hjd are NOT computed and will be np.nan.
    This is because ipca_weights are not available.

    To get full statistics, use compute_full_stats() instead.
    """
    # Explicit about limitations
    return {...}
```

### Why This Matters:

Placeholder implementations are **dangerous** because:
1. Users assume `np.nan` means missing data, not "we didn't implement this"
2. Downstream code may silently fail or produce wrong results
3. It's unclear whether the function is broken or incomplete
4. It creates technical debt that never gets fixed
5. It violates the principle of least surprise

### When You See Existing Placeholders:

If you encounter code with placeholders or simplified implementations:
1. **Flag it immediately** to the user
2. **Do not perpetuate the pattern** in new code
3. **Offer to fix it properly** if asked to work on that code

## 🚨 CRITICAL: FILE EDITING PROTOCOL 🚨

**ABSOLUTE REQUIREMENT - THIS IS NON-NEGOTIABLE:**

When editing any file in this project:

### Mandatory Process (Silent Verification):

**The user has given blanket approval for the backup/diff/cleanup process to happen automatically and silently.**

1. **Create backup first** (silently): Before ANY edit, create a backup of the original file
   - Backup naming: `{filename}.backup` or `{filename}.bak`
   - Do NOT announce backup creation to the user

2. **Make the requested edit**: Perform only the specific changes requested
   - Do not fix unrelated issues
   - Do not refactor surrounding code
   - Do not apply "improvements" without approval

3. **Compare and verify** (silently): After editing, run diff to compare the new file to the backup
   - Do NOT show the diff output to the user
   - Verify ONLY the requested changes were made
   - **ONLY if unexpected changes are found**: Alert the user and ask for approval

4. **Cleanup** (silently): After verification succeeds, delete the backup file
   - Do NOT announce cleanup to the user

### When to Alert the User:

- ✅ Proceed silently: If only the expected changes are present
- ⚠️ Alert immediately: If ANY unexpected changes are detected
- ⚠️ Alert immediately: If other issues are noticed that should be fixed

### Process Example:

✅ **CORRECT Workflow (Silent Success):**
```
1. User requests: "Change variable name from 'x' to 'data'"
2. Silently create: file.py.backup
3. Edit file.py: Change only 'x' to 'data'
4. Silently run: diff file.py file.py.backup
5. Verify: Only 'x' → 'data' changes present
6. Silently delete: file.py.backup
7. Report to user: "Updated variable name from 'x' to 'data'" (no mention of backup/diff/cleanup)
```

✅ **CORRECT Workflow (Alert on Unexpected Changes):**
```
1. User requests: "Change variable name from 'x' to 'data'"
2. Silently create: file.py.backup
3. Edit file.py: Change 'x' to 'data', but also accidentally changed formatting
4. Silently run: diff file.py file.py.backup
5. Detect: Unexpected formatting changes
6. ALERT USER: "I found unexpected changes beyond the requested edit: [show diff]. Should I proceed?"
7. Keep backup until user responds
```

❌ **WRONG Workflow:**
```
1. User requests: "Change variable name from 'x' to 'data'"
2. Report: "Creating backup file.py.backup..."
3. Edit file.py
4. Report: "Running diff..."
5. Report: "Diff output: ..."
6. Report: "Deleting backup..."
7. Report: "Done!"
(TOO VERBOSE - user doesn't need to see the verification process)
```

### Why This Matters:

Unauthorized changes are **dangerous** because:
1. User loses control over what's being modified
2. Unintended changes can introduce bugs
3. Makes code review difficult
4. Breaks trust in the editing process
5. Can conflict with intentional design decisions

## Critical Windows Issues

### 1. Console Encoding (cp1252)

**NEVER use Unicode box-drawing or special characters in print statements.**

❌ **DO NOT USE:**
```python
print(f"  {'─'*50}")  # U+2500 box drawing character
print(f"  {'━'*50}")  # U+2501 heavy box drawing
print(f"  {'•'*50}")  # U+2022 bullet
print(f"  {'…'}")      # U+2026 horizontal ellipsis
```

✅ **ALWAYS USE:**
```python
print(f"  {'-'*50}")  # ASCII hyphen-minus
print(f"  {'='*50}")  # ASCII equals
print(f"  {'*'*50}")  # ASCII asterisk
print(f"  {'...'}")   # ASCII periods
```

**Reason:** Windows console uses cp1252 encoding by default, which cannot encode Unicode characters beyond the basic ASCII + Latin-1 range. This causes `UnicodeEncodeError` exceptions.

### 2. File Paths

- Always use `os.path.join()` or `pathlib.Path` for cross-platform compatibility
- Windows uses backslashes `\` but Python accepts forward slashes `/`
- Use raw strings `r"path\to\file"` or escape backslashes `"path\\to\\file"`

### 3. Line Endings

- Windows uses CRLF (`\r\n`) line endings
- Git may show warnings about LF being replaced with CRLF - this is normal
- Use `.gitattributes` to control line ending behavior if needed

### 4. Shell Commands

- When using `subprocess` or `Bash` tool:
  - Quote paths with spaces: `"C:\Program Files\..."`
  - Use `&&` to chain commands, not `;` or newlines
  - Be aware that some Unix commands don't exist on Windows

## Project-Specific Notes

### Parallel Processing

- This project uses `joblib.Parallel` with `n_jobs=10` for parallel processing
- The SDF compute modules (BGN, KP14, GS21) do NOT use nested parallelization
- `calculate_moments.py` processes in chunks to manage memory

### File Naming Convention

- Arrays: `{model}_{index}_arrays.pkl`
- Moments: `{model}_{index}_moments.pkl`
- Factors: `{model}_{index}_fama.pkl`, `{model}_{index}_dkkm_{nfeatures}.pkl`

### Performance Characteristics

- Later chunks in `calculate_moments.py` take longer due to increasing historical data
- This is expected behavior - computation at month `t` requires all data from `0` to `t`

## Common Errors to Avoid

1. **UnicodeEncodeError** - Use only ASCII characters in print statements
2. **Path errors** - Always use `os.path.join()` or forward slashes
3. **Encoding errors in file I/O** - Specify `encoding='utf-8'` when writing text files
4. **Line ending warnings** - Normal on Windows, can be ignored

## Testing on Windows

When making changes:
- Test all print statements work in Windows console
- Verify file paths work with Windows backslashes
- Check that subprocess commands work in Windows environment
