# Sample Files Explanation

This document explains why each sample file behaves the way it does when processed by the HDR Proposal Verification Assistant.

## Overview

The verification system uses two AI models (Transformer and TF-IDF) to detect four types of compliance issues:
1. **Crosswalk errors** - Requirement citations in wrong document sections
2. **Banned phrases** - Prohibited language in proposals
3. **Name inconsistency** - Name variations across the document
4. **Date inconsistency** - Illogical or conflicting dates

---

## 1. Clean Proposal (scratch_clean.txt / .pdf)

### Expected Behavior
**No errors detected** - All compliance checks pass

### Why It's Clean
This sample represents a properly formatted, compliant HDR proposal with:
- Consistent professional names (no abbreviations)
- Logical dates (all project dates flow chronologically)
- No banned/prohibited language
- No requirement cross-reference errors

### Key Characteristics
- **Names**: "Sarah Martinez", "David Chen", "Jennifer Thompson", "Michael Rodriguez" - all appear consistently
- **Dates**: Project duration April 15, 2024 to March 30, 2026 with logical milestone progression
- **Language**: Professional phrasing without guarantees or absolute claims
- **Structure**: Standard proposal sections with no requirement citations

### Test Results
```
✅ Crosswalk Error:      PASS (confidence: 0.000)
✅ Banned Phrases:       PASS (confidence: 0.010)
✅ Name Inconsistency:   PASS (confidence: 0.009)
✅ Date Inconsistency:   PASS (confidence: 0.000)
```

---

## 2. Crosswalk Error (scratch_crosswalk.txt / .pdf)

### Expected Behavior
**Flags ONLY crosswalk_error** - Detects requirement citation mismatch

### What Triggers the Error

**Line 8 (Executive Summary):**
```
Requirement R7 documentation is noted in Schedule.
```

**Line 26 (Team Qualifications):**
```
Requirement R7 response is detailed in Qualifications.
```

### Why This Is an Error
The same requirement (R7) is cited in **two different document sections**:
- First claimed to be in the **Schedule** section
- Later claimed to be in the **Qualifications** section

This inconsistency suggests:
1. Confusion about document organization
2. Copy-paste errors from templates
3. Potential missing or misplaced content

### How Models Detect It
- **Transformer**: Learned pattern recognition of "Requirement R[number] ... in [Section A]" followed by "Requirement R[same number] ... in [Section B]"
- **TF-IDF**: Statistical correlation between requirement notation patterns and document structure keywords

### Test Results
```
❌ Crosswalk Error:      FAIL (confidence: 1.000) ✓ CORRECT
✅ Banned Phrases:       PASS (confidence: 0.003)
✅ Name Inconsistency:   PASS (confidence: 0.005)
✅ Date Inconsistency:   PASS (confidence: 0.002)
```

---

## 3. Banned Phrases (scratch_banned.txt / .pdf)

### Expected Behavior
**Flags ONLY banned_phrases** - Detects prohibited language

### What Triggers the Error

**Line 56 (Commitment to Quality):**
```
This proposal provides an unconditional guarantee of our services and deliverables.
```

### Why This Is an Error
The phrase **"unconditional guarantee"** is prohibited in government/public sector proposals because:
1. **Legal liability**: Creates absolute contractual obligations
2. **Unrealistic promises**: No engineering project can guarantee perfection
3. **Industry standards**: Professional services use "best efforts" language instead
4. **Compliance**: Many RFPs explicitly prohibit such language

### Acceptable Alternatives
- "We are committed to delivering high-quality services"
- "Our approach emphasizes technical excellence"
- "We will use best efforts to achieve project goals"

### How Models Detect It
- **Transformer**: Contextual understanding of prohibited language patterns
- **TF-IDF**: Direct keyword matching for banned phrase combinations ("unconditional" + "guarantee")

### Test Results
```
✅ Crosswalk Error:      PASS (confidence: 0.001)
❌ Banned Phrases:       FAIL (confidence: 1.000) ✓ CORRECT
✅ Name Inconsistency:   PASS (confidence: 0.010)
✅ Date Inconsistency:   PASS (confidence: 0.001)
```

---

## 4. Name Inconsistency (scratch_name.txt / .pdf)

### Expected Behavior
**Flags ONLY name_inconsistency** - Detects name variation

### What Triggers the Error

**Line 7 (Executive Summary):**
```
Sarah Martinez will serve as the primary contact and Project Manager...
```

**Line 26 (Team Qualifications):**
```
Project Manager: Sarah M. (40% allocation)
```

### Why This Is an Error
The same person's name appears in two different formats:
- **Full name**: "Sarah Martinez"
- **Abbreviated**: "Sarah M."

This inconsistency:
1. **Appears unprofessional**: Suggests lack of attention to detail
2. **Creates ambiguity**: Could Sarah M. be a different person?
3. **Violates standards**: Most RFPs require consistent name formatting

### How Models Detect It
- **Transformer**: Understands name patterns and abbreviation relationships
- **TF-IDF**: Recognizes co-occurrence of full names and abbreviated forms in the same document

### Important Note
The model is **context-aware**:
- ✅ "Dr. Sarah Martinez" vs "Sarah Martinez, PE" - ACCEPTABLE (titles/credentials)
- ❌ "Sarah Martinez" vs "Sarah M." - **INCONSISTENT** (abbreviated last name)
- ❌ "John Smith" vs "J. Smith" - **INCONSISTENT**

### Test Results
```
✅ Crosswalk Error:      PASS (confidence: 0.002)
✅ Banned Phrases:       PASS (confidence: 0.014)
❌ Name Inconsistency:   FAIL (confidence: 1.000) ✓ CORRECT
```

---

## 5. Date Inconsistency (rule-based sample)

### Expected Behavior
**Rule detects illogical chronological order** between the anticipated submission date sentence and the signed/ sealed sentence.

### What Triggers the Error

**Lines 7-8 (Executive Summary):**
```
Our anticipated submission date remains March 15, 2024.
This proposal was signed and sealed on March 28, 2024.
```

### Why This Is an Error
**Temporal violation**: The document claims:
- **Submission date**: March 15, 2024
- **Signed date**: March 28, 2024

The signature happens **after** the anticipated submission. This indicates either a workflow error or stale template text.

### Correct Date Pattern
```
Signed: March 1, 2024
Submission: March 15, 2024  ✓ CORRECT (signature precedes submission)
```

### How Models Detect It
They don't anymore—**the UI now enforces a deterministic rule**:
1. Extract the “anticipated submission date …” sentence.
2. Extract the “signed and sealed on …” sentence.
3. Parse both dates and compare them.

If both dates are found and the signed date is later than the anticipated submission date, the UI flags **FAIL**. Missing or malformed dates default to **PASS (insufficient data)** to avoid false alarms.

### Rule Outcome (UI)
- Crosswalk Error: PASS
- Banned Phrases: PASS
- Name Inconsistency: PASS
- **Date Inconsistency (Rule-based)**: FAIL (signed 2024-03-28 after anticipated 2024-03-15)

---

## Understanding Model Behavior

### Why Manual Edits May Not Trigger Errors

The models are **pattern-based**, not rule-based. They detect specific learned patterns from training data:

#### ✅ Will Trigger Detection
- **Exact patterns from training**: Names like "John Smith" → "John S." (same pattern as training)
- **Similar contexts**: Requirement citations in wrong sections
- **Common variations**: Standard banned phrases like "guarantee", "unconditional"

#### ❌ May NOT Trigger Detection
- **Novel patterns**: Completely new error types not in training data
- **Subtle variations**: Minor name changes in unusual contexts
- **Single-word changes**: Random word replacements that don't create learned error patterns
- **Different locations**: Errors in unexpected document sections

### Example: Why Changing Names Manually May Fail

```python
# Original (triggers error):
"Sarah Martinez will serve..." → "Project Manager: Sarah M."
# ✓ Detected (exact training pattern)

# Your manual edit (may not trigger):
"Sarah Martinez will serve..." → "Sarah Martinez will coordinate..."
# ✗ Not detected (no name inconsistency created)

# To trigger manually, you need:
"Sarah Martinez will serve..." → keep as-is
"Project Manager: Sarah Martinez" → change to "Project Manager: S. Martinez"
# ✓ Will detect (creates abbreviated vs full name pattern)
```

### Key Insight: Location Matters

Models learned that certain errors appear in specific contexts:
- **Crosswalk errors**: Between executive summary and qualifications sections
- **Name inconsistency**: Between introduction and team roster
- **Date inconsistency**: Between cover letter dates and project schedule

Changing text in **random locations** may not trigger detection because the model hasn't seen that pattern during training.

---

## Summary Table

| Sample | Triggers | Key Pattern | Confidence |
|--------|----------|-------------|------------|
| **Clean** | None | Fully compliant document | N/A |
| **Crosswalk** | crosswalk_error | R7 in Schedule vs Qualifications | 100% |
| **Banned** | banned_phrases | "unconditional guarantee" phrase | 100% |
| **Name** | name_inconsistency | Sarah Martinez vs Sarah M. | 100% |
| **Date** | date_inconsistency | Signed Feb 28 vs Submit Mar 15 | 100% |

---

## For Developers

### Testing New Samples

When creating custom test cases:

1. **Use similar patterns**: Mimic the exact patterns from these samples
2. **Keep context**: Place errors in similar document sections
3. **Use similar names/dates**: The model learned specific patterns (R7, Sarah, dates in 2024-2026 range)
4. **Test incrementally**: Change one thing at a time and verify detection

### Model Limitations

These models are **supervised learners** trained on synthetic data with specific error patterns. They:
- ✅ Excel at detecting **learned patterns**
- ❌ May miss **novel error types** not in training data
- ✅ Achieve 100% accuracy on **similar examples**
- ❌ Are not perfect **general-purpose validators**

For production use, consider:
- Expanding training data with more diverse error patterns
- Adding rule-based validators for critical checks
- Regular model retraining with real-world examples
