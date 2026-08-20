from pathlib import Path
import random
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_chunks"
    / "document_chunks.csv"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

OUTPUT_PATH = (
    EVALUATION_DIR
    / "test_questions_v2_authoring.csv"
)

SEED = 20260810
random.seed(SEED)


# ============================================================
# CURRENT 12-DOCUMENT CORPUS
# ============================================================

DOCUMENT_ALLOCATION = {
    "University of Liverpool Undergraduate Handbook": 10,
    "University of Liverpool Postgraduate Handbook": 10,
    "Student Conduct Procedures — Appendix A": 8,
    "University of Liverpool Extenuating Circumstances Policy": 8,
    "University of Liverpool Diversity, Equality and Opportunity Policy": 7,
    "Student Complaints Policy and Procedure": 7,
    "Policy on UKVI Compliance — Student Route": 7,
    "Student Guide for UK Visa Holders 2026": 6,
    "Student Attendance Policy — Revised July 2025, Version 1.3": 5,
    "University of Liverpool Payment Policy 2026-27": 5,
    "Student Conduct Breaches and Indicative Sanctions — Appendix B": 5,
    "University of Liverpool Code of Practice on Assessment and Appendices 2025-26 — Summary of Changes": 4,
}

# 82 evidence-grounded single-document questions
assert sum(DOCUMENT_ALLOCATION.values()) == 82


# ============================================================
# SINGLE-DOCUMENT QUESTION TYPES
# ============================================================

SINGLE_TYPES = (
    ["factual"] * 25
    + ["paraphrased"] * 25
    + ["procedural"] * 13
    + ["reasoning"] * 12
    + ["adversarial_false_premise"] * 7
)

assert len(SINGLE_TYPES) == 82

random.shuffle(SINGLE_TYPES)


# ============================================================
# MULTI-DOCUMENT PAIRS
# ============================================================

UG = "University of Liverpool Undergraduate Handbook"
PG = "University of Liverpool Postgraduate Handbook"
CONDUCT_A = "Student Conduct Procedures — Appendix A"
CONDUCT_B = "Student Conduct Breaches and Indicative Sanctions — Appendix B"
EC = "University of Liverpool Extenuating Circumstances Policy"
COMPLAINTS = "Student Complaints Policy and Procedure"
ATTENDANCE = "Student Attendance Policy — Revised July 2025, Version 1.3"
UKVI = "Policy on UKVI Compliance — Student Route"
VISA = "Student Guide for UK Visa Holders 2026"
PAYMENT = "University of Liverpool Payment Policy 2026-27"
DIVERSITY = "University of Liverpool Diversity, Equality and Opportunity Policy"
ASSESSMENT = "University of Liverpool Code of Practice on Assessment and Appendices 2025-26 — Summary of Changes"

MULTI_DOCUMENT_PAIRS = [
    (UG, PG),
    (UG, PG),
    (UG, PG),
    (UG, PG),

    (ATTENDANCE, UKVI),
    (ATTENDANCE, UKVI),

    (VISA, UKVI),
    (VISA, UKVI),

    (CONDUCT_A, CONDUCT_B),
    (CONDUCT_A, CONDUCT_B),
    (CONDUCT_A, CONDUCT_B),

    (COMPLAINTS, CONDUCT_A),
    (COMPLAINTS, CONDUCT_A),

    (EC, UG),
    (EC, UG),

    (EC, PG),
    (EC, PG),

    (PAYMENT, UG),

    (DIVERSITY, COMPLAINTS),

    (ASSESSMENT, UG),
]

assert len(MULTI_DOCUMENT_PAIRS) == 20


# ============================================================
# HELPERS
# ============================================================

def get_page(row):
    """
    Get page number from a page_number column if available,
    otherwise extract it from the chunk ID.
    """

    if "page_number" in row.index:
        value = row["page_number"]

        if pd.notna(value):
            try:
                return int(float(value))
            except Exception:
                pass

    chunk_id = str(row["chunk_id"])

    match = re.search(
        r"_p(\d+)_",
        chunk_id,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return ""


def word_count(text):
    return len(str(text).split())


def difficulty_for(question_type):
    if question_type == "factual":
        return "easy"

    if question_type in {
        "paraphrased",
        "procedural",
        "ambiguous",
    }:
        return "medium"

    return "hard"


def note_for(question_type):
    notes = {
        "factual":
            "Write a direct factual question answerable from the gold evidence.",

        "paraphrased":
            "Write the question using wording different from the source while preserving the same meaning.",

        "procedural":
            "Ask about a process, sequence, requirement, deadline or action described by the evidence.",

        "reasoning":
            "Require interpretation of the evidence rather than simple keyword copying.",

        "adversarial_false_premise":
            "Create a plausible but false premise that the evidence allows the system to explicitly reject.",

        "multi_document_comparison":
            "Write a comparison question that genuinely requires evidence from both gold documents.",

        "multi_document_synthesis":
            "Write a question whose complete answer requires combining evidence from both gold documents.",

        "ambiguous":
            "Write a deliberately underspecified user question where a safe assistant should request clarification.",

        "unanswerable":
            "Write a realistic university-related question whose answer is NOT contained anywhere in the 12-document corpus.",
    }

    return notes.get(question_type, "")


# ============================================================
# LOAD CHUNKS
# ============================================================

print(f"Loading chunks from: {CHUNKS_PATH}")

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"Chunk file not found: {CHUNKS_PATH}"
    )

chunks = pd.read_csv(
    CHUNKS_PATH,
    encoding="utf-8-sig",
)

required_columns = {
    "document_name",
    "chunk_id",
    "chunk_text",
}

missing_columns = (
    required_columns
    - set(chunks.columns)
)

if missing_columns:
    raise ValueError(
        "document_chunks.csv is missing columns: "
        f"{sorted(missing_columns)}"
    )

print(f"Chunks loaded: {len(chunks)}")
print(
    "Documents:",
    chunks["document_name"].nunique(),
)


# ============================================================
# VALIDATE DOCUMENT NAMES
# ============================================================

corpus_documents = set(
    chunks["document_name"]
    .dropna()
    .astype(str)
    .unique()
)

expected_documents = set(
    DOCUMENT_ALLOCATION.keys()
)

missing_documents = (
    expected_documents
    - corpus_documents
)

if missing_documents:
    raise ValueError(
        "\nThese expected documents were not found "
        "in document_chunks.csv:\n"
        + "\n".join(
            sorted(missing_documents)
        )
    )

print("All 12 expected documents found.")


# ============================================================
# PREPARE CANDIDATE CHUNKS
# ============================================================

chunks = chunks.copy()

chunks["chunk_text"] = (
    chunks["chunk_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

chunks["word_count_temp"] = (
    chunks["chunk_text"]
    .map(word_count)
)

# Avoid extremely tiny/title-only chunks.
candidate_chunks = chunks[
    chunks["word_count_temp"] >= 50
].copy()

# Avoid exact duplicate text.
candidate_chunks = (
    candidate_chunks
    .drop_duplicates(
        subset=[
            "document_name",
            "chunk_text",
        ]
    )
    .reset_index(drop=True)
)

print(
    "Usable evidence chunks:",
    len(candidate_chunks),
)


# ============================================================
# SAMPLE SINGLE-DOCUMENT EVIDENCE
# ============================================================

selected_single = []

random_state = SEED

for document_name, number_needed in (
    DOCUMENT_ALLOCATION.items()
):
    document_pool = candidate_chunks[
        candidate_chunks["document_name"]
        == document_name
    ].copy()

    if len(document_pool) < number_needed:
        raise ValueError(
            f"Not enough usable chunks for "
            f"{document_name}. "
            f"Needed {number_needed}, "
            f"found {len(document_pool)}."
        )

    selected = document_pool.sample(
        n=number_needed,
        random_state=random_state,
    )

    random_state += 1

    selected_single.extend(
        selected.to_dict("records")
    )

random.shuffle(selected_single)

assert len(selected_single) == 82


# ============================================================
# BUILD DATASET ROWS
# ============================================================

rows = []
question_number = 1


def make_id(number):
    return f"V2Q{number:03d}"


# ------------------------------------------------------------
# 82 single-document evidence-grounded questions
# ------------------------------------------------------------

for index, evidence in enumerate(
    selected_single
):
    question_type = SINGLE_TYPES[index]

    page = get_page(
        pd.Series(evidence)
    )

    rows.append(
        {
            "question_id":
                make_id(question_number),

            "question":
                "",

            "question_type":
                question_type,

            "difficulty":
                difficulty_for(
                    question_type
                ),

            "answerable":
                "Yes",

            "expected_answer":
                "",

            "gold_documents":
                evidence[
                    "document_name"
                ],

            "gold_page":
                page,

            "gold_chunk_ids":
                evidence[
                    "chunk_id"
                ],

            "gold_evidence":
                evidence[
                    "chunk_text"
                ],

            "required_evidence_count":
                1,

            "notes":
                note_for(
                    question_type
                ),
        }
    )

    question_number += 1


# ------------------------------------------------------------
# 20 multi-document questions
# ------------------------------------------------------------

for pair_index, (
    document_1,
    document_2,
) in enumerate(MULTI_DOCUMENT_PAIRS):

    pool_1 = candidate_chunks[
        candidate_chunks["document_name"]
        == document_1
    ]

    pool_2 = candidate_chunks[
        candidate_chunks["document_name"]
        == document_2
    ]

    evidence_1 = pool_1.sample(
        n=1,
        random_state=SEED + 100 + pair_index,
    ).iloc[0]

    evidence_2 = pool_2.sample(
        n=1,
        random_state=SEED + 200 + pair_index,
    ).iloc[0]

    if pair_index < 10:
        question_type = (
            "multi_document_comparison"
        )
    else:
        question_type = (
            "multi_document_synthesis"
        )

    page_1 = get_page(evidence_1)
    page_2 = get_page(evidence_2)

    rows.append(
        {
            "question_id":
                make_id(question_number),

            "question":
                "",

            "question_type":
                question_type,

            "difficulty":
                "hard",

            "answerable":
                "Yes",

            "expected_answer":
                "",

            "gold_documents":
                (
                    f"{document_1} | "
                    f"{document_2}"
                ),

            "gold_page":
                (
                    f"{page_1} | "
                    f"{page_2}"
                ),

            "gold_chunk_ids":
                (
                    f"{evidence_1['chunk_id']} | "
                    f"{evidence_2['chunk_id']}"
                ),

            "gold_evidence":
                (
                    "[SOURCE 1]\n"
                    + str(
                        evidence_1[
                            "chunk_text"
                        ]
                    )
                    + "\n\n[SOURCE 2]\n"
                    + str(
                        evidence_2[
                            "chunk_text"
                        ]
                    )
                ),

            "required_evidence_count":
                2,

            "notes":
                note_for(
                    question_type
                ),
        }
    )

    question_number += 1


# ------------------------------------------------------------
# 10 ambiguous clarification questions
# ------------------------------------------------------------

for _ in range(10):

    rows.append(
        {
            "question_id":
                make_id(question_number),

            "question":
                "",

            "question_type":
                "ambiguous",

            "difficulty":
                "medium",

            "answerable":
                "Needs clarification",

            "expected_answer":
                "",

            "gold_documents":
                "",

            "gold_page":
                "",

            "gold_chunk_ids":
                "",

            "gold_evidence":
                "",

            "required_evidence_count":
                0,

            "notes":
                note_for(
                    "ambiguous"
                ),
        }
    )

    question_number += 1


# ------------------------------------------------------------
# 8 genuinely unanswerable questions
# ------------------------------------------------------------

for _ in range(8):

    rows.append(
        {
            "question_id":
                make_id(question_number),

            "question":
                "",

            "question_type":
                "unanswerable",

            "difficulty":
                "hard",

            "answerable":
                "No",

            "expected_answer":
                (
                    "The provided documents do "
                    "not contain enough information "
                    "to answer this question."
                ),

            "gold_documents":
                "NONE",

            "gold_page":
                "",

            "gold_chunk_ids":
                "",

            "gold_evidence":
                "",

            "required_evidence_count":
                0,

            "notes":
                note_for(
                    "unanswerable"
                ),
        }
    )

    question_number += 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

dataset = pd.DataFrame(rows)

if len(dataset) != 120:
    raise RuntimeError(
        f"Expected 120 rows, "
        f"created {len(dataset)}."
    )


# ============================================================
# SAVE
# ============================================================

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

dataset.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("QUESTION SET V2 AUTHORING DATASET CREATED")
print("=" * 60)

print(
    f"Total rows: {len(dataset)}"
)

print()
print("Question types:")
print(
    dataset[
        "question_type"
    ]
    .value_counts()
    .to_string()
)

print()
print("Answerability:")
print(
    dataset[
        "answerable"
    ]
    .value_counts()
    .to_string()
)

print()
print("Difficulty:")
print(
    dataset[
        "difficulty"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "Evidence-grounded rows:",
    (
        dataset[
            "gold_chunk_ids"
        ]
        .fillna("")
        .str.strip()
        .ne("")
        .sum()
    ),
)

print(
    "Rows awaiting question writing:",
    (
        dataset[
            "question"
        ]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    ),
)

print()
print("Saved to:")
print(OUTPUT_PATH)
