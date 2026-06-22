# planning.md — Hacker News Post Classification

## 1. Community

I chose the **Hacker News (HN) discussion community**, which consists of tech-focused news posts and comment threads submitted by developers, researchers, and startup founders.Each post on Hacker News contains a title, a link (optional), and a comment section where users engage in discussions.
Each post contains nested comments referred to as "kids", meaning replies are stored as IDs rather than directly embedded text. To retrieve full discussions, we must follow a specific URL/API pattern and recursively fetch these child comment IDs, making data extraction slightly indirect.

This community is a strong fit for classification because:
- Posts range from very short reactions (“interesting”, “wow”) to long technical explanations.
- The same topic can be discussed at multiple depths (surface reaction vs deep analysis).
- Users frequently mix opinion, technical reasoning, and critique in the same thread.

This variation makes it ideal for a classification task focused on **depth of reasoning and intent**, rather than topic.

Reddit was considered as a potential data source; however, accessing Reddit data requires authentication using a client_id and client_secret. At this stage, we were unable to generate these credentials due to permission requirements. Therefore, Reddit data is not accessible for this project and is excluded from the dataset.

## 2. Labels

### Label 1: `surface_reaction`
**Definition:**  
A short or low-effort reaction to a post with minimal reasoning or explanation.

**Examples:**
- “Wow, I didn’t expect this.”
- “Interesting article.”

**Uncertain case:**
- “I think this is interesting because CPUs are getting faster.”  
  → Could be surface reaction or opinion_reasoned depending on reasoning depth.

### Label 2: `opinion_reasoned`
**Definition:**  
A subjective opinion that includes some reasoning, explanation, or justification, but does not deeply analyze systems or technical mechanisms.

**Examples:**
- “This is bad policy because it increases costs without improving outcomes.”
- “I think this approach works better, but it still has scalability issues.”

**Uncertain case:**
- “This seems inefficient because it uses too many layers of abstraction.”  
  → Borderline between opinion_reasoned and analysis.

### Label 3: `analysis`
**Definition:**  
A deep, technical, or structured explanation that breaks down systems, mechanisms, or cause-effect relationships in detail.

**Examples:**
- “This works because the caching layer reduces O(n) queries to O(1) lookups using hash indexing.”
- “NTSC was designed for backward compatibility, which constrained signal encoding decisions.”

**Uncertain case:**
- “This article explains that costs drive weapon system differences.”  
  → Could be analysis or opinion_reasoned depending on depth of supporting explanation.

## 3. Mutual Exclusivity Strategy

Most ambiguity occurs between:
- `opinion_reasoned` vs `analysis`

To handle this:
- If the post explains **why something is true using mechanisms, systems, or structured breakdown → analysis**
- If it is mainly **argumentative or interpretive without deep mechanism → opinion_reasoned**
- If it is short or emotional → surface_reaction

This rule ensures most posts map to exactly one label.

## 4. Community + Label Rationale

Hacker News discussions combine technical depth with casual commentary. Distinguishing between reaction, opinion, and analysis is important because:
- Moderation systems can prioritize high-signal technical posts.
- Recommendation systems can surface deeper insights over low-effort comments.
- Researchers can analyze how technical understanding spreads in discussions.

## 5. Hard Edge Cases

### Hardest case:
Posts that include **light technical explanation embedded in opinion**, such as:
> “This is inefficient because it adds unnecessary layers between the API and database.”

**Why it's hard:**
- Contains reasoning (“because”)
- But lacks deep system-level breakdown

**Resolution rule:**
- If it explains *mechanism or architecture in detail → analysis*
- Otherwise → opinion_reasoned

## 6. Data Collection Plan

- Source: Hacker News API (top stories + comments)
- Dataset size: **200–400 posts**
- Collection method: automated API extraction + manual cleaning
- Labeling method: manual annotation using label definitions

If a label is underrepresented after 200 examples:
- Actively sample more from HN comment threads (not just top-level posts)
- Oversample rare label categories during collection

## 7. Evaluation Metrics

Metrics used:
- Accuracy (overall correctness)
- Precision, Recall, F1-score per class
- Confusion matrix

Why these metrics:
- Accuracy alone hides class imbalance issues
- F1-score captures tradeoff between precision and recall
- Confusion matrix reveals systematic misclassification (especially between opinion_reasoned and analysis)

## 8. Definition of Success

The classifier is successful if:
- Overall accuracy ≥ 75%
- Minimum per-class F1 ≥ 0.70 for all labels
- No single label dominates predictions (>60% bias)

This ensures the model is usable for:
- Filtering high-quality technical discussions
- Identifying low-effort comments in feeds
- Structuring community analysis tools

## 9. AI Tool Plan

-

## AI Tool Plan (Rule-Based + Human Validated Labeling)

### 1. Label Generation Strategy (Heuristic-Based Initialization + Manual Review)

The dataset labels were generated using a rule-based Python function (`assign_label`) and then manually reviewed and corrected.

The labeling logic is as follows:

* **surface_reaction**

  * Assigned when text length is **less than 12 words**
  * Indicates short, low-effort, or immediate reactions

* **analysis**

  * Assigned when text length is **greater than 25 words**
  * AND contains reasoning indicators such as:

    * "because", "therefore", "this means", "as a result"
  * Represents structured reasoning or explanation-based content

* **opinion_reasoned**

  * Assigned when text contains opinion or judgment cues such as:

    * "i think", "i believe", "should", "probably", "however", "but"
  * Serves as the default reasoning/opinion category when no stronger rule applies

* **Fallback rule**

  * If no other condition is matched, the label defaults to `opinion_reasoned`

### 2. Human-in-the-Loop Validation Process

Although labels were initially assigned using heuristics, the dataset is considered **human-validated** because:

* Step 1: Automatic labeling using the `assign_label(text)` function
* Step 2: Manual review of all assigned labels
* Step 3: Correction of incorrect or ambiguous cases by a human annotator
* Step 4: Final labels stored as ground truth in the dataset

This ensures that the dataset reflects **human judgment rather than purely rule-based output**.

### 3. Labeling Bias and Design Considerations

The labeling strategy introduces intentional behavior:

* Default fallback to `opinion_reasoned` ensures uncertain cases are not lost
* `analysis` is only assigned when strong reasoning + sufficient length is present
* `surface_reaction` is strictly limited to short texts, making it a stable and high-confidence class

### 4. Data Quality Checks

After labeling and manual validation:

* Label distribution was verified using `value_counts()`
* Ensured all labels match predefined `LABEL_MAP`
* Confirmed no unknown or invalid labels exist in the dataset

### 5. Summary of Dataset Construction

* Labels are **rule-initialized + human-corrected**
* Final dataset represents **manually validated ground truth**
* Heuristic rules were used only as a **bootstrapping mechanism**
* Human review ensures consistency, correctness, and reliability for model training

### 6. Label Stress-Testing (Pre-Annotation Validation)

Before annotation, label definitions are tested using an LLM to generate borderline examples between classes.

Process:

Provide the model with full label definitions:
surface_reaction
opinion_reasoned
analysis
Ask it to generate 5–10 posts per boundary pair

Boundary pairs tested:

opinion_reasoned vs analysis
surface_reaction vs opinion_reasoned
analysis vs surface_reaction (edge ambiguity check)

Evaluation step:

Manually attempt to classify generated posts
Identify cases where classification is unclear or inconsistent

Outcome:

If posts cannot be cleanly classified, label definitions are refined before dataset annotation begins
Ensures strong separation between classes prior to training data creation

### 7. Annotation Pipeline (Rule-Based + Human Validation)
Important Clarification
The LLM is not used for ground-truth labeling
Final dataset labels are not AI-generated
Labeling Workflow
Initial labeling using rule-based function (assign_label())
Manual human review of all labeled samples
Correction of incorrect or ambiguous labels
Dataset Storage Policy
No ai_label column is maintained
Only final validated labels are stored in:
df["label"]
Design Rationale
Ensures deterministic labeling logic
Improves reproducibility
Avoids LLM-driven label bias
Guarantees human accountability in final dataset quality

### 8.Failure Analysis (Post-Model Evaluation)

After model evaluation, misclassified examples are analyzed using an LLM to identify systematic error patterns.

Input to LLM

For each misclassified example:

Input text
Predicted label
True label
LLM Role
Cluster similar error cases
Suggest recurring confusion patterns
Highlight weak boundaries between classes
Common Error Patterns Investigated
analysis vs opinion_reasoned
opinion_reasoned vs surface_reaction
Over-prediction of analysis
Missed detection of opinion_reasoned in borderline cases