# Hacker News Post Classification — Fine-Tuned DistilBERT

## 1. Overview

This project builds a 3-class text classification system for Hacker News posts and comments.

We compare two approaches:

Fine-tuned transformer model (DistilBERT)
Zero-shot classification using an LLM (Groq)

The goal is to classify reasoning depth in discussions:

surface reaction
opinion with reasoning
deep technical analysis

---

## 2. Dataset

- Source: Hacker News API (top stories + comments)
- Size: 200+ labeled examples
- Format: CSV with `text`, `label`, `label_id`

Label distribution was balanced using oversampling during training.

**Dataset Construction Pipeline**

Fetch top story IDs
Fetch each story using item endpoint
Extract title + text
Extract up to 3 comments per story
Clean HTML content
Store into CSV (text, label, label_id)

## 3. Labels

- `surface_reaction` — short reactions with no reasoning  
- `opinion_reasoned` — opinions with justification  
- `analysis` — deep technical/system-level explanation  

**Labeling Strategy**

Step 1: Rule-based labeling

Heuristics used:

Keywords: "because", "I think", "however"
Length thresholds
Structural patterns

Step 2: Manual validation

Every sample was:

Manually reviewed
Corrected where needed
Verified against label definitions

✔ This ensures dataset quality and reduces noise

## 4. Model Training

### Fine Tune Model
- distilbert-base-uncased (HuggingFace Transformers)

**Training Setup** 
- Epochs: 4  
- Learning rate: 2e-5  
- Batch size: 16  
- Loss: Weighted cross-entropy (to handle imbalance)  
- Platform: Google Colab (T4 GPU)

### Baseline (Zero-shot Groq)

- Model: `llama-3.3-70b-versatile`
- Prompt-based classification using label definitions from `planning.md`
- Output constrained to label-only responses

Baseline was evaluated on the same test split as the fine-tuned model and used to measure how well a general model performs without training.

## 5. Results Summary

|       Model           | Accuracy|
|-----------------------|---------|
| Fine-tuned DistilBERT | *0.917* |
| Zero-shot baseline    | *0.630* |


Improvement = Fine-tuned accuracy − Baseline accuracy=0.917-0.630=*0.287*


## 6. Confusion Matrix (Fine-tuned Model)

| True \ Pred      | surface_reaction | opinion_reasoned | analysis |
|------------------|------------------|------------------|----------|
| surface_reaction | 36               | 0                | 0        |
| opinion_reasoned | 2                | 27               | 7        |
| analysis         | 0                | 0                | 36       |

**Confusion Matrix Interpretation**

The confusion matrix shows:
- Strong diagonal performance overall
- Concentrated confusion between opinion_reasoned and analysis
- Minimal confusion involving surface_reaction except very short borderline cases

## 7.Per-class metrics

### Fine-tuned Model

                    precision    recall  f1-score   support

surface_reaction       0.95      1.00      0.97        36
opinion_reasoned       1.00      0.75      0.86        36
        analysis       0.84      1.00      0.91        36

        accuracy                           0.92       108
       macro avg       0.93      0.92      0.91       108
    weighted avg       0.93      0.92      0.91       108


**Precision**

“When the model predicts this class, how often is it correct?”

Precision=TP+FP/TP
High precision → few false positives.

**Recall**

“Out of all real items of this class, how many did the model catch?”

Recall=TP/(TP+FN)
	
High recall → few false negatives.

**F1-score**

Balance between precision and recall

F1=2*(Precision.Recall)/Precision+Recall

Useful when you want a single balanced metric.

*Class-wise interpretation*

**surface_reaction**
Precision: 0.95
Recall: 1.00
F1: 0.97

Interpretation:

Model caught all surface_reaction cases (perfect recall)
Very few false positives
This is your strongest class

**opinion_reasoned**
Precision: 1.00 (perfect)
Recall: 0.75 (weakest recall)

Interpretation:

When model predicts this class → it's always correct
BUT it misses 25% of actual cases

So:

Model is conservative
It under-predicts this class

It only predicts “opinion_reasoned” when it is very confident, otherwise it labels something else.

**analysis**
Precision: 0.84
Recall: 1.00

Interpretation:

It catches all real “analysis” cases (perfect recall)
But mislabels some other classes as “analysis” (lower precision)

So:

Model is over-predicting analysis
Some false positives exist

*Macro Average vs Weighted Average*

Macro Average
Simple average of all classes
Treats all classes equally
precision = 0.93
recall    = 0.92
f1-score  = 0.91

👉 Good indicator when classes are balanced (like yours)

📌 Weighted Average
Weighted by support (number of samples per class)
precision = 0.93
recall    = 0.92
f1-score  = 0.91

Since the dataset is balanced, macro ≈ weighted.

### Baseline Model
                  precision    recall  f1-score   support

surface_reaction       1.00      0.56      0.71        36
opinion_reasoned       0.46      0.58      0.51        36
        analysis       0.64      0.75      0.69        36

        accuracy                           0.63       108
       macro avg       0.70      0.63      0.64       108
    weighted avg       0.70      0.63      0.64       108


## 8. Error Analysis

### Total errors (from notebook):
- Wrong predictions: **9 / 108**

### Key failure pattern:
The model confuses:
- opinion_reasoned → analysis
- occasional `analysis → surface_reaction`
This indicates:

- model struggles with distinguishing structured reasoning vs explanatory tone

### Sample Misclassifications

Wrong predictions: 9 / 108

--- #1 ---

Text:      The article also raises an interesting question. My understanding is the big difference in North American and British color TV is that NTSC was engineered to be backwards compatible with existing blac...
True:      opinion_reasoned
Predicted: analysis  (confidence: 0.49)

Reason: Interpreted as technical explanation instead of opinionated interpretation.

--- #2 ---

Text:      This article uses so many words to focus on the political reasons, but completely ignores the primary driver: Cost.<p>Korean weapons systems are 40-60% cheaper than their American counterparts.<p>The ...
True:      opinion_reasoned
Predicted: analysis  (confidence: 0.58)

Reason: Argument expressed in analytical tone.

--- #3 ---

Text:      Parasympathetic nervous activation <i>increased</i> risk-taking behavior? That's interesting/unexpected (at least to me). Also, this part caught my eye:<p>> The selective impact of prolonged exhalatio...
True:      opinion_reasoned
Predicted: analysis  (confidence: 0.55)

Reason: Scientific explanation misread as neutral analysis.

--- #4 ---

Text:      > In-game constructions of NAND gates and a perceptron (forward prop and training) as described in in 'If LLMs Have Human-Like Attributes, Then So Does Age of Empires II'.<p>Interesting concept<p>> We...
True:      opinion_reasoned
Predicted: analysis  (confidence: 0.90)

Reason:  Technical keywords triggered analysis label.

--- #5 ---

Text:      > [...] the maker was almost certainly a transcriber who used it to keep his place on the page and note the column he was writing in when he stopped. The wheel would be moved to the stopping point and...
True:      opinion_reasoned
Predicted: analysis  (confidence: 0.51)

Reason: Historical explanation interpreted as analysis.

--- #6 ---
Text:      AMD will reinstate memory encryption on Ryzen 9000 CPUs via BIOS update in July
True:      opinion_reasoned
Predicted: surface_reaction  (confidence: 0.49)

Reason: Model treated as simple news reaction.

--- #7 ---

Text:      This article uses so many words to focus on the political reasons, but completely ignores the primary driver: Cost.<p>Korean weapons systems are 40-60% cheaper than their American counterparts.<p>The ...
True:      opinion_reasoned
Predicted: analysis  (confidence: 0.58)

Reason:contains structured comparison + numerical evidence, which looks like analytical reasoning even though the intent is opinionated critique.

--- #8 ---

Text:      A dead comment says:<p>> Of course, this assumes independent events. World Cup, super bowls, etc break these assumptions.<p>Yes, this is very true. The model here works for Poisson arrivals and expone...
True:      opinion_reasoned
Predicted: analysis  (confidence: 0.79)

Reason: Mathematical explanation misinterpreted as deep analysis.

--- #9 ---

Text:      A "one-time" tax to fund recurring health care and educational expenses is an obvious lie.
True:      opinion_reasoned
Predicted: surface_reaction  (confidence: 0.45)

Reason: Model focused on emotional phrasing (“obvious lie”).

### Key Insight

The model tends to **over-predict `analysis`**, especially when:
- Posts contain any technical vocabulary
- Posts include structured sentences or citations
- Opinion + explanation overlap occurs

This suggests:
- The boundary between `opinion_reasoned` and `analysis` is the hardest
- Label definitions may need stricter separation based on “mechanism depth”

## 9. Reflection

The model successfully learned general distinctions between reaction, opinion, and analysis. However, it struggles when reasoning depth is ambiguous.

What it captured well:
- Surface-level vs long-form distinction
- Strong technical keywords often map to analysis

What it missed:
- Subtle reasoning that is not fully technical
- Context-dependent interpretation of “depth”

This suggests that label boundaries are semantically close and may require:
- more edge-case training data
- stronger annotation consistency rules

## 10. Spec vs Implementation Reflection

### What worked well
- Clear label definitions improved training stability
- Weighted loss helped reduce class imbalance bias

### What diverged
- Real-world posts often blend multiple label characteristics
- Some annotations were ambiguous despite definitions

## 11. AI Usage

1. **Label design support**

   - Used LLM to test borderline cases between analysis and opinion_reasoned
   - Revised definitions to emphasize “mechanism depth”

2. **Failure pattern analysis**
   - Used LLM to cluster misclassified examples
   - Verified that most errors involved analysis over-prediction

## 12.Verification of Posts/Comments

https://hacker-news.firebaseio.com/v0/topstories.json

This does NOT return posts or comments.
It returns only a list of story IDs

Example output:
[ 48620462, 48621153, 48621059, ... ]

These are IDs of top posts on Hacker News.

**How do you get actual posts**

You must fetch each ID like this:

https://hacker-news.firebaseio.com/v0/item/<ID>.json

Example:https://hacker-news.firebaseio.com/v0/item/48620462.json

That returns:

{"by":"theanonymousone","descendants":59,"id":48620462,"kids":[48621153,48623008,48621059,48622858,48621774,48622757,48621162,48621470,48621419,48623670,48621099,48622535,48621536,48621702],"score":115,"time":1782060978,"title":"Burnout is real for open source maintainers","type":"story","url":"https://openjsf.org/blog/burnout-is-real-for-open-source-maintainers"}

**How comments are fetched**

If a post has:
"kids": [48621153,48623008,48621059]

Each kid is a comment ID, and you fetch:

https://hacker-news.firebaseio.com/v0/item/48621153.json

That returns:

{"by":"FinnLobsien","id":48621153,"kids":[48622404,48622298,48621728,48621524,48621382,48622700],"parent":48620462,"text":"If you have a hobby project like writing a blog, crocheting, or almost any other creative hobby, you can dip in and out however it suits you. If you deal with major life events, sicknesses, etc., you can leave the hobby and come back. Nobody is paying you for it, so nobody can complain (maybe the friends who miss you, but it&#x27;s not actively impacting the real world).<p>Open source is one of those weird things where your hobby project can become an essential piece of infrastructure.<p>It&#x27;s like if you loved crocheting, but somehow if you stopped crocheting everyone in your city would no longer have clothes and need to walk around naked.","time":1782065474,"type":"comment"}

Data was collected using the Hacker News public Firebase API. Each text entry corresponds to a real post or comment retrieved using item endpoints from top story IDs.

## 12. Conclusion

The fine-tuned model significantly improves structured classification over zero-shot prompting and demonstrates strong performance on short and medium-length posts.

Remaining improvements depend primarily on:
- Sharpening label boundaries
- Adding more edge-case training examples
- Reducing overlap between reasoning and analysis categories# ai201-project3-takemeter_starter