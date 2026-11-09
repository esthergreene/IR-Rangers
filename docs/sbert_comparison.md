# Sentence-BERT: Pretrained vs. Fine-Tuned Comparison

## Objective
The goal of this part of the project was to see if fine-tuning the Sentence-BERT on our own training data would improve the paragraph retrieval compared to using the original pretrained model.

## Experimental Setup

For this experiment, I used `sentence-transformers/all-MiniLM-L6-v2` as the base model and fine-tuned it using the training data we prepared for the project. So if we look here:

- **Base model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Training data:** 1,800 queries with 5,138 positive training pairs
- **Validation data:** 200 held-out queries
- **Paragraph collection:** 2,275,547 paragraphs
- **Retrieval method:** FAISS dense vector search
- **Results retrieved:** Top 100 paragraphs per query
- **Evaluation metrics:** nDCG@10, MRR@10, and Precision@10

Then I built separate FAISS indexes for the pretrained and fine-tuned models. Both models were tested using the same 200 validation queries and paragraph collection so that I could compare their performance fairly.

## Evaluation Results

| Metric | Pretrained | Fine-Tuned | Relative Improvement |
|---|---:|---:|---:|
| nDCG@10 | 0.3760 | 0.4123 | +9.65% |
| MRR@10 | 0.4393 | 0.4825 | +9.83% |
| Precision@10 | 0.1050 | 0.1160 | +10.48% |

## Findings

The fine-tuned model performed better than the pretrained model across all three metrics.

The nDCG@10 score increased from 0.3760 to 0.4123, suggesting that fine-tuning helped the model rank relevant paragraphs more effectively. 

MRR@10 also improved, meaning the model generally returned the first relevant paragraph higher in the rankings. 

Precision@10 increased as well, showing that more judged-relevant paragraphs appeared within the top 10 resutls. 

Overall the fine-tuning resulted in roughly a 10% relative improvement across all three metrics, which was a good outcome for this experiment. 

## Limitations

One thing to keep in mind is that these results are based on 200 held out validation queries rather than a separate test set. I also haven't tested whether the improvements are statistically significant. 

Our team's existing BMM25 and TF-IDF results were evaluated using the Study queries, so those scores aren't directly comparable to these results just yet.

Another limitation is that not every relevant paragraph is necessarily included in the relevance judgements, which could affect the evaluation scores. 

## Conclusion

Based on these resylts, fine-tuning Sentence-BERT appears to have improved paragraph retrieval performance for our dataset, but not as good as our other model.

The fine-tuned model outperformed the original pretrained model across all three metrics, with improements of approximately 10%, which is good. 

The next step is to evaluate Sentence-BERT using the Study queries so we can compare its performance against the other retrieval models our team has implemented.