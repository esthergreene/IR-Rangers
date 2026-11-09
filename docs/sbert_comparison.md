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

## Study Query Evaluation

After evaluating both Sentence-BERT models on the 200 held-out validation queries, I also tested them using the 250 Study queries. This allowed me to compare the pretrained and fine-tuned models against BM25 using the same queries, relevance judgments, and evaluation pipeline.

All three models retrieved the top 100 paragraphs from the same collection of 2,275,547 paragraphs.

| Metric | BM25 | Pretrained SBERT | Fine-Tuned SBERT |
|---|---:|---:|---:|
| nDCG | 0.457 | 0.458 | 0.447 |
| nDCG@5 | 0.394 | 0.385 | 0.373 |
| MRR | 0.487 | 0.484 | 0.476 |
| Precision@5 | 0.164 | 0.174 | 0.170 |
| Precision@10 | 0.101 | 0.107 | 0.109 |
| Recall@100 | 0.594 | 0.633 | 0.605 |

These results were different from what I observed during validation. Although fine-tuning improved performance on the held-out validation queries, the pretrained Sentence-BERT model performed better on most metrics when evaluated using the Study queries.

The pretrained model achieved higher nDCG, MRR, Precision@5, and Recall@100 than the fine-tuned model. However, the fine-tuned model had a slightly higher Precision@10 score.

One interesting result was that pretrained Sentence-BERT achieved a Recall@100 of 0.633, compared to 0.594 for BM25 and 0.605 for fine-tuned Sentence-BERT. According to the evaluation script's paired significance test, this difference was statistically significant (p < 0.05) against both models.

Overall, these results suggest that the improvements from fine-tuning did not generalize to the Study queries as expected. One possible explanation is that the fine-tuned model learned patterns from the training data that were less useful for the Study queries, although further testing would be needed to determine why.

## Limitations

One limitation of this experiment is that the initial validation results were based on 200 held-out queries from the Train dataset. Although the fine-tuned model performed better on those queries, the improvement did not carry over to the 250 Study queries.

The validation improvements were not tested for statistical significance. In the Study evaluation, the pretrained model achieved significantly higher Recall@100 than both BM25 and the fine-tuned model, but the other differences were not reported as statistically significant.

Another limitation is that not every relevant paragraph is necessarily included in the relevance judgments, which could affect the evaluation scores.

It is also possible that the fine-tuning process caused the model to become too specialized for the training data. However, more testing would be needed to determine whether this was actually the case.

## Conclusion

This experiment showed that fine-tuning Sentence-BERT improved paragraph retrieval performance on the held-out validation queries, with approximately 10% relative improvement across nDCG@10, MRR@10, and Precision@10.

However, when I evaluated both models using the Study queries, the pretrained Sentence-BERT model performed better on most metrics. The fine-tuned model achieved a slightly higher Precision@10, but the pretrained model had better overall ranking and recall results.

The pretrained model also achieved the highest Recall@100 of the three models evaluated, significantly outperforming both BM25 and fine-tuned Sentence-BERT on that metric.

Overall, I found that fine-tuning can improve retrieval performance, but those improvements do not necessarily generalize to different query sets. This was an important finding because it showed why evaluating models on more than one set of queries matters.