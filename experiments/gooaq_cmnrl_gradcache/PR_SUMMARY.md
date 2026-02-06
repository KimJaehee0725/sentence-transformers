# GooAQ Compare Summary (PR Notes)

## Context
- Dataset: GooAQ (train 100k / eval 10k)
- Eval corpus: random negatives 50k
- Metric: NDCG@10
- Results source files:
- `experiments/gooaq_cmnrl_gradcache/results/mpnet-base-gooaq-compare-*.post_eval.json`
- `experiments/gooaq_cmnrl_gradcache/results/mpnet-base-gooaq-compare-*.train.json`
- GradCache + repr cache: `experiments/gooaq_cmnrl_gradcache/results/caching/mpnet-base-gooaq-compare-grad-cache.*`

## Key Results (NDCG@10, train samples/s)

| Run | bank_size | grad_accum | train_bs | mini_bs | NDCG@10 | train_samples/s |
|---|---:|---:|---:|---:|---:|---:|
| vanilla-small | — | 1 | 16 | — | 0.6811 | 135.679 |
| grad-accum | — | 4 | 16 | — | 0.6840 | 136.724 |
| cont-accum | 4 | 4 | 16 | — | 0.7094 | 135.709 |
| grad-cache (baseline) | 0 | 1 | 64 | 16 | 0.7141 | 114.525 |
| grad-cache + repr cache | 1 | 1 | 64 | 16 | 0.7176 | 117.584 |

## Takeaways
- ContAccum preserves or improves quality: NDCG@10 0.7094 vs vanilla 0.6811 (+0.0283).
- Training speed is comparable: cont-accum 135.709 samples/s vs vanilla 135.679 (+0.02%).
- GradCache에서도 적용 가능: NDCG@10 0.7141 → 0.7176 with repr cache (+0.0035).
- Throughput +2.7% (114.525 → 117.584 samples/s).
