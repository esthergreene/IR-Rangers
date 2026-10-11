from ranx import Qrels, Run, evaluate, compare

qrels = Qrels.from_file("/TO BE DETERMINED.tsv", kind="trec")
run = Run.from_file("/TO BE DETERMINED.tsv", kind="trec")
qrels.set_relevance_level(rel_lvl=2)

eval_res = evaluate(qrels, run, "precision@5", make_comparable=True)
print(eval_res)

eval_res = evaluate(qrels, run, "precision@10", make_comparable=True)
print(eval_res)