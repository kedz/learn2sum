
#python -m cProfile -o train-profile train.py \
#    --duc-inputs duc2003.input.feats.tsv \
#    --duc-models duc2003.models.tsv \
#    --stopped-lemma-length-filter 10 \
#    --iters 10

time python train.py --duc-inputs duc2003.input.feats.tsv \
                --duc-models duc2003.models.tsv \
                --stopped-lemma-length-filter 10 \
                --iters 10 \
                -o all_features
