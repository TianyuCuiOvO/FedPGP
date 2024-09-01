#!/bin/bash

cd ../

# custom config
DATA="DATA/"
MODEL=fedavgmaple
TRAINER=MaPLe
OT=COT
TOP_PERCENT=0.80
EPS=0.01
THRESH=0.001
MAX_ITER=100
PRETRAINED=True
LR=0.001
GAMMA=1
LOGITS2=False
USERS=10
FRAC=1
ROUND=25
LOCAL_EPOCH=2
NUM_PROMPT=1
#DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
IID=False
CSC=False  # class-specific context (False or True)
USEALL=False
TEMP=0.5
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in oxford_pets oxford_flowers dtd food101 caltech101
  do
    for SHOTS in 8 16
    do
      for SEED in 0 1
      do
        DIR=output_base/${DATASET}/${MODEL}_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}
#               DIR=output_base/${DATASET}/${MODEL}_${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
        if [ -d "$DIR" ]; then
          echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
          python federated_main.py \
          --root ${DATA} \
          --model ${MODEL} \
          --seed ${SEED} \
          --num_users ${USERS} \
          --frac ${FRAC} \
          --lr ${LR} \
          --temp ${TEMP} \
          --logits2 ${LOGITS2} \
          --OT ${OT} \
          --top_percent ${TOP_PERCENT} \
          --eps ${EPS} \
          --thresh ${THRESH} \
          --max_iter ${MAX_ITER} \
          --trainer ${TRAINER} \
          --round ${ROUND} \
          --local_epoch ${LOCAL_EPOCH} \
          --num_prompt ${NUM_PROMPT} \
          --num_shots ${SHOTS} \
          --train_batch_size ${SHOTS} \
          --dataset-config-file configs/datasets/${DATASET}.yaml \
          --config-file configs/trainers/PLOT/${CFG}.yaml \
          --output-dir ${DIR} \
          DATASET.SUBSAMPLE_CLASSES base \
          DATASET.USEALL False
        fi
      done
    done
  done
