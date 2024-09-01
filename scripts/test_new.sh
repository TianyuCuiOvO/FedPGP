#!/bin/bash

cd ../

# custom config
DATA="DATA/"
MODEL=test
TRAINER=PLOT
OT=COT
TOP_PERCENT=0.80
EPS=0.1
THRESH=0.01
MAX_ITER=100
PRETRAINED=True
LR=0.001
GAMMA=1
LOGITS2=False
USERS=10
FRAC=1
ROUND=1
LOCAL_EPOCH=1
NUM_PROMPT=2
#DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
IID=False
CSC=False  # class-specific context (False or True)
USEALL=True
TEMP=0.5
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in oxford_flowers
do
  for SHOTS in 8 16
  do
    for SEED in 1
    do
#               DIR=output_base/${DATASET}/reparaFL_${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
#                DIR=output_base/${DATASET}_mu${MU}/reparaFL_${TRAINER}_${BOTTLENECK}neck/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_25round_seed${SEED}
      DIR=output_base/${DATASET}/FedOTP_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_25round_seed${SEED}
      if [ -d "$DIR" ]; then
        echo "The results exist at ${DIR} (go test new acc now)"
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
        DATASET.SUBSAMPLE_CLASSES new \
        DATASET.USEALL True
      fi
    done
  done
done
