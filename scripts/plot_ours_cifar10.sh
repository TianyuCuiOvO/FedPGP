#!/bin/bash

cd ../

# custom config
DATA="DATA/"
MODEL=reparaFL
TRAINER=Repara
PRETRAINED=True
FEATURE=False
LOGITS2=False
OT=COT
TOP_PERCENT=1
EPS=0.01
THRESH=0.001
MAX_ITER=100
LR=0.001
GAMMA=1
USERS=100
FRAC=0.1
LOCAL_EPOCH=10
ROUND=150
NUM_PROMPT=1
#DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CTXINIT=False
IID=False
CSC=False  # class-specific context (False or True)
USEALL=True
BETA=0.3
#MU=0.5
TEMP=0.5
# SEED=1
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in cifar10
do
  for BOTTLENECK in 8
    do
      for PARTITION in noniid-labeldir
      do
        for MU in 0.5
        do
          for SEED in 0
          do
            DIR=output_cifar/${DATASET}_${PARTITION}_con_mu${MU}_temp${TEMP}_10_150/${MODEL}_${TRAINER}_${BOTTLENECK}neck/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}
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
              --mu ${MU} \
              --temp ${TEMP} \
              --bottleneck ${BOTTLENECK} \
              --feature ${FEATURE} \
              --OT ${OT} \
              --top_percent ${TOP_PERCENT} \
              --eps ${EPS} \
              --thresh ${THRESH} \
              --max_iter ${MAX_ITER} \
              --logits2 ${LOGITS2} \
              --gamma ${GAMMA} \
              --trainer ${TRAINER} \
              --round ${ROUND} \
              --local_epoch ${LOCAL_EPOCH} \
              --partition ${PARTITION} \
              --beta ${BETA} \
              --n_ctx ${NCTX} \
              --num_prompt ${NUM_PROMPT} \
              --dataset-config-file configs/datasets/${DATASET}.yaml \
              --config-file configs/trainers/PLOT/vit_b16.yaml \
              --output-dir ${DIR}
            fi
          done
        done
      done
    done
done