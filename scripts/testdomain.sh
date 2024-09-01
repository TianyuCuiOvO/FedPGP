#!/bin/bash

cd ../

# custom config
DATA="DATA/"
MODEL=test_domain
TRAINER=Repara
OT=COT
TOP_PERCENT=0.80
EPS=0.01
THRESH=0.001
MAX_ITER=100
PRETRAINED=True
LR=0.001
GAMMA=1
LOGITS2=False
USERS=6
FRAC=1
ROUND=1
LOCAL_EPOCH=1
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
for DATASET in office
do
  for BOTTLENECK in 8
    do
      for MU in 0.5
        do
          for SHOTS in 64
          do
            for SEED in 0
            do
              DIR=output_office_self/${DATASET}_mu${MU}_0/reparaFL_domain_${TRAINER}_${BOTTLENECK}neck/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_1round_seed${SEED}
              if [ -d "$DIR" ]; then
                echo "The results exist at ${DIR} (go test acc now)"
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
                DATASET.SUBSAMPLE_CLASSES all \
                DATASET.USEALL True
              fi
            done
          done
        done
    done
done