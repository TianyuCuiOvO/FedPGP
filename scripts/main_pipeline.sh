#!/bin/bash

cd ...

# custom config
DATA="DATA/"
TRAINER=PromptFL
PRETRAINED=True
LR=0.001

#DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
IID=False
CSC=True  # class-specific context (False or True)
USEALL=False
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in caltech101 dtd
do
  for SHOTS in 2
  do
    for REPEATRATE in 0.0
    do
      for USERS in 10
      do
        for EPOCH in 5
        do
          for ROUND in 10
          do
            for SEED in 1
            do
              DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/pretrain_${PRETRAINED}/iid_${IID}_repeatrate_${REPEATRATE}/${USERS}_users/lr_${LR}/${EPOCH}epoch_${ROUND}round/seed${SEED}
              if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
              else
                python federated_main.py \
                --root ${DATA} \
                --seed ${SEED} \
                --model ${FedAvg} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                TRAINER.PROMPTFL.N_CTX ${NCTX} \
                TRAINER.PROMPTFL.CSC ${CSC} \
                TRAINER.PROMPTFL.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.USERS ${USERS} \
                DATASET.IID ${IID} \
                DATASET.REPEATRATE ${REPEATRATE} \
                OPTIM.MAX_EPOCH ${EPOCH} \
                OPTIM.ROUND ${ROUND}\
                OPTIM.LR ${LR}\
                MODEL.BACKBONE.PRETRAINED ${PRETRAINED}\
                DATASET.USEALL ${USEALL}
              fi
            done
          done
        done
      done
    done
  done
done

