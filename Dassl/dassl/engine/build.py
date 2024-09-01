from Dassl.dassl.utils import Registry, check_availability
# from trainers.clip import CLIP
from trainers.promptfl import PromptFL, Baseline
from trainers.plot import PLOT
from trainers.coop2 import CoOp2
from trainers.repara import Repara
from trainers.f_f_con import F_f_con
from trainers.f_f_close import F_f_close
from trainers.repara_close import Repara_close
from trainers.repara_push import Repara_push
from trainers.maple import MaPLe
from trainers.cocoop import CoCoOp
from trainers.fedpgp import FedPGP
TRAINER_REGISTRY = Registry("TRAINER")
# TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(PromptFL)
TRAINER_REGISTRY.register(Baseline)
TRAINER_REGISTRY.register(PLOT)
TRAINER_REGISTRY.register(CoOp2)
TRAINER_REGISTRY.register(Repara)
TRAINER_REGISTRY.register(F_f_con)
TRAINER_REGISTRY.register(F_f_close)
TRAINER_REGISTRY.register(Repara_close)
TRAINER_REGISTRY.register(Repara_push)
TRAINER_REGISTRY.register(MaPLe)
TRAINER_REGISTRY.register(CoCoOp)
TRAINER_REGISTRY.register(FedPGP)


def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    # print("avai_trainers",avai_trainers)
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
