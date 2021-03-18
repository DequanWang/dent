import logging
import os.path as osp

from autoattack import AutoAttack
from robustbench.data import load_cifar10
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from dent import Dent
from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    assert cfg.CORRUPTION.DATASET == 'cifar10'
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       'cifar10', ThreatModel.Linf).cuda()
    if cfg.MODEL.ADAPTATION == "dent":
        assert cfg.MODEL.EPISODIC
        dent_model = Dent(base_model, cfg.OPTIM)
        logger.info(dent_model.model)
    x_test, y_test = load_cifar10(cfg.CORRUPTION.NUM_EX, cfg.DATA_DIR)
    x_test, y_test = x_test.cuda(), y_test.cuda()
    adversary = AutoAttack(
        dent_model, norm='Linf', eps=8./255., version='standard',
        log_path=osp.join(cfg.SAVE_DIR, cfg.LOG_DEST))
    adversary.run_standard_evaluation(
        x_test, y_test, bs=cfg.TEST.BATCH_SIZE)


if __name__ == '__main__':
    evaluate('"CIFAR-10 AutoAttack Linf 8/255.')
