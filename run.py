import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.cuda.indices))

    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.mode == 'train':
        os.environ['DETECTRON2_DATASETS'] = str(cfg.data_path.detectron2)
        import train
        train.train(cfg)
    else:
        import evaluate
        evaluate.evaluate(cfg)


if __name__ == '__main__':
    main()