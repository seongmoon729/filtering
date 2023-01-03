import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.cuda.indices))
    os.environ['DETECTRON2_DATASETS'] = str(cfg.data_path.detectron2)

    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.command.mode == 'train':
        import train
        train.train(cfg)
    else:
        import evaluate
        print(cfg.command.codec.qps)
        # evaluate.evaluate_for_object_detection(cfg)


if __name__ == '__main__':
    main()