from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from dataset import dataset_factory
from models import model_factory
from components import lr_scheduler_factory, optimizers_factory, logger_factory
from training import training_factory
from datetime import datetime
import torch
import numpy as np
import random
def set_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_training(cfg: DictConfig):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)

    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    return training.train()



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    all_fold_results = []
    fold_results = []
    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"


    for _ in range(cfg.repeat_time):
         best_result = model_training(cfg)
         all_fold_results.append(best_result)
         fold_results.append(best_result)

    print("\nResults for each fold:")
    for fold, result in enumerate(fold_results):
        seed_value = 142 + fold 
        set_seed(seed_value)
        print(f"\nResults for fold {fold + 1}:")
        for key, value in result.items():
            print(f"{key}: {value:.4f}")
            

    metrics = list(all_fold_results[0].keys())
    print(metrics)
    avg_results = {metric: np.mean([fold_result[metric] for fold_result in all_fold_results]) for metric in metrics}


    print("\nFinal Average Results Across All Folds:")
    for key, value in avg_results.items():
        print(f"{key}: {value:.4f}")



if __name__ == '__main__':
    main()
