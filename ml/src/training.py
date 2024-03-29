import argparse
import os

import torch
import torch.utils.data
from omegaconf import OmegaConf, MISSING

from ml.src.trainingData import TrainingData
from ml.src.phase import Phase
import wandb
from tqdm import tqdm
from ml.src.configuration.configuration import (
    ConfigStore,
    Config,
)
from ml.src.trainingUtils import (
    create_dataloaders,
    create_model,
    create_loss,
    create_optimizer,
    create_scheduler,
    train_epoch,
    val_epoch,
    print_metrics,
)


def train():
    with wandb.init(
        entity="pg-pug-tennis-betting",
        project="tennis-betting",
        mode=ConfigStore.cfg.wandb_mode,
    ) as run:

        ConfigStore.handle_wandb(run)

        cfg: Config = ConfigStore.cfg

        dataloaders = create_dataloaders(cfg)

        model = create_model(cfg)
        if torch.cuda.is_available():
            model.cuda(cfg.training.gpu)

        loss = create_loss(cfg)
        optimizer = create_optimizer(cfg, model)
        scheduler = create_scheduler(cfg, optimizer)

        training = TrainingData(cfg, dataloaders, model, loss, optimizer, scheduler)

        for _ in tqdm(range(cfg.training.epochs)):
            training_metrics = train_epoch(training)
            print_metrics(
                training_metrics,
                len(training.dataloaders[Phase.TRAIN]),
                wandb,
                "Training",
            )

            val_metrics = val_epoch(training)
            print_metrics(val_metrics, len(dataloaders[Phase.VAL]), wandb, "val")

            training.update_best(val_metrics["loss"])
        # load best model
        training.model.load_state_dict(training.best_model_weights)
        # Make test run
        test_metrics = val_epoch(training, Phase.TEST)
        print_metrics(
            test_metrics, len(training.dataloaders[Phase.TEST]), wandb, "Test"
        )
        # Save model TODO: add better model name and consider using artifact system
        torch.save(model, os.path.join(wandb.run.dir, "model.pt"))


def main():
    parser = argparse.ArgumentParser(prog="Tennis betting")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file path, absolute or relative to ml/config",
    )
    args = parser.parse_args()

    ConfigStore.load(args.config)
    cfg = ConfigStore.cfg
    if cfg.sweep is None:
        train()
    else:
        sweep_cfg = OmegaConf.to_container(ConfigStore.cfg.sweep, resolve=True)
        if ConfigStore.cfg.name is not None and ConfigStore.cfg.name != "":
            sweep_cfg["name"] = ConfigStore.cfg.name

        assert (
            ConfigStore.cfg.wandb_mode == "online"
        ), "Sweep only supported in online mode"

        sweep_id = wandb.sweep(
            sweep_cfg, entity="pg-pug-tennis-betting", project="tennis-betting"
        )
        wandb.agent(sweep_id, train)


if __name__ == "__main__":
    main()
