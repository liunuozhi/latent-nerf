import pyrallis
import torch
from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.training.trainer import Trainer
from src.latent_nerf.models.network_grid import NeRFNetwork


class GUITrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nerf = self.init_nerf().eval()
        self.load_checkpoint('./weights.pth', model_only=True)
        self.dataloaders = self.init_dataloaders()

    def init_nerf(self):
        return NeRFNetwork(self.cfg.render).to(self.device)

    def init_diffusion(self):
        return None

    def calc_text_embeddings(self):
        return None


@pyrallis.wrap()
def main(cfg: TrainConfig):
    trainer = GUITrainer(cfg)
    trainer.full_eval()
    return 0


if __name__ == '__main__':
    main()
