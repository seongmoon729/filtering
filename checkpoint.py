from pathlib import Path
import torch


class Checkpoint:
    def __init__(self, path):
        self.base_path = Path(path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_name_template = "checkpoint_{}.pth"

    def save(self, network, step) -> None:
        ckpt_path = self.base_path / self._ckpt_name_template.format(step)
        target_objects = {'network': network.state_dict(), 'step': step}
        torch.save(target_objects, ckpt_path)

    def load(self, network, step) -> None:
        ckpt_path = self.base_path / self._ckpt_name_template.format(step)
        assert ckpt_path.exists(), f"There is no saved checkpoint for {step} steps."
        loaded_objects = torch.load(ckpt_path, map_location=network.device)
        network.load_state_dict(loaded_objects['network'])