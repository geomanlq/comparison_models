from pathlib import Path
import numpy as np
from scipy.io import loadmat


class Dataset:
    """Imputation dataset.
    
    Available Datasets
    ------------------
    1. guangzhou
    2. t4c-london
    
    """
    def __init__(self, name, root="data/"):
        self.name = name
        self.root = Path(root)
        self.tensor = self._load_tensor()
        
    def _load_tensor(self):
        if self.name == "guangzhou":
            filename = self.root / self.name / "tensor.mat"
            tensor = loadmat(filename)["tensor"]
            dim = tensor.shape
            n_time = dim[1] * dim[2]
            good_idx = np.sum(tensor.reshape(dim[0], -1) == 0, axis=1) <= 0.05 * n_time
            tensor = tensor[good_idx]
            return tensor
        elif "t4c" in self.name:
            city = self.name.split("-")[1]
            filename = self.root / "t4c-s" / f"{city}_tensor.npy"
            tensor = np.load(filename)
            tensor = tensor[:, :15 * 7]
            return tensor
        else:
            raise ValueError(f"{self.name} is not available.")
            
    @property
    def shape(self):
        return self.tensor.shape


if __name__ == "__main__":
    gz = Dataset("guangzhou").tensor
    print(gz.shape)
    london = Dataset("t4c-london")
    print(london.shape)
