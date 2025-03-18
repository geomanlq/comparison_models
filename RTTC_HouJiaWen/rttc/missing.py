import numpy as np


class MissingMask:
    """Generating missing masks.

    Parameters
    ----------
    dim : tuple
        Dimension of the input.
    seed : int
        Random seed.
        
    """

    def __init__(self, dim: tuple, seed: int = 33):
        self.dim = dim
        self.n_elements = np.prod(self.dim)
        self.seed = seed

    def generate_mask(self, missing_rate, missing_rate_add=None,
                      pattern="random", deterministic=True):
        if "sensor" in pattern:
            mask = self.generate_axis_mask(missing_rate, 0, deterministic)
        elif "day" in pattern:
            mask = self.generate_axis_mask(missing_rate, 1, deterministic)
        elif "time" in pattern:
            mask = self.generate_axis_mask(missing_rate, 2, deterministic)
        elif "blackout" in pattern:
            mask = self.generate_blackout_mask(missing_rate, deterministic)
        else:
            mask = self.generate_random_mask(missing_rate, deterministic)
        if "add" in pattern:
            if missing_rate_add is None:
                missing_rate_add = missing_rate
            mask = self.generate_additional_random_mask(missing_rate_add, mask, deterministic)
        return mask

    def generate_random_mask(self, missing_rate, deterministic=True):
        """Implementing random missing sampling.

        Parameters
        ----------
        missing_rate : float
            Missing rate of the entire tensor.
        deterministic : bool
            Whether use preset random generator.

        Returns
        -------
        mask : np.ndarray
            Mask with the same shape as `dim`.

        """
        rng = self._get_rng(deterministic)
        assert 0 <= missing_rate <= 1
        p = [missing_rate, 1 - missing_rate]
        mask = rng.choice([0, 1], size=self.dim, replace=True, p=p)
        return mask.astype(float)

    def generate_axis_mask(self, missing_rate, axis=0, deterministic=True):
        """Implementing missing sampling along the given axis. 
        Also known as fiber missing in literature. An entire fiber of values
        along the given axis are set as missing.

        Parameters
        ----------
        missing_rate : float
            Missing rate of the entire tensor.
        axis : int
            Axis to apply missing sampling.
        deterministic : bool
            Whether use preset random generator.

        Returns
        -------
        mask : np.ndarray
            Mask with the same shape as `dim`.

        """
        rng = self._get_rng(deterministic)
        assert 0 <= missing_rate <= 1
        p = [missing_rate, 1 - missing_rate]
        axis_dim = tuple(d if i != axis else 1 for i, d in enumerate(self.dim))
        mask = rng.choice([0, 1], size=axis_dim, replace=True, p=p)
        mask = np.repeat(mask, repeats=self.dim[axis], axis=axis)
        return mask.astype(float)
    
    def generate_blackout_mask(self, missing_rate, deterministic=True):
        """Implementing missing sampling when blackout happens across all sensors.
        It is a special case of axis missing, when axis- 1 & 2 missing happen at 
        the same time.

        Parameters
        ----------
        missing_rate : float
            Missing rate of the entire tensor.
        deterministic : bool
            Whether use preset random generator.

        Returns
        -------
        mask : np.ndarray
            Mask with the same shape as `dim`.

        """
        rng = self._get_rng(deterministic)
        assert 0 <= missing_rate <= 1
        p = [missing_rate, 1 - missing_rate]
        axis_dim = (1, self.dim[1], 1)
        mask = rng.choice([0, 1], size=axis_dim, replace=True, p=p)
        mask = np.repeat(mask, repeats=self.dim[0], axis=0)
        mask = np.repeat(mask, repeats=self.dim[2], axis=2)
        return mask.astype(float)

    def generate_additional_random_mask(self, missing_rate, base_mask, deterministic=True):
        """Implementing random missing sampling on the basis of other masks.

        Parameters
        ----------
        missing_rate : float
            Missing rate of the entire tensor.
        base_mask : np.ndarray
            Base mask 
        deterministic : bool
            Whether use preset random generator.

        Returns
        -------
        mask : np.ndarray
            Mask with the same shape as `dim`.

        """
        rng = self._get_rng(deterministic)
        assert 0 <= missing_rate <= 1

        n_remaining_elements = int(base_mask.sum())
        n_additional_missing = int(self.n_elements * missing_rate)
        pos = rng.choice(n_remaining_elements, size=n_additional_missing)
        idx = tuple(np.take(base_mask.nonzero(), pos, axis=1))
        mask = base_mask.copy()
        mask[idx] = 0
        return mask.astype(float)

    def _get_rng(self, deterministic=True):
        if deterministic:
            return np.random.default_rng(self.seed)
        else:
            return np.random.default_rng()
