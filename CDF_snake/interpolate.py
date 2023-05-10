import numpy as np
try:
    import cupy as cp
    CDF_SNAKE_GPU_AVAILABLE = True
except ImportError or ModuleNotFoundError:
    CDF_SNAKE_GPU_AVAILABLE = False

class CDFSnake:
    def __init__(self, grid_values, pdfs, normalise_cdfs=False, use_gpu=False, cache_cdfs=False) -> None:
        if use_gpu and CDF_SNAKE_GPU_AVAILABLE:
            xp = cp
        else:
            xp = np

        self.xp = xp
        self.grid_values = self.xp.asarray(grid_values)
        self.dx = self.xp.append(0, self.xp.diff(self.grid_values))
        self.pdfs = self.xp.asarray(pdfs)
        self.num_pdfs = self.pdfs.shape[1]
        self.arange = self.xp.arange(self.num_pdfs)
        self.max_grid_value = self.grid_values.max()

        self.grid_snake = (self.grid_values[:,None] + self.max_grid_value*self.arange[None,:]).T.flatten()
        self.construct_snake(normalise_cdfs, cache_cdfs)

    def construct_snake(self, normalise_cdfs, cache_cdfs):
        cdfs = self.xp.nan_to_num(self.xp.cumsum(self.pdfs, axis=0)*self.dx[:,None])
        if normalise_cdfs:
            cdfs /= cdfs.max(axis=0)
        if cache_cdfs:
            self.cdfs = cdfs
        self.cdf_snake = (cdfs + self.arange[None,:]).T.flatten()
    
    def sample_snake(self):
        self.uniform_samples = self.xp.random.uniform(0,1, size=self.num_pdfs)
        self.random_sample_snake = self.uniform_samples + self.arange
        
        inverse_samples = self.xp.interp(self.random_sample_snake, self.cdf_snake, self.grid_snake) - self.max_grid_value*self.arange
        return inverse_samples
