from typing import Optional, Union

import gouda
import numpy as np
import numpy.typing as npt
import opensimplex

RANDOM_TYPE = Optional[Union[int, npt.ArrayLike, np.random.Generator]]


def simplex3d(shape, res, seed=None):
    # uses opensimplex - https://github.com/lmas/opensimplex
    if seed is None or isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)

    (shape, res), _ = gouda.match_len(shape, res, count=3)
    xyz = [np.linspace(0, s, s, endpoint=False) / r for s, r in zip(shape, res)]
    if random is not None:
        xyz = [x + random.integers(-s, s) for x, s in zip(xyz, shape)]
    noise = opensimplex.noise3array(*xyz)
    return noise


def generate_fractal_noise_3d(shape: tuple[int, int, int], res: int, octaves: int = 1, persistence: float = 0.5, lacunarity: int = 2, seed: RANDOM_TYPE = None) -> np.ndarray:
    """Generate a 3D numpy array of fractal simplex noise

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the output noise array
    res : int
        The number of periods of noise to generate along each axis
    octaves : int, optional
        The number of octaves in the noise, by default 1
    persistence : float, optional
        The scaling factor between two octaves, by default 0.5
    lacunarity : int, optional
        The frequency factor between two octaves, by default 2
    seed : RANDOM_TYPE, optional
        The random seed or generator, by default None

    Returns
    -------
    np.ndarray
        The array of fractal simplex noise

    NOTE
    ----
    Great blog post on perlin/simplex noise: https://leatherbee.org/index.php/2018/10/24/perlin-and-simplex-noise/
    """
    (shape, res), _ = gouda.match_len(shape, res, count=3)
    if isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)

    noise = np.zeros(shape, dtype=np.float32)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        local_res = tuple([frequency * r for r in res])
        noise += amplitude * simplex3d(shape, local_res, seed=random)
        frequency *= lacunarity
        amplitude *= persistence
    return noise
