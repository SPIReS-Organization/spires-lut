# Pulled from the ISOFIT repo for convenience so the full install would not be needed.
# https://github.com/isofit/isofit

from typing import List
import numpy as np


Cache = {"stats": {}}

class VectorInterpolator:
    """Linear look up table interpolator.  Support linear interpolation through radial space by expanding the look
    up tables with sin and cos dimensions.

    Args:
        grid_input: list of lists of floats, indicating the gridpoint elements in each grid dimension
        data_input: n dimensional array of radiative transfer engine outputs (each dimension size corresponds to the
                    given grid_input list length, with the last dimensions equal to the number of sensor channels)
        version: version to use: 'rg' for scipy RegularGridInterpolator, 'mlg' for multilinear grid interpolator
    """

    def __init__(
        self,
        grid_input: List[List[float]],
        data_input: np.array,
        version="mlg",
    ):
        # Determine if this a singular unique value, if so just return that directly
        val = data_input[(0,) * data_input.ndim]
        if np.isnan(val) and np.isnan(data_input).all() or np.all(data_input == val):
            self.method = -1
            self.value = val
            return

        self.single_point_data = None

        # Lists and arrays are mutable, so copy first
        grid = grid_input.copy()
        data = data_input.copy()

        # Check if we are using a single grid point. If so, store the grid input.
        if np.prod(list(map(len, grid))) == 1:
            self.single_point_data = data
        self.n = data.shape[-1]

        # RegularGrid
        if version == "rg":
            raise ValueError("Not supported.")

        # Multilinear Grid
        elif version == "mlg":
            self.method = 2

            # None to disable, 0 for unlimited, negatives == 1
            self.cache_size = 1

            self.gridtuples = [np.array(t) for t in grid]
            self.gridarrays = data
            self.binwidth = [
                t[1:] - t[:-1] for t in self.gridtuples
            ]  # binwidth arrays for each dimension
            self.maxbaseinds = np.array([len(t) - 1 for t in self.gridtuples])

        else:
            raise AttributeError(f"Unknown interpolator version: {version!r}")

    def _interpolate(self, points):
        """
        Supports style 'rg'
        """
        # If we only have one point, we can't do any interpolation, so just
        # return the original data.
        if self.single_point_data is not None:
            return self.single_point_data

        x = np.zeros((self.n, len(points) + 1))
        x[:, :-1] = points

        # This last dimension is always an integer so no
        # interpolation is performed. This is done only
        # for performance reasons.
        x[:, -1] = np.arange(self.n)
        res = self.itp(x)

        return res

    def _lookup(self, i, point):
        """
        Calculates the slicing for the cube in _multilinear_grid
        """
        j = np.searchsorted(self.gridtuples[i][:-1], point) - 1

        # Bounds functions
        lower = lambda: max(min(self.maxbaseinds[i], j), 0)
        upper = lambda: max(min(self.maxbaseinds[i] + 2, j + 2), 2)

        if point >= self.gridtuples[i][-1]:
            return None, upper() - 1
        elif point <= self.gridtuples[i][0]:
            return None, lower()
        else:
            delta = (point - self.gridtuples[i][j]) / self.binwidth[i][j]
            return delta, slice(lower(), upper())

    def _multilinear_grid(self, points):
        """
        Cached version of Jouni's implementation

        Args:
            points: The point being interpolated. If at the limit, the extremal value in
                    the grid is returned.

        Returns:
            cube: np.ndarray
        """
        deltas = [None] * points.size
        idxs = [None] * points.size

        for i, point in enumerate(points):
            if self.cache_size is not None:
                cache = Cache.setdefault(i, {})
                stats = Cache["stats"].setdefault(i, {"hit": 0, "miss": 0})

                if point in cache:
                    data = cache[point]
                    stats["hit"] += 1
                else:
                    # Simple FIFO
                    if self.cache_size and len(cache) >= self.cache_size:
                        cache.pop(list(cache)[0])

                    data = self._lookup(i, point)
                    cache[point] = data
                    stats["miss"] += 1
            else:
                data = self._lookup(i, point)

            deltas[i], idxs[i] = data

        cube = np.copy(self.gridarrays[tuple(idxs)], order="A")

        # Only linear interpolate sliced dimensions
        for i, idx in enumerate(idxs):
            if isinstance(idx, slice):
                cube[0] *= 1 - deltas[i]
                cube[1] *= deltas[i]
                cube[0] += cube[1]
                cube = cube[0]

        return cube

    def __call__(self, *args, **kwargs):
        """
        Passes args to the appropriate interpolation method defined by the version at
        object init.
        """
        if self.method == -1:
            return self.value
        elif self.method == 1:
            return self._interpolate(*args, **kwargs)
        elif self.method == 2:
            return self._multilinear_grid(*args, **kwargs)

