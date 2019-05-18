
from sklearn.mixture import GaussianMixture
import transformation as tr
import transformation_with_dirs as tr_dir
#import frequencies as freq
#import probabilities as prob

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import matplotlib.patches as pat
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap




class DrawArrows:
    """
    """

    def __init__(self, dir_prob_model, human_freq_model,
                 edges_of_cell = np.array([1200.0, 0.5, 0.5])):
        self.freqs = human_freq_model
        self.probs = dir_prob_model
        self.edges_of_cell = edges_of_cell


    def make_png(self, time, counter):
        hour = int(time / 3600)
        minute = int((time % 3600)/60)
        XYUVC = self._create_grid_model(time)
        grey_red = LinearSegmentedColormap('GreyRed', self.create_cmap())
        plt.quiver(XYUVC[:, 0], XYUVC[:, 1], XYUVC[:, 2], XYUVC[:, 3], XYUVC[:, 4], angles='xy', scale_units='xy', scale=1.0, cmap=grey_red)
        #plt.quiver(XYUVC[:, 0], XYUVC[:, 1], XYUVC[:, 2], XYUVC[:, 3], angles='xy', scale_units='xy', scale=1.0)
        plt.title(str(hour) + ' hours, ' + str(minute) + ' minutes')
        plt.clim(-1.0, 1.0)
        clb = plt.colorbar()
        clb.ax.set_ylabel('decimal logarithm of predicted number of people', rotation=270)
        s = "{:04d}".format(counter)
        plt.savefig('sipky/' + s + '.png', dpi=200)
        plt.close()


    def _create_grid_model(self, time):
        table = self._create_coords(time)
        all_freqs = self.freqs.predict(tr.create_X(table.values, self.freqs.structure))
        all_dirs = self.probs.predict(tr_dir.create_X(table.values, self.probs.structure))
        table[4] = np.sin(table[3])
        table[3] = np.cos(table[3])
        table[5] = all_freqs
        table[6] = all_dirs
        a = table.groupby([1, 2]).apply(self._UVC_max).rename(columns={0:3, 1:4, 2:5})
        #a = table.groupby([1, 2]).apply(self._UVC_mean).rename(columns={0:3, 1:4, 2:5})
        a.reset_index(inplace=True)
        return a.values


    def _create_coords(self, time):
        X = np.arange(-7.0, 13.0, self.edges_of_cell[1])
        Y = np.arange(-7.0, 13.0, self.edges_of_cell[2])
        angles = np.arange(-np.pi, np.pi, np.pi/36.0)
        coords = self._cartesian_product(np.array([time]), X, Y, angles)
        return pd.DataFrame(coords)


    def _cartesian_product(self, *arrays):
        """
        coppied from:
        'https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of'+\
        '-x-and-y-array-points-into-single-array-of-2d-points'
        input: *arrays enumeration of central_points
        output: numpy array (central positions of cels of grid)
        uses: np.empty(),np.ix_(), np.reshape()
        objective: to perform cartesian product of values in columns
        """
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la],
                       dtype=arrays[0].dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)


    def _UVC_max(self, group):
        xyw = group[[3, 4, 5, 6]].values
        xyw = xyw[xyw[:, 3] == np.max(xyw[:, 3]), :]
        idx = np.random.choice(range(np.shape(xyw)[0]))
        xyw = pd.Series(xyw[idx, :])
        xyw[0] = xyw[0] * xyw[3]
        xyw[1] = xyw[1] * xyw[3]
        xyw[2] = np.log10(xyw[2])
        #if np.isnan(xyw[2]) or xyw[2] <= 0.1:
        #    xyw[2] = 0.0
        #elif xyw[2] >= 10:
        #    xyw[2] = 1.0
        #else:
        #    xyw[2] = (np.log10(xyw[2]) + 1.0) / 2.0
        #xyw[0] = xyw[0] * xyw[2] * xyw[3]
        #xyw[1] = xyw[1] * xyw[2] * xyw[3]
        #return xyw
        return xyw.iloc[:3]


    def _UVC_mean(self, group):
        group[3] = group[3] * group[6]
        group[4] = group[4] * group[6]
        xyw = group[[3, 4, 5]].values
        xyw = pd.Series(np.mean(xyw, axis=0))
        xyw[2] = np.log10(xyw[2] + 1)
        return xyw


    def create_cmap(self):
        cdict = {'red':   ((0.0, 1.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 1.0, 0.0)),

                 'green': ((0.0, 1.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),

                 'blue':  ((0.0, 1.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0))
                }
        return cdict

