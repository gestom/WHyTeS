import numpy as np
from collections import defaultdict
from time import clock



class Fremen():
    """
    basic FreMEn to find most influential periodicity, call
    chosen_period(T, time_frame_sums, time_frame_freqs, longest, shortest, W, ES):
    it returns the most influential period in the timeseries, where timeseries are
        the residues between reality and model
    where
    input: T numpy array Nx1, time positions of measured values
           time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           time_frame_freqs numpy array shape_of_grid[0]x1, sum of
                                                            frequencies(stat)
                                                            in model
                                                            over every
                                                            timeframe
           W numpy array Lx1, sequence of reasonable frequencies
           ES float64, squared sum of squares of residues from the last
                       iteration
    and
    output: P float64, length of the most influential frequency in default
            units
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
            ES_new float64, squared sum of squares of residues from
                            this iteration
            dES float64, difference between last and new error

    for the creation of a list of reasonable frequencies call
    build_frequencies(longest, shortest):
    where
    longest - float, legth of the longest wanted period in default units,
            - usualy four weeks
    shortest - float, legth of the shortest wanted period in default units,
             - usualy one hour.
    It is necessary to understand what periodicities you are looking for (or what
        periodicities you think are the most influential)
    """
    def __init__(self, #data_path='../data/two_weeks_days_nights_weekends.txt', 
                 structure,
                 minutes=0, hours=24, days=14, weeks=0):
        self.used = set(structure[2])
        #self.path = data_path
        self.Ps, self.W_parted = self._build_frequencies(minutes, hours, days, weeks)
        #self.T, self.values = self._create_time_series()


    #def _create_time_series(self):
    #    """
    #    """
    #    timeseries = np.loadtxt(self.path)[:, [0, -1]]
    #    multiplied = timeseries[timeseries[:, -1] > 1.0, :]
    #    timeseries = timeseries[timeseries[:, -1] <= 1.0, :]
    #    # !!! training data values must be whole numbers !!!
    #    if len(multiplied) > 1:
    #        print('training data must be expanded')
    #        new_time_series = []
    #        for time, value in multiplied:
    #            for j in xrange(int(np.ceil(value))):
    #                new_time_series.append([time, value])
    #        print(np.shape(np.array(new_time_series)))
    #        timeseries = np.r_[timeseries, np.array(new_time_series)]
    #    return timeseries[:, 0], timeseries[:, 1]


    def find_prominent(self, path, model):
        S, T = self._get_timeseries(model, path)
        P = self._chosen_period(T, S)
        return P


    def _get_timeseries(self, model, path):
        X, target, T = model.transform_data(path, for_fremen=True)
        y = model.predict(X)
        S = target - y
        #T = np.loadtxt(path)[:, 0]
        return S, T


    def _chosen_period(self, T, S):
        """
        input: T numpy array Nx1, time positions of measured values
               time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                                over every
                                                                timeframe
               time_frame_freqs numpy array shape_of_grid[0]x1, sum of
                                                                frequencies(stat)
                                                                over every
                                                                timeframe
               W numpy array Lx1, sequence of reasonable frequencies
               ES float64, squared sum of squares of residues from the last
                           iteration
        output: P float64, length of the most influential frequency in default
                units
                W numpy array Lx1, sequence of reasonable frequencies without
                                   the chosen one
                ES_new float64, squared sum of squares of residues from
                                this iteration
                dES float64, difference between last and new error
        uses: np.sum(), np.max(), np.absolute()
              complex_numbers_batch(), max_influence()
        objective: to choose the most influencing period in the timeseries, where
                   timeseries are the residues between reality and model
        """
        # originally: S = (time_frame_sums - time_frame_freqs)[valid_timesteps]
        Gs = self._complex_numbers_batch(T, S)
        P = self._max_influence(Gs)
        # power spectral density ???
        #sum_of_amplitudes =  np.sum(np.absolute(G) ** 2)
        #sum_of_amplitudes = np.sum(np.absolute(G))
        return P#, sum_of_amplitudes


    def _complex_numbers_batch(self, T, S):
        """
        input: T numpy array Nx1, time positions of measured values
               S numpy array Nx1, sequence of measured values
               W numpy array Lx1, sequence of reasonable frequencies
        output: G numpy array Lx1, sequence of complex numbers corresponding
                to the frequencies from W
        uses: np.e, np.newaxis, np.pi, np.mean()
        objective: to find sparse(?) frequency spectrum of the sequence S
        """
        # old version
        #Gs = S * (np.e ** (W[:, np.newaxis] * T * (-1j) * np.pi * 2))
        #print('mam Gs')
        #G = np.mean(Gs, axis=1)
        # sum by times version
        #t, s = sum_by_times(T, S)
        #G = []
        #for i in xrange(len(W)):
        #    Gs = s * (np.e ** (W[i] * t * (-1j) * np.pi * 2))
        #    G.append(np.mean(Gs))
        #G = np.array(G)
        #return G
        # without sum by times
        #print('fuck the RAM')
        #G = []
        #for w in W:
        #    start = clock()
        #    G.append(np.mean(S * (np.e ** (w * T * (-1j) * np.pi * 2))))
        #    finish = clock()
        #    print(str(1/w) + str(finish-start))
        #return np.array(G)
        # all in the RAM
        #return np.mean(S * (np.e ** (W[:, np.newaxis] * T * (-1j) * np.pi * 2)), axis=1)
        Gs = np.array([])
        for W in self.W_parted:
            if len(W) > 0:
                try:
                    Gs_part = np.mean(S * (np.e ** (W[:, np.newaxis] * T * (-1j) * np.pi * 2)), axis=1)
                except MemoryError:
                    print('this will probably take a longer time')
                    Gs_part = []
                    for w in W:
                        Gs_part.append(np.mean(S * (np.e ** (w * T * (-1j) * np.pi * 2))))
                    Gs_part = np.array(Gs_part)
                Gs = np.r_[Gs, Gs_part]
        return Gs


    def _max_influence(self, G):
        """
        input: W numpy array Lx1, sequence of reasonable frequencies
               G numpy array Lx1, sequence of complex numbers corresponding
                                  to the frequencies from W
        output: P float64, length of the most influential frequency in default
                           units
                W numpy array Lx1, sequence of reasonable frequencies without
                                   the chosen one
        uses: np.absolute(), np.argmax(), np.float64(),np.array()
        objective: to find length of the most influential periodicity in default
                   units and return changed list of frequencies
        """
        #print(np.absolute(G))
        #maximum_position = np.argmax(np.absolute(G[1:])) + 1
        maximum_position = np.argmax(np.absolute(G))  # no zero length included
        influential_frequency_length = self.Ps[maximum_position]
        # not sure if it is necessary now
        #if influential_frequency == 0 or np.isnan(np.max(np.absolute(G))):
        #    print('problems in fremen.max_influence')
        #    P = np.float64(0.0)
        #else:
        #    P = 1 / influential_frequency
        #return P
        return influential_frequency_length


    #def sum_by_times(T, S):
    #    """
    #    """
    #    TS = np.c_[T, S]
    #    ts_dict = defaultdict(float)
    #    for t, s in TS:
    #        ts_dict[t] += s
    #    ts_array = np.array(ts_dict.items())
    #    return ts_array[:, 0], ts_array[:, 1]


    def _build_frequencies(self, minutes, hours, days, weeks):  # should be part of initialization of learning
        """
        input: longest float, legth of the longest wanted period in default
                              units
               shortest float, legth of the shortest wanted period
                               in default units
        output: W numpy array Lx1, sequence of frequencies
        uses: np.arange()
        objective: to find frequencies w_0 to w_k
        """
        if minutes > 0:
            p_minutes = ((np.arange(minutes - 1) + 1.0) * 60.0)
            use_now = []
            for val in p_minutes:
                if val not in self.used:
                    use_now.append(val)
            p_minutes = np.array(use_now)
            w_minutes = 1.0 / p_minutes
        else:
            p_minutes = np.array([])
            w_minutes = np.array([])
        if hours > 0:
            p_hours = ((np.arange(hours - 1) + 1.0) * 3600.0)
            use_now = []
            for val in p_hours:
                if val not in self.used:
                    use_now.append(val)
            p_hours = np.array(use_now)
            
            w_hours = 1.0 / p_hours
        else:
            p_hours = np.array([])
            w_hours = np.array([])
        if days > 0:
            p_days = ((np.arange(days - 1) + 1.0) * 86400.0)
            use_now = []
            for val in p_days:
                if val not in self.used:
                    use_now.append(val)
            p_days = np.array(use_now)
            w_days = 1.0 / p_days
        else:
            p_days = np.array([])
            w_days = np.array([])
        if weeks > 0:
            p_weeks = ((np.arange(weeks - 1) + 1.0) * 604800.0)
            use_now = []
            for val in p_weeks:
                if val not in self.used:
                    use_now.append(val)
            p_weeks = np.array(use_now)
            w_weeks = 1.0 / p_weeks
        else:
            p_weeks = np.array([])
            w_weeks = np.array([])
        #zero = np.array([0])
        Ps = np.r_[p_minutes, p_hours, p_days, p_weeks]
        W_parts = (w_minutes, w_hours, w_days, w_weeks)
        #print(Ps)
        return Ps, W_parts


