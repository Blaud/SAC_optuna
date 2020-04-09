import numpy as np

import optuna
from optuna import distributions
from optuna.samplers import BaseSampler
from optuna import structs

from test_functions import TestFunc
from inform_tf import TEST_FUNC_1
tf = TestFunc(TEST_FUNC_1)

class AvgCoordsSampler(BaseSampler):
    def __init__(self, temperature=100, cooldown_factor=10, neighbor_range_factor=2, seed=None):
        self._rng = np.random.RandomState(seed)
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._temperature = temperature
        self.cooldown_factor = cooldown_factor
        self.neighbor_range_factor = neighbor_range_factor
        self._current_trial = None

    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}

        prev_trial = self._get_last_complete_trial(study)

        if self._rng.uniform(0, 1) <= self._transition_probability(study, prev_trial):
            self._current_trial = prev_trial

        params = self._sample_neighbor_params(search_space)

        self._temperature *= self.cooldown_factor

        return params

    def _sample_neighbor_params(self, search_space):

        params = {}
        for param_name, param_distribution in search_space.items():
            if isinstance(param_distribution, distributions.UniformDistribution):
                current_value = self._current_trial.params[param_name]
                width = (param_distribution.high -
                         param_distribution.low) * self.neighbor_range_factor
                neighbor_low = max(current_value - width, param_distribution.low)
                neighbor_high = min(current_value + width, param_distribution.high)
                params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)
            else:
                if isinstance(param_distribution, distributions.IntUniformDistribution):
                    current_value = self._current_trial.params[param_name]
                    width = (param_distribution.high -
                             param_distribution.low) * self.neighbor_range_factor
                    neighbor_low = max(current_value - width, param_distribution.low)
                    neighbor_high = min(current_value + width, param_distribution.high)
                    params[param_name] = round(self._rng.uniform(neighbor_low, neighbor_high))
                else:
                    raise NotImplementedError(
                        'Unsupported distribution {}.'.format(param_distribution))

        return params

    def _transition_probability(self, study, prev_trial):
        if self._current_trial is None:
            return 1.0

        prev_value = prev_trial.value
        current_value = self._current_trial.value

        if study.direction == structs.StudyDirection.MINIMIZE and prev_value <= current_value:
            return 1.0
        elif study.direction == structs.StudyDirection.MAXIMIZE and prev_value >= current_value:
            return 1.0

        return np.exp(-abs(current_value - prev_value) / self._temperature)

    @staticmethod
    def _get_last_complete_trial(study):
        complete_trials = [t for t in study.trials if t.state == structs.TrialState.COMPLETE]
        return complete_trials[-1]

    def sample_independent(self, study, trial, param_name, param_distribution):

        return self._independent_sampler.sample_independent(study, trial, param_name,
                                                            param_distribution)



import pandas as pd
import numpy as np
df = pd.DataFrame({"1": [10, 4, 9, None, -7, 33],
                   "2": [None, 3, 16, 50, None, 26],
                   "3": [24, 10, None, 3, 19, 45],
                   "4": [22, -1, None, None, 22, 19],
                   "5": [8, -1, 31, None, 0, 1],
                   "6": [0, 10, 11, None, 12, 1]})

df2 = pd.DataFrame({"1": [43, 4, 9, None, -7, 3],
                   "2": [None, 3, 16, 50, 0, 26],
                   "3": [21, 10, None, 3, 19, 45],
                   "4": [22, -9, None, 0, 43, 1],
                   "5": [8, -1, 31, 8, 0, 1],
                   "6": [0, 10, 9, None, 12, 1]})

df = df.interpolate(method ='linear', limit_direction ='forward').interpolate(method ='linear', limit_direction ='backward').to_numpy()
df2 = df2.interpolate(method ='linear', limit_direction ='forward').interpolate(method ='linear', limit_direction ='backward').to_numpy()
print(df, df2)

def objective(trial):
    x = trial.suggest_int('x', 0, 5)
    y = trial.suggest_int('y', 0, 5)
    optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])
    print(optimizer)
    if optimizer == 'MomentumSGD':
        return df[x,y]
    else:
        return df2[x, y]


# def objective(trial):
#     return tf.get_value([trial.suggest_uniform('x1', -10, 10), trial.suggest_uniform('x2', -10, 10)])




if __name__ == '__main__':

    sampler = AvgCoordsSampler()
    # study = optuna.create_study(sampler=sampler)
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print('    {}: {}'.format(key, value))
