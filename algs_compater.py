"""
Optuna example that implements a user-defined relative sampler based on Simulated Annealing
algorithm. Please refer to https://en.wikipedia.org/wiki/Simulated_annealing for Simulated
Annealing itself.

Note that this implementation isn't intended to be used for production purposes and
has the following limitations:
- The sampler only supports `UniformDistribution` (i.e., `Trial.suggest_uniform` method).
- The implementation prioritizes simplicity over optimization efficiency.

You can run this example as follows:
    $ python simulated_annealing_sampler.py

"""

import numpy as np

import optuna
from optuna import distributions
from optuna.samplers import BaseSampler
from optuna import structs


class SimulatedAnnealingSampler(BaseSampler):
    def __init__(self, temperature=100, cooldown_factor=0.9, neighbor_range_factor=0.1, seed=None):
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
            # The relative search space is empty (it means this is the first trial of a study).
            return {}

        # The rest of this method is an implementation of Simulated Annealing (SA) algorithm.
        prev_trial = self._get_last_complete_trial(study)

        # Update the current state of SA if the transition is accepted.
        if self._rng.uniform(0, 1) <= self._transition_probability(study, prev_trial):
            self._current_trial = prev_trial

        # Pick a new neighbor (i.e., parameters).
        params = self._sample_neighbor_params(search_space)

        # Decrease the temperature.
        self._temperature *= self.cooldown_factor

        return params

    def _sample_neighbor_params(self, search_space):
        # Generate a sufficiently near neighbor (i.e., parameters).
        #
        # In this example, we define a sufficiently near neighbor as
        # `self.neighbor_range_factor * 100` percent region of the entire
        # search space centered on the current point.

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
                raise NotImplementedError(
                    'Unsupported distribution {}.'.format(param_distribution))

        return params

    def _transition_probability(self, study, prev_trial):
        if self._current_trial is None:
            return 1.0

        prev_value = prev_trial.value
        current_value = self._current_trial.value

        # `prev_trial` is always accepted if it has a better value than the current trial.
        if study.direction == structs.StudyDirection.MINIMIZE and prev_value <= current_value:
            return 1.0
        elif study.direction == structs.StudyDirection.MAXIMIZE and prev_value >= current_value:
            return 1.0

        # Calculate the probability of accepting `prev_trial` that has a worse value than
        # the current trial.
        return np.exp(-abs(current_value - prev_value) / self._temperature)

    @staticmethod
    def _get_last_complete_trial(study):
        complete_trials = [t for t in study.trials if t.state == structs.TrialState.COMPLETE]
        return complete_trials[-1]

    def sample_independent(self, study, trial, param_name, param_distribution):
        # In this example, this method is invoked only in the first trial of a study.
        # The parameters of the trial are sampled by using `RandomSampler` as follows.
        return self._independent_sampler.sample_independent(study, trial, param_name,
                                                            param_distribution)


# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial):
    x = trial.suggest_uniform('x', -100, 100)
    y = trial.suggest_uniform('y', -1, 1)
    return x**2 + y


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



if __name__ == '__main__':
    # Run optimization by using `SimulatedAnnealingSampler`.
    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=1000)

    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print('    {}: {}'.format(key, value))
