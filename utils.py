import gymnasium as gym
import np



def IncreasePositiveRewards(r):
    if r > 0:
        return r*1.5
    else:
        return r


def IncreaseNegativeRewards(r):
    if r < 0:
        return r*5
    else:
        return r


class NormalizeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.asarray(obs) / 255.0
    


class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=(96, 96, 5), low=0, high=255)

    def observation(self, obs):
        return np.transpose(np.asarray(obs), (1, 2, 0))
    


class ConcatenateFrames(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=(96, 96, 9), low=0, high=255)

    def observation(self, obs):
        # From (3, 96, 96, 3) -> ()
        return np.asarray(obs).reshape(96, 96, 9)