from collections import deque
import np
import random



class DQNAgent:   
    action_space = (
        (-1, 0, 1), # left, brake
        (-1, 1, 0), # left, gas
        (-1, 0, 0), # left, nothing
        (0, 0, 1), # center, brake
        (0, 1, 0), # center, gas
        (0, 0, 0), # center, nothing
        (1, 0, 1), # right, brake
        (1, 1, 0), # right, gas
        (1, 0, 0), # right, nothing
    )

    # action_space    = [
    #         (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
    #         (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
    #         (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
    #         (-1, 0,   0), (0, 0,   0), (1, 0,   0)
    #     ]


    def __init__(
            self, 
            network             = None, 
            network_data_name   = "",
            memory_size         = 1000,
            gamma               = 0.95,  # discount rate
            epsilon             = 0,   # exploration rate
            epsilon_min         = 0.02,
            epsilon_decay       = 0.999,
            learning_rate       = 0.0001
    ):
        """
        A DQN agent.

        @network: The deep learning network to use.
        @learning_rate: Learning rate of model.
        @network_data_name: Name of the trained network model, e.g. 'NetworkA_400episodes'. Data will be saved in 'foo/data/NetworkA_400episodes'.
        @train: True=train the model, False=use the existing trained model.
        """

        self.network                    = network()
        self.learning_rate              = learning_rate
        self.network_data_name          = network_data_name
        self.experience_replay_buffer   = deque(maxlen=memory_size)
        self.gamma                      = gamma
        self.epsilon                    = epsilon
        self.epsilon_min                = epsilon_min
        self.epsilon_decay              = epsilon_decay

        


    def initialize_models(self, input_shape):
        # print(input_shape)
        self.model = self.network.buildModel(input_shape, self.learning_rate, len(self.action_space))
        self.frozen_model = self.network.buildModel(input_shape, self.learning_rate, len(self.action_space))
        # print(self.model.summary())
    

    def update_frozen_model(self):
        self.frozen_model.set_weights(self.model.get_weights())


    def load_model_weights(self, name):
        self.model.load_weights(name)
        self.update_frozen_model()
        print('loaded weights')


    def save_model_weights(self, name):
        self.frozen_model.save_weights(name)


    def policy(self, env, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            action_index = np.argmax(act_values[0])
            print(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    
    def add_to_experience_replay_buffer(self, state, action, reward, next_state, terminated):
        self.experience_replay_buffer.append((state, self.action_space.index(action), reward, next_state, terminated))


    def train_from_experience_replay_buffer(self, batch_size):
        print("Epsilon:", self.epsilon)
        print("Training from experience replay buffer...")
        minibatch = random.sample(self.experience_replay_buffer, batch_size)
        train_x_states = []
        train_y_rewards = []

        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0, use_multiprocessing=True)[0]

            if done:
                target[action_index] = reward
            else:
                t = self.frozen_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                print(t)
                target[action_index] = reward + self.gamma * np.amax(t)

            train_x_states.append(state)
            train_y_rewards.append(target)

        print("Fitting model to samples from ERB...")

        self.model.fit(np.array(train_x_states), np.array(train_y_rewards), epochs=1, verbose=0, use_multiprocessing=True)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay