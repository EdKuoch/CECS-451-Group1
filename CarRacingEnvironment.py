import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack, TimeLimit, GrayScaleObservation
from utils import NormalizeImage, TransposeObservation
import sys
from PIL import Image as im



class CarRacingEnvironment:
    env = None
    """Gym environment variable"""


    def __init__(
            self, 
            agent = None,
            num_episodes = 0,
            num_input_frame_stack = 0, 
            num_sticky_actions = 0, 
            domain_randomize = False,
            render_mode = None,
            max_episode_steps = 1000,
            update_target_model_frequency = 3,
            save_training_frequency = 25,
            training_batch_size = 32,
            train = True
    ):
        """
        A Box2D car racing reinforcement learning environment.

        @agent: The agent to use.
        @num_episodes: How many episodes (complete games) to run.
        @num_input_frame_stack: How many frames to send to the agent as one input stack.
        @num_sticky_actions: How many frames an action will apply for. Actions are applied to the n frames following the input frames.
        @domain_randomize: True=Set the objects to have different colors each time.
        @render_mode: 'human'=Watch the model play, default='None'
        @max_episode_steps: Max steps in an episode before it hits a time limit
        """
        
        self.agent = agent
        self.num_episodes = num_episodes
        self.num_input_frame_stack = num_input_frame_stack
        self.num_sticky_actions = num_sticky_actions
        self.domain_randomize = domain_randomize
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.update_target_model_frequency = update_target_model_frequency
        self.save_training_frequency = save_training_frequency
        self.training_batch_size = training_batch_size
        self.train = train

        self.env = gym.make("CarRacing-v2", domain_randomize=self.domain_randomize, render_mode=self.render_mode)
        self.env = GrayScaleObservation(self.env, keep_dim=True)
        self.env = NormalizeImage(self.env)
        # self.env = FrameStack(self.env, self.num_input_frame_stack) 
        # self.env = TransposeObservation(self.env)
        self.env = TimeLimit(self.env, self.max_episode_steps)
        

        # Returns the last n frames as the observation -> (3, 27648) numpy array
        # self.env = FrameStack(self.env, self.num_input_frame_stack) 
        # self.env = ConcatenateFrames(self.env)
        self.agent.initialize_models(self.env.observation_space.shape)


    def run(self):
        if self.train: 
            self.train_agent()
        else: 
            self.evaluate_agent()


    def train_agent(self):
        for e in range(self.num_episodes):
            # Reset the game for every episode.
            state, info = self.env.reset(options={"randomize": self.domain_randomize})

            total_episode_reward = 0
            action_frame_counter = 0
            negative_reward_counter = 0
            global_step_counter = 0

            while True:
                action_frame_counter += 1
                global_step_counter +=1
                print('Episode:', e, '| Steps:', global_step_counter, '| Total Reward:', total_episode_reward)

                # Input the state into the policy to get back the policy's action.
                action = self.agent.policy(self.env, state)

                # Repeat the action for n frames.
                current_reward = 0
                for _ in range(self.num_sticky_actions):
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    global_step_counter +=1
                    print('Episode:', e, '| Steps:', global_step_counter, '| Total Reward:', total_episode_reward)
                    current_reward += reward

                    if terminated or truncated:
                        break
                
                # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
                negative_reward_counter = negative_reward_counter + 1 if global_step_counter > 10 and reward < 0 else 0
                total_episode_reward += reward

                # Add the experience to the experience replay buffer.
                self.agent.add_to_experience_replay_buffer(state, action, current_reward, next_state, terminated)

                # Transition to the next state
                state = next_state

                if terminated or negative_reward_counter >= 25 or total_episode_reward < 0 and global_step_counter == 100:
                    print(f"Episode {e}: {total_episode_reward}")
                    break
                if len(self.agent.experience_replay_buffer) > self.training_batch_size:
                    self.agent.train_from_experience_replay_buffer(self.training_batch_size)

            if e % self.update_target_model_frequency == 0:
                self.agent.update_frozen_model()

            if e % self.save_training_frequency == 0:
                self.agent.save_model_weights('./save/trial_{}.h5'.format(e))

    
    def evaluate_agent(self):
        self.agent.load_model_weights('./save/trial_600.h5')

        for e in range(self.num_episodes):
            # Reset the game for every episode.
            state, info = self.env.reset(options={"randomize": self.domain_randomize})

            total_episode_reward = 0
            step_counter = 0
            

            while True:
                print(step_counter)
                step_counter += 1

                # Input the state into the policy to get back the policy's action.
                action = self.agent.policy(self.env, state)

                # next_state, reward, terminated, truncated, info = self.env.step(action)

                # Repeat the action for n frames.
                current_reward = 0
                for _ in range(self.num_sticky_actions):
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    current_reward += reward

                    if terminated or truncated:
                        break
                
                total_episode_reward += reward

                # Transition to the next state
                state = next_state

                if terminated:
                    print(f"Episode {e}: {total_episode_reward}")
                    break