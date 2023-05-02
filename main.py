from CarRacingEnvironment import CarRacingEnvironment
import DQN



def main():
    dqn_agent = DQN.DQNAgent(
        network             = DQN.networks.NetworkC,
        network_data_name   = "NetworkA_Test1",
    )

    car_racing_environment = CarRacingEnvironment(
        agent                   = dqn_agent,
        num_episodes            = 500,
        num_input_frame_stack   = 5, 
        num_sticky_actions      = 1, 
        render_mode             = "human",
        train                   = False,
    )

    car_racing_environment.run()



if __name__ == "__main__":
    main()