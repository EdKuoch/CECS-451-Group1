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
        num_input_frame_stack   = 3, 
        num_sticky_actions      = 3, 
        render_mode             = "human",
        train                   = True,
    )

    car_racing_environment.run()



if __name__ == "__main__":
    main()