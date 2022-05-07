from learning import CartpoleAI


def main():
    agent = CartpoleAI()
    # env.render()
    # agent.render_random_games(num_games=10)
    agent.train_model()
    agent.render_episode()


if __name__ == '__main__':
    main()
