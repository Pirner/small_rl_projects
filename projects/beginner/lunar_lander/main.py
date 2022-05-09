from learning import LunarLanderAI


def main():
    agent = LunarLanderAI()
    # agent.render_random_games()
    agent.training_routine()


if __name__ == '__main__':
    main()
