from learning import CartpoleAI


def main():
    agent = CartpoleAI()
    # env.render()
    agent.render_random_games(num_games=100)


if __name__ == '__main__':
    main()
