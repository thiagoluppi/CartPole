import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cartpole.train import train

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, '../data/models/dqn_cartpole')
    log_dir = os.path.join(current_dir, '../data/logs/')

    train(model_save_path,
          log_dir,
          total_timesteps=100000)  # Você pode ajustar o número de timesteps aqui
