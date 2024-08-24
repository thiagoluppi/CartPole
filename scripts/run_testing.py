import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cartpole.test import test

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, '../data/models/dqn_cartpole')

    test(model_save_path)
