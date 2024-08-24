"""
    Treina o modelo DQN no ambiente CartPole-v1.

    Args:
        model_save_path (str): Caminho para salvar o modelo treinado.
        log_dir (str): Diretório para salvar os logs do TensorBoard.
        total_timesteps (int): Número total de passos de treinamento.

    Returns:
        model: O modelo DQN treinado.
    """
import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from cartpole.callbacks.tensorboard_callback import TensorboardCallback

def train(model_save_path, log_dir, total_timesteps):
    """
    Treina o modelo DQN no ambiente CartPole-v1.

    Args:
        model_save_path (str): Caminho para salvar o modelo treinado.
        log_dir (str): Diretório para salvar os logs do TensorBoard.
        total_timesteps (int): Número total de passos de treinamento.

    Returns:
        model: O modelo DQN treinado.
    """

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = DummyVecEnv([lambda: env])

    if os.path.exists(model_save_path + ".zip"):
        print("Carregando modelo existente...")
        model = DQN.load(model_save_path, env=env)
    else:
        print("Criando um novo modelo...")
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

    tensorboard_callback = TensorboardCallback()
    print("Iniciando treinamento...")
    model.learn(total_timesteps=total_timesteps, tb_log_name="DQN_CartPole", callback=tensorboard_callback, progress_bar=True)
    print("Treinamento concluído.")

    model.save(model_save_path)
    print(f"Modelo salvo em {model_save_path}")
    return model
