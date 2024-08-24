import os
import gymnasium as gym
from stable_baselines3 import DQN #DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def test(model_save_path):
    """
    Testa o modelo DQN no ambiente CartPole-v1.

    Args:
        model_save_path (str): Caminho do modelo treinado para carregar e testar.
    """

    env = gym.make('CartPole-v1', render_mode="human")
    env = DummyVecEnv([lambda: env])

    if not os.path.exists(model_save_path + ".zip"):
        raise FileNotFoundError(f"Modelo não encontrado em {model_save_path}. Treine o modelo antes de testar.")

    model = DQN.load(model_save_path)

    obs = env.reset()
    total_reward = 0

    print("Iniciando teste do modelo...")
    for _ in range(2000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()  # Renderiza a simulação
        total_reward += reward  # Acumula a recompensa

        # Verifica se a chave "is_success" está presente e imprime seu valor
        if "is_success" in info:
            print("Sucesso:", info["is_success"])

        if done:
            print(f"Episode finished with total reward: {total_reward}")
            obs = env.reset()
            total_reward = 0  # Reseta a recompensa acumulada para o próximo episódio

    # print("Iniciando teste do modelo...")
    # for _ in range(200):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)

    #     env.render()  # Renderiza a simulação

    #     if done:
    #         obs = env.reset()

    env.close()
    print("Teste concluído.")
