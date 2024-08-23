import sys
import os
import gym

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from callbacks.tensorboard_callback import TensorboardCallback

# Define o caminho absoluto para salvar o modelo e os logs
model_save_path = os.path.join(os.path.dirname(__file__), '../data/models/dqn_cartpole')
log_dir = os.path.join(os.path.dirname(__file__), '../data/logs/')

# Cria o ambiente CartPole-v1 sem renderização para o treinamento
env = gym.make('CartPole-v1')  # Sem render_mode, sem renderização
env = DummyVecEnv([lambda: env])  # Necessário para compatibilidade com Stable-Baselines3

# Cria o modelo DQN com uma política de MLP (Multi-Layer Perceptron)
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# Cria uma instância do callback do TensorBoard
tensorboard_callback = TensorboardCallback()

# Treina o modelo por 10.000 steps com o callback do TensorBoard
print("Iniciando treinamento...")
model.learn(total_timesteps=10000, tb_log_name="DQN_CartPole", callback=tensorboard_callback)
print("Treinamento concluído.")

# Salva o modelo treinado usando o caminho absoluto
model.save(model_save_path)
print(f"Modelo salvo em {model_save_path}")

# --- Fim do Treinamento ---

# Carrega o modelo salvo
model = DQN.load(model_save_path)

# Testa o modelo treinado com renderização
env = gym.make('CartPole-v1', render_mode="human")
env = DummyVecEnv([lambda: env])  # Necessário para compatibilidade com Stable-Baselines3

obs = env.reset()
print("Iniciando teste do modelo...")
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    env.render()  # Renderiza a simulação

    if done:
        obs = env.reset()

env.close()
print("Teste concluído.")
