from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    # def __init__(self, verbose=0):
    #     super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Esta função é chamada a cada passo
        return True

    def _on_training_end(self) -> None:
        pass

    def _on_training_start(self) -> None:
        pass
