from copy import deepcopy


# https://github.com/Thunderbeee/ZSCL/blob/63a2b97a626821b155153ec76765c046e14970a2/mtil/src/models/helpers.py#L31
def merge_we(current_model, we_model, sma_count):
    for param_q, param_k in zip(current_model.parameters(), we_model.parameters()):
        param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)


def get_weight_ensemble_trainer_class(meta_trainer_class):
    class WeightEnsembleTrainer(meta_trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self._we_model = deepcopy(self._model).cuda()
            self._we_model.eval()

            self._weight_update_counter = 0

        @property
        def weight_update_counter(self):
            return self._weight_update_counter

        @weight_update_counter.setter
        def weight_update_counter(self, value):
            self._weight_update_counter = value

        @property
        def weight_space_config(self):
            return self.config.method.get("weight_space_config", None)

        @property
        def eval_model(self):
            return self._we_model

        def train_step(self, images, labels):
            loss_dict = super().train_step(images, labels)

            if self.current_num_iterations % self.weight_space_config.interval == 0:
                self.weight_update_counter += 1
                merge_we(
                    self._model, self._we_model, sma_count=self.weight_update_counter
                )

            return loss_dict

    return WeightEnsembleTrainer
