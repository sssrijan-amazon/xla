import jax
from torch.overrides import TorchFunctionMode
import torch
import torch_xla2.tensor


class FunctionMode(TorchFunctionMode):
    def __init__(self, env):
        self.env = env

    def __torch_function__(self, func, types, args, kwargs=None):
        print('captured:', func)
        if func == torch.randn:
            arr = jax.random.normal(self.env._prng_key, *args)
            return torch_xla2.tensor.wrap(arr)
        kwargs = kwargs or {}
        return func(*args, **kwargs)


class Environment:
    """This class holds a set of configurations and "globals" needed

    for executing torch program using jax.
    Things included so far:

    op registry
    PRNGKey
    Configs

    Also helper functions to manipulate those.
    """

    _prng_key: jax.random.PRNGKey



    def __init__(self, random_seed):
        self._prng_key = jax.random.PRNGKey(random_seed)
        self._torch_function_overrides = FunctionMode(self)

    def get_and_rotate_prng_key(self):
        self._prng_key, key = jax.random.split(self._prng_key)

    def enable_globally(self):
        self._torch_function_overrides.__enter__()

    def treat_cuda_as_xla_on_tpu(self):
        pass

