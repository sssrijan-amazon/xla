import torch_xla2
import torch

torch_xla2.env.enable_globally()
torch_xla2.env.treat_cuda_as_xla_on_tpu()

a = torch.randn((10, 10), device='cuda')

class M(torch.nn.Module):

    def forward(self, x):
        return x * 2 + torch.randn((10, ))

m = M()
print(m(a))

import torch_xla2.extra
m = torch_xla2.extra.jax_jit(M())
print(m(a))