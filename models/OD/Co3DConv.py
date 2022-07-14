import torch
import continual as co

example = torch.randn((1, 1, 5, 3, 3))

conv = co.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3))

# Same exact computation as torch.nn.Conv3d 
output = conv(example)

# But can also perform online inference efficiently 
firsts = conv.forward_steps(example[:, :, :4])
last = conv.forward_step(example[:, :, 4])

assert torch.allclose(output[:, :, : conv.delay], firsts)
assert torch.allclose(output[:, :, conv.delay], last)

# Temporal properties
assert conv.receptive_field == 3
assert conv.delay == 2


class Co3DConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        

