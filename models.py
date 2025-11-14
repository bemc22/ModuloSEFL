import deepinv as dinv
import torch
import torch.nn.functional as F


def grad_y(y):
    Mdx_y = F.pad(torch.diff(y, 1, dim=-1), (1, 0))
    Mdy_y = F.pad(torch.diff(y, 1, dim=-2), (0, 0, 1, 0))
    return Mdx_y, Mdy_y

class ModuloSEFLnet(torch.nn.Module):
    def __init__(self, 
                 mx=1.0,
                 in_channels=3, 
                 out_channels=3, 
                 features=8,):
        super(ModuloSEFLnet, self).__init__()

        self.mx = mx

        in_channels = in_channels * 3
        nc = (features * (2 ** i) for i in range(4))

        self.model = dinv.models.DRUNet(
            in_channels=in_channels-1,
            out_channels=out_channels,
            nc=list(nc),
            nb=4,
            pretrained=None)

        self.act = torch.nn.ReLU()

        self.modulo_round = dinv.physics.spatial_unwrapping.modulo_round

    def forward(self, x):
        
        yinput = self.feature_lifting(x, self.mx)
        out = self.model.forward_unet(yinput)
        out = self.act(out)
        return out


    def feature_lifting(self, y, mx):

        yinput = [y]

        Dhy, Dvy = grad_y(y)
        Dy = torch.cat([Dhy, Dvy], dim=1)
        Dy = self.modulo_round(Dy, mx)
        yinput.append(Dy)

        yinput = torch.cat(yinput, dim=1)

        return yinput
