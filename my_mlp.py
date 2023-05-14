import torch


class MyMLP(torch.nn.Module):
    '''My MLP module, which has 5 linear layers, 4 ReLU layers and 4 LayerNorm layers. '''
    def __init__(self, input_dim: int, output_dim: int):
        super(MyMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 8192), 
            torch.nn.ReLU(), 
            torch.nn.LayerNorm(), 
            torch.nn.Linear(8192, 2048), 
            torch.nn.ReLU(), 
            torch.nn.LayerNorm(), 
            torch.nn.Linear(2048, 512), 
            torch.nn.ReLU(), 
            torch.nn.LayerNorm(), 
            torch.nn.Linear(512, 128), 
            torch.nn.ReLU(), 
            torch.nn.LayerNorm(), 
            torch.nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)