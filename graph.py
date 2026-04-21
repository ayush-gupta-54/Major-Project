import torch

def build_graph(parsed):
    class Dummy:
        def __init__(self):
            self.x = torch.rand(3, 128)
    
    return Dummy(), {}