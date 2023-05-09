from argparse import Namespace

class ConfigGenerator:
    """
    
    Given a width and depth, generate all sane configurations for the MLP.

    Generation should be of form Namespace, with the following attributes:
    - hidden_dim
    - depth
    - lr
    - batch_size
    - dropout
    
    """
    def __init__(self):
        self.configs = []
    
    def generate(self, width, depth):
        """
        
        Maybe it's worth grid searching for now just to get an idea

        Heuristics:
        - width 32 depths 1 and 2 prefer 0.05 dropout
        
        """
        pass
