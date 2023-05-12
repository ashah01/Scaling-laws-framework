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
    def __init__(self, bsz_ss, lr_ss, drop_ss):
        self.heuristics = []
        self.bsz_ss = bsz_ss
        self.lr_ss = lr_ss
        self.drop_ss = drop_ss
    
    def generate(self, width, depth):
        """
        
        Maybe it's worth grid searching for now just to get a preliminary search space

        bsz: [32, 64, 100]
        lr: [1e-4, 3e-4, 5e-4]
        dropout: [0.01, 0.02, 0.05]



        Heuristics:
        - all widths prefer decreased dropout after depth 2
        
        """
        import IPython; IPython.embed()
        for h in self.heuristics:
            h(width, depth)
        for bsz in self.bsz_ss:
            for lr in self.lr_ss:
                for drop in self.drop_ss:
                    yield Namespace(hidden_dim=width, depth=depth, lr=lr, batch_size=bsz, dropout=drop)


    def heuristic(self, h):
        def decorator(width, height):
            self.heuristics.append(h())
        
        return decorator

    @heuristic
    def drop_heuristic(self, width, height):
        if height > 2 and width <= 128:
            self.drop_ss = [0.01, 0.02]

test_config = ConfigGenerator([32, 64, 100], [1e-4, 3e-4, 5e-4], [0.01, 0.02, 0.05])
test_config.generate(32, 3)