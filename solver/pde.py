class PDE:
    def __init__(self, name: str, alpha: float):
        if name not in ['heat', 'heat2D']:
            raise ValueError(f"Unknown PDE name: {name}")
        else:
            if name == 'heat':
                self.pde = self.heat
                self.required_derivatives = ['dudt', 'd2udx2']
            elif name == 'heat2D':
                self.pde = self.heat2D
                self.required_derivatives = ['dudt', 'd2udx2', 'd2udy2']
        self.alpha = alpha

    def heat(self, dudt, d2udx2):
        return dudt - self.alpha**2 * d2udx2
    
    def heat2D(self, dudt, d2udx2, d2udy2):
        return dudt - self.alpha**2 * (d2udx2 + d2udy2)

    def substitute_into_equation(self, derivatives):
        """
        Substitute the derivatives into the PDE to calculate the residual.

        Parameters:
            derivatives (dict): a dictionary containing the derivatives

        Returns:
            torch.Tensor: the result of the PDE calculation
        """
        return self.pde(**{key: derivatives[key] for key in self.required_derivatives})
