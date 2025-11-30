'''Improvement:
1. add the functional form of the bias model as String instance of the BiasModel object
2. add the name of the parameters. This way, the dictionary of parameters can be formed
 without the need to input the name of parameters.'''



import numpy as np


class BiasModel:
    def __init__(self, name, parameters, function, priors):
        self.name = name
        self.parameters = parameters
        self.function = function
        self.priors = priors
    


    @classmethod
    def linear_bias_model(cls):
        parameters = ["A", "b1"]
        function = cls.lin_bias
        name = "linear bias"
        priors = cls.linear_bias_priors
        return cls(parameters = parameters, function = function, name = name, priors=priors)


    @classmethod
    def quadratic_bias_model(cls):
        parameters = ["A", "b1", "b2"]
        function = cls.quad_bias
        name = "quadratic bias"
        priors = cls.quadratic_bias_priors
        return cls(parameters = parameters, function = function, name = name, priors = priors)
    
    @classmethod
    def power_law_bias_model(cls):
        parameters = ["alpha","beta"]
        function = cls.power_law_bias
        name = "power law bias"
        priors = cls.power_law_bias_priors
        return cls(parameters = parameters, function = function, name = name, priors = priors)
   




    @classmethod
    def sigmoid_bias_model(cls):
        parameters = ["a","b", "c", "d"]
        function = cls.sigmoid_bias
        name = "sigmoid bias"
        priors = cls.sigmoid_bias_prior
        return cls(parameters = parameters, function = function, name = name, priors = priors)
    

    @classmethod
    def exponential_bias_model(cls):
        parameters = ["a","b", "c", "d"]
        function = cls.exponential_bias
        name = "exponential bias"
        priors = cls.exponential_bias_prior
        return cls(parameters = parameters, function = function, name = name, priors = priors)

    @classmethod
    def splined_bias_model(cls, function):
        parameters = []
        name = "Spline Bias"
        priors = cls.splined_bias_prior
        return cls(parameters = parameters, function = function, name = name, priors = priors)

    @classmethod
    def double_linear_bias_model(cls):
        parameters = ["m1", "p1", "m2", "p2"]
        name = "double linear"
        priors = cls.double_linear_prior
        function = cls.double_linear
        return cls(parameters = parameters, function = function, name = name, priors = priors)

    


    @staticmethod
    def lin_bias(delta_m_2D, **kwargs): 
        '''delta_m_2D_flattened: a vector containing the projected matter density perturbation in 2D
        A = n_bar*N_k, where n is the average number density of halos in a voxel and N_k is the depth of a slab
        b1: linear bias parameter
        '''
        A =  kwargs["A"]
        b1 = kwargs["b1"]
        
        return A*(1 + b1*delta_m_2D)
    



    @staticmethod
    def quad_bias(delta_m_2D, **kwargs):
        '''delta_m_2D_flattened: a vector containing the projected matter density perturbation in 2D
        A = n_bar*N_k, where n is the average number density of halos in a voxel and N_k is the depth of a slab
        b1: linear bias parameter
        b2: quadratic bias parameter
        '''
        A = kwargs["A"]
        b1 = kwargs["b1"]
        b2 = kwargs["b2"]
        return A*(1 + b1*delta_m_2D + b2*delta_m_2D**2)
    

    @staticmethod
    def power_law_bias(delta_m_2D, **kwargs):
        alpha =  kwargs["alpha"]
        beta =  kwargs["beta"]
        return alpha*(1 + delta_m_2D)**beta
    


    @staticmethod
    def sigmoid_bias(delta_m_2D, **kwargs):
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        d = kwargs["d"]
        return a*BiasModel.sigmoid((delta_m_2D - c)/d) + b


    @staticmethod
    def exponential_bias(delta_m_2D, **kwargs):
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        d = kwargs["d"]
        # b = -a*(1 - np.exp((-1-c)/d))
        
        return a*BiasModel.exponential((delta_m_2D - c)/d) + b

    @staticmethod
    def double_linear(delta_m_2D, **kwargs):
        m1 = kwargs["m1"]
        p1 = kwargs["p1"]
        m2 = kwargs["m2"]
        p2 = kwargs["p2"]

        delta_m_t = (p2 - p1)/(m1 - m2)

        return np.heaviside(-(delta_m_2D - delta_m_t), 1)*(m1*delta_m_2D + p1) + \
                np.heaviside(delta_m_2D - delta_m_t, 1)*(m2*delta_m_2D + p2)




    @staticmethod
    def linear_bias_priors(**kwargs):
        A =  kwargs["A"]
        b1 = kwargs["b1"]

        if (A > 0) and (b1 >= 0 and b1<=5):
            return True
        else:
            return False



    @staticmethod
    def quadratic_bias_priors(**kwargs):
        A =  kwargs["A"]
        b1 = kwargs["b1"]
        b2 = kwargs["b2"]

        if (A > 0) and (b1 >= 0 and b1<=5) and (b2 > -1 and b2 < 2):
            return True
        else:
            return False
        


    

    @staticmethod
    def power_law_bias_priors(**kwargs):
        alpha =  kwargs["alpha"]
        beta = kwargs["beta"]

        if (alpha>0) or (beta>0):
            return True
        else:
            return False
    

    @staticmethod
    def sigmoid_bias_prior(**kwargs):
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        d = kwargs["d"]
        if a > 0 and c > 0 and d>0 and b < 0:
            return True
        else:
            return False 
        
    @staticmethod
    def exponential_bias_prior(**kwargs):
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        d = kwargs["d"]
        if a > 0 and b > 0 and d < 0:
            return True
        else:
            return False 

    @staticmethod
    def double_linear_prior(**kwargs):
        m1 = kwargs["m1"]
        p1 = kwargs["p1"]
        m2 = kwargs["m2"]
        p2 = kwargs["p2"]


        if (m1 > 0) and (m2 >0) and (p1 > 0):
            return True
        else:
            return False




    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    def exponential(x):
        return 1 - np.exp(x)
    


    def set_priors(self, priori_condition_funcion):
        self.priors = priori_condition_funcion
    
    
    def triple_linear_function(delta_m_2D, **kwargs):
        a1 = kwargs["a1"]
        b1 = kwargs["b1"]
        a2 = kwargs["a2"]
        b2 = kwargs["b2"]
        c  = kwargs["c"]
        delta_m_t1 = (c - b1)/a1
        delta_m_t2 = (b2 - b1)/(a1 - a2)
        
        H1 = np.heaviside(-(delta_m_2D - delta_m_t1), 1)
        H2 = np.heaviside(delta_m_2D - delta_m_t1, 1) 
        H3 = np.heaviside(-(delta_m_2D - delta_m_t2), 1)
        H4 = np.heaviside(delta_m_2D - delta_m_t2 ,1)
        return c*H1 + \
                (H2 + H3 - 1)*(a1*delta_m_2D + b1) + H4*(a2*delta_m_2D + b2)
    def triple_linear_prior(**kwargs):
        a1 = kwargs["a1"]
        b1 = kwargs["b1"]
        a2 = kwargs["a2"]
        b2 = kwargs["b2"]
        c  = kwargs["c"]

        return True