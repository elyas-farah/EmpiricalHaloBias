import numpy as np

class ThetaModel:
    def __init__(self, parameters, function, name, priors):
        self.parameters = parameters
        self.function = function
        self.name = name
        self.priors = priors

    
    @classmethod
    def poissonian_model(cls):
        parameters = None
        function = cls.poissonian
        name = "Poissonian"
        priors = None
        return cls(parameters, function, name, priors)

    @classmethod
    def linear_theta_model(cls):
        parameters = ["theta1", "theta2"]
        function = cls.linear_theta
        name = "linear theta"
        priors = cls.linear_theta_priors
        return cls(parameters, function, name, priors)
    

    @classmethod
    def exponential_theta_model(cls):
        parameters = ["theta1", "theta2", "theta3"]
        function = cls.exponential_theta
        name = "exponential theta"
        priors = cls.exponential_theta_priors
        return cls(parameters, function, name, priors)

    @classmethod
    def exact_sigmoid_theta_model(cls):
        parameters = ["alpha1", "alpha2"]
        function = cls.exact_sigmoid_theta
        name = "sigmoid stochasticity"
        priors = cls.exact_sigmoid_theta_priors
        return cls(parameters, function, name, priors)
    

    @classmethod
    def exact_linear_theta_model(cls):
        parameters = ["alpha1", "alpha2"]
        function = cls.exact_linear_theta
        name = "linear stochasticity"
        priors = cls.exact_linear_theta_priors
        return cls(parameters, function, name, priors)
    


    @classmethod
    def exact_exponential_theta_model(cls):
        parameters = ["alpha1", "alpha2"]
        function = cls.exact_exponential_theta
        name = "exponential stochasticity"
        priors = cls.exact_exponential_theta_priors
        return cls(parameters, function, name, priors)



    @staticmethod
    def poissonian(delta_m_2D, **kwargs):
        
        return 0    

    @staticmethod
    def linear_theta(delta_m_2D, **kwargs):
        theta1 = kwargs["theta1"]
        theta2 = kwargs["theta2"]
        return theta1 + theta2*delta_m_2D    
    

    @staticmethod
    def exponential_theta(delta_m_2D, **kwargs):
        theta1 = kwargs["theta1"]
        theta2 = kwargs["theta2"]
        

        return theta1*(1 - np.exp(theta2*delta_m_2D))
    

    @staticmethod
    def exact_sigmoid_theta(delta_m_2D, **kwargs):
        alpha1 = kwargs["alpha1"]
        alpha2 = kwargs["alpha2"]
        return 1 - 1/np.sqrt(alpha1*ThetaModel.sigmoid(delta_m_2D) + alpha2)
    
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def exact_linear_theta(delta_m_2D, **kwargs):
        alpha1 = kwargs["alpha1"]
        alpha2 = kwargs["alpha2"]
        return 1 - 1/np.sqrt(alpha1 + alpha2*delta_m_2D)
    

    @staticmethod
    def exact_exponential_theta(delta_m_2D, **kwargs):
        alpha1 = kwargs["alpha1"]
        alpha2 = kwargs["alpha2"]
        return 1 - 1/np.sqrt(alpha1*(1 - np.exp(alpha2*delta_m_2D)))
    







    @staticmethod
    def linear_theta_priors(**kwargs):
        theta1 = kwargs["theta1"]
        theta2 = kwargs["theta2"]


        if (theta1 > 0):
            return True
        else:
            return False
        

    
    @staticmethod
    def exponential_theta_priors(**kwargs):
        theta1 = kwargs["theta1"]
        theta2 = kwargs["theta2"]


        if (theta1 > 0) and (theta2<0):
            return True
        else:
            return False
        

    @staticmethod
    def exact_sigmoid_theta_priors(**kwargs):
        alpha1 = kwargs["alpha1"]
        alpha2 = kwargs["alpha2"]

        if (alpha1 > 0) and (alpha2 < 2):
            return True
        else:
            return False
        


    @staticmethod
    def exact_linear_theta_priors(**kwargs):
        alpha1 = kwargs["alpha1"]
        alpha2 = kwargs["alpha2"]

        if (alpha1 > 0.2) and (alpha2 > 0):
            return True
        else:
            return False
        
    

    @staticmethod
    def exact_exponential_theta_priors(**kwargs):
        alpha1 = kwargs["alpha1"]
        alpha2 = kwargs["alpha2"]

        if (alpha1 > 0.5) and (alpha2 < 0):
            return True
        else:
            return False
    


    @staticmethod
    def linear_theta_stoch_relative_error(delta_m_2D, **kwargs):
        theta1 = kwargs["theta1"]
        theta2 = kwargs["theta2"]
        theta1_error = kwargs["theta1 error"]
        theta2_error = kwargs["theta2 error"]

        first_term = 2*theta1_error/ThetaModel.linear_theta(delta_m_2D, theta1 = theta1, theta2=theta2) 
        second_term = (2*delta_m_2D*theta2_error)/ThetaModel.linear_theta(delta_m_2D, theta1 = theta1, theta2=theta2) 
        return np.abs(first_term) + np.abs(second_term)
    

    def exact_linear_theta_error(delta_m_2D, **kwargs):
        alpha1_error = kwargs["alpha1 error"]
        alpha2_error = kwargs["alpha2 error"]
        return alpha1_error + delta_m_2D*alpha2_error
    
    def exact_exponential_theta_error(delta_m_2D, **kwargs):
        alpha1 = kwargs["alpha1"]
        alpha2 = kwargs["alpha2"]
        alpha1_error = kwargs["alpha1 error"]
        alpha2_error = kwargs["alpha2 error"]
        return (1 - np.exp(delta_m_2D*alpha2))*alpha1_error + alpha1*delta_m_2D*np.exp(alpha2*delta_m_2D)*alpha2_error
    


    def exact_sigmoid_theta_error(delta_m_2D, **kwargs):
        alpha1_error = kwargs["alpha1 error"]
        alpha2_error = kwargs["alpha2 error"]
        return alpha1_error/ThetaModel.sigmoid(delta_m_2D) + alpha2_error