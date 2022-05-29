# p is the size of the training set 
# w is an array of synaptic weights
# N is the dimension of the Input --> as we are working with points N=2
# x is an array with the training_set (Input) It already has an E0=1 so we do not add w0
# y is an array of expected outputs

# Desconozco los pesos sinápticos que tiene que tener la neurona. Dsp de varias iteraciones voy a obtener el w que necesito. 
# La salida de la neurona es el estado de activación. Pero yo quiero que la salida sea Z (expected_output)
# Si la salidad de la neurona (O, activation_state) es distinta a lo que yo queria (Z) --> le aplico la correcion delta_w, sino la dejo como estaba, esa corrección depende de mi entrada (E)
# El objetivo es que el perceptron converja a la solución.

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize_scalar
from utils import LoadingBar

class Perceptron(ABC):

    def __init__(self, training_set, expected_output,learning_rate):
        # self.training_set = np.array(list(map(lambda t: [1]+t, training_set)))
        self.training_set = np.append(np.ones((training_set.shape[0],1)), training_set, axis=1)
        print(self.training_set)
        self.expected_output = np.array(expected_output)
        self.learning_rate = learning_rate
        self.w_min = None
        self.error_min = None
    
    def train(self, limit, show_loading_bar: bool=True):
        i = 0
        n = 0
        p = len(self.training_set)
        dimension = len(self.training_set[0])   
        w = np.random.uniform(-1, 1, dimension) # array de longitud p+1 con valores random entre -1 y 1  

        if show_loading_bar:
            loading_bar = LoadingBar()
            loading_bar.init()
        error = 1
        # self.error_min = p*2
        self.error_min = float('inf')
        while error > 0 and i < limit:
            if show_loading_bar:
                loading_bar.update(1.0*i / limit)
            if n >= 100 * p: # initialize weights again
                w = np.random.uniform(-1, 1,dimension)  
                n = 0
                
            i_x = np.random.randint(0, p) # get a random point index from the training set  
         
            excited_state = np.inner(self.training_set[i_x], w) # internal product: sum (e[i_x]*w_i) --> hiperplano
 
            activation_state = self.activation(excited_state) 
 
            delta_w = (self.learning_rate * (self.expected_output[i_x] - activation_state)) * self.training_set[i_x] * self.delta_w_correction(excited_state)

            w += delta_w 
            
            error = self.error(w)
            
            #self.optimize_learning_rate(w, activation_state, excited_state, i_x)

            if error < self.error_min:
                self.error_min = error
                self.w_min = w

            i += 1
            n += 1
        if show_loading_bar:
            loading_bar.end()
        
    # Funcion que recibe array de arrays y con el perceptron entrenado, 
    # devuelve el valor de activation_state sea el esperado
    def get_output(self, input):
        print("MIN ERROR:", self.error_min)
        outputs = []
        aux_input = np.array(list(map(lambda t: [1]+t, input)))
        for i in range(len(aux_input)):
            excited_state = np.inner(aux_input[i], self.w_min)
            outputs.append(self.activation(excited_state))
        return outputs

        
    def optimize_learning_rate(self, w, activation_state, excited_state, i_x): 
        ans = minimize_scalar(
            self.calculate_error, 
            bounds=(0,0.1), # search a local min between 0 and 1
            args=( w, activation_state, excited_state, i_x), 
            method = 'bounded', 
            options= { 'maxiter': 100} 
        )
        
        if ans.success: 
            print("Optimized eta")
            print(ans.x)
    
    
    def calculate_error(self,alpha,w, activation_state, excited_state, i_x): 
        delta_w = (alpha * (self.expected_output[i_x] - activation_state)) * self.training_set[i_x] * self.delta_w_correction(excited_state)
        w += delta_w 
        return self.error(w)
    
    @abstractmethod
    def activation(self, excited_state):
        pass

    # funcion que calcula el error en cada iteracion utilizando el conjunto de entrenamiento,
    # la salida esperada, el vector de pesos y la longitud del conjunto de entranamiento
    @abstractmethod
    def error(self,w):
        pass
    
    # en el perceptron no lineal hay que multiplicar delta_w * g'(h)
    def delta_w_correction(self,h):
        return 1

#      w0 e1 e2     
#     [ 1 -1,1    1 1,-1    1 -1,-1    1 1,1  ]  training set (E)
# and   -1         -1          -1         1      expected_outputs (Z) --> es lo que quiero aprender
# xor    1          1          -1        -1

class SimplePerceptron(Perceptron):

    def __init__(self, training_set, expected_output,learning_rate):
        super().__init__(training_set,expected_output,learning_rate)
    
    def activation(self, excited_state):
        return 1.0 if excited_state >= 0.0 else -1.0   

    def error(self,w):
        # Para cada elemento del conjunto de entrada aplicandole su peso correspondiente, 
        # tengo que ver si da la salida esperada, en caso de que no de voy acumulando dicho error
        training_size = len(self.training_set)
        error = 0
        for i in range(training_size):
            excited_state = np.inner(self.training_set[i], w)
            #if abs(self.activation(excited_state) - self.expected_output[i]) != 0:
            #   print("Error for line %d" % i)
            error += abs(self.activation(excited_state) - self.expected_output[i])
        return error
    