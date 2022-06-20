import math
import random
import numpy as np
import matplotlib.pyplot as plt
import operator
# initialize k, eta, w, r
# select row 
# search winning neuron 
# update neighbours weights, eta and r

class KohonenNeuron:
    def __init__(self,weights):
        self.weights = weights

    def __repr__(self):
        return f"Neuron - w: {self.weights}\n"
    
    def __str__(self):
        return f"Neuron - w: {self.weights}\n"


class Kohonen:
    def __init__(self, training_set, grid_dimension, radius, input_weights, learning_rate=1):
        self.training_set = training_set
        self.input_dimension = len(training_set[0])
        self.grid_dimension = grid_dimension
        self.learning_rate = learning_rate
        if radius[0]:
            self.radius = radius[1]
        else: 
            self.radius = grid_dimension
        self.neurons = [None] * grid_dimension
        #initialize weights 
        for i in range(self.grid_dimension):
            self.neurons[i] = [None] * grid_dimension
            for j in range(self.grid_dimension):
                if input_weights == False:
                    w = np.random.default_rng().uniform(0,1,self.input_dimension) #TODO cambiar a la otra opcion con un flag
                else: 
                    i_x = np.random.randint(0, len(self.training_set))               # get random input
                    input_ = self.training_set[i_x]
                    w = input_.copy()
                self.neurons[i][j] = KohonenNeuron(w)

    def train(self,epochs):
        tr_length = len(self.training_set)
        activations = np.zeros((self.grid_dimension, self.grid_dimension))

        for i in range(epochs):
            aux_training = self.training_set.copy()
            self.learning_rate = 1                                          # restart eta
            aux_radius = self.radius
            #print(f"---------- EPOCH {i} ----------")
            while len(aux_training) > 0: 
                i_x = np.random.randint(0, len(aux_training))               # get random input
                input_ = aux_training[i_x]
                aux_training = np.delete(aux_training, i_x, axis=0)

                
                (x,y,winner_neuron) = self.get_winner_neuron(input_)        # search winner neuron 
                activations[x][y] += 1
                self.update_weights(x,y,winner_neuron,input_,aux_radius)               # update neighbours and curr neuron weights 
                
                if(aux_radius > 1):
                    aux_radius -= 1
                self.learning_rate = 1/(tr_length - len(aux_training))
    
        #print("---------------- END ------------------")    
        umatrix = self.make_u_matrix()     

        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        aux = np.arange(self.grid_dimension)
        im1,cbar1 = self.heatmap(activations,ax1,"Entries amount","Entries per neuron",aux,aux,cmap="magma_r")
        print(f"Matrix of activations per neuron for all epochs:\n {activations}")
        
        im2,cbar2 = self.heatmap(umatrix,ax2,"Average euclidian distance","Average euclidian distance per neuron",aux,aux,cmap="binary")
        last_activations = np.zeros((self.grid_dimension, self.grid_dimension))
        
        to_return = []
        for entry in self.training_set:
            (i,j,winner) = self.get_winner_neuron(entry)
            to_return.append((i,j,winner))
            last_activations[i][j] += 1
        im3,cbar3 = self.heatmap(last_activations,ax3,"Entries amount","Final entries per neuron",aux,aux,cmap="magma_r")
        return to_return

    def heatmap(self,data,ax,cbar_label,title,row_label,col_label,cmap=""):
        im = ax.imshow(data,cmap=cmap)
        ax.set_title(title)
        cbar = ax.figure.colorbar(im,ax=ax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
        # We want to show all ticks...
        aux = np.arange(self.grid_dimension)
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries
        ax.set_xticklabels(row_label)
        ax.set_yticklabels(col_label)
        return im,cbar

    def update_weights(self,x,y,winner_neuron,input_,radius):
        # Iterar por todos las neuronas
            # Calcular su distancia al ganador 
            # Si dist < R: cambiar peso (es vecina)
            # Sino nada
        for i,row in enumerate(self.neurons):
            for j,neuron in enumerate(row): 
                if self.input_distance((i,j), (x,y)) <= radius: #x,y es la pos de la winner neuron -->  es vecina
                    neuron.weights += self.learning_rate * (input_ - neuron.weights)
                    # print(f"New wight for neuron({i},{j}): {neuron.weights}")

    def get_winner_neuron(self,input_):
        to_return = None # neuron position in output grid 
        min_dist = float('inf')
        for i,row in enumerate(self.neurons):
            for j,neuron in enumerate(row):
                
                dist = self.input_distance(input_, neuron.weights)
                # print(f"neuron({i},{j}): distance {dist}")
                if dist < min_dist: 
                    min_dist = dist
                    winner_neuron = self.neurons[i][j]
                    to_return = (i,j,winner_neuron)

        # print(f"Returning - x: {to_return[0]} - y: {to_return[1]}")
        return to_return

            
    def input_distance(self,p1,p2):
        dif_squares = [(v1 - v2)**2 for v1,v2 in zip(p1,p2)]
        return math.sqrt(sum(dif_squares))
    
    def make_u_matrix(self):
        u_matrix = np.zeros((self.grid_dimension, self.grid_dimension))
        for i in range(self.grid_dimension):
            for j in range(self.grid_dimension):
                # Check all 4 directions assuming radius = 1
                directions = list(filter(lambda direction: direction[0] >= 0 and direction[0] < self.grid_dimension and direction[1] >= 0 and direction[1] < self.grid_dimension,[(i-1,j), (i+1,j),(i,j-1), (i,j+1)]))

                for direction in directions:
                    u_matrix[i][j] += self.input_distance(self.neurons[i][j].weights,self.neurons[direction[0]][direction[1]].weights)
                u_matrix[i][j] /= len(directions)
        return u_matrix


def run_kohonen(training_set, grid_dimension, radius,learning_rate,epochs, use_input_as_weights,countries):
    kohonen = Kohonen(training_set, grid_dimension, radius,use_input_as_weights,learning_rate)
    last_activations = kohonen.train(epochs)
    country_activations = []
    for country,activations in zip(countries,last_activations):
        country_activations.append((country, *activations))
    country_activations = sorted(country_activations, key = operator.itemgetter(1, 2))

    for country_activation in country_activations:
        print(f"Country: {country_activation[0]} activated neuron({country_activation[1]},{country_activation[2]})")
    plt.show()