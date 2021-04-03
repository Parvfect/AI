from tensorflow.keras.optimizers import Adam
from tenorflow.keras.layers import Input, Embedding, Dense, LTSM, Bidirectional
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as K
from tensorflow import config as config
#from .AttentionWeightedAverage import AttentionWeightedAverage

def textgenrnn_model(num_classes, cfg, context_size= None, 
                    weights_path = None, 
                    dropout = 0.0,
                    optimizer = Adam(lr = 4e-3)):
    '''
    Builds the model architecture for textgenrnn and
    loads the specified weights for the model.
    '''

    
