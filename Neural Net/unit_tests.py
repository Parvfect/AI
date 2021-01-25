
from neural import *

def forward_test():
    t = Net(2, 2, 4, 1, [[0.3, 0.2], [0.1, 0.4]])
    t.forward_pass()



forward_test()