import numpy as np
# import matplotlib.pyplot as plt


# task 1, 2
# NOTE: here we work with scalars

# initialize some data
q_sigmoid = lambda x: 0  if x < 0 else (1 if x > 0 else 0.5)
q_tanh =    lambda x: -1 if x < 0 else (1 if x > 0 else 0)

sigmoid = lambda x: 1/(1+np.exp(-1*x))
tanh = np.tanh

W_f_h = W_i_h = W_o_h = 0
W_f_x = 0
W_i_x = W_o_x = 100
W_c_h = -100
W_c_x = 50

b_f = -100
b_i = 100
b_o = 0
b_c = 0

# input_sequence = [0, 0, 1, 1, 1, 0]  # task 1
input_sequence =  [1, 1, 0, 1, 1]  # task 2

# make calculations
h_prev = c_prev = 0
for i, x in enumerate(input_sequence):
    # update indices
    f_curr = sigmoid(
        W_f_h*h_prev + W_f_x*x + b_f
    )
    i_curr = sigmoid(
        W_i_h * h_prev + W_i_x * x + b_i
    )
    o_curr = sigmoid(
        W_o_h * h_prev + W_o_x * x + b_o
    )
    c_curr = f_curr*c_prev + i_curr*tanh(
        W_c_h * h_prev + W_c_x * x + b_c
    )
    h_curr = o_curr*tanh(c_curr)

    # output data
    print("h[{idx}]={val}".format(
        idx=i, val=h_curr
    ))

    # remember indices
    c_prev = c_curr
    h_prev = h_curr
