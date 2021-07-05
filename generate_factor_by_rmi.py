import numpy as np
import pandas as pd
import scipy as sp
import os
import time
import datetime
from dateutil import rrule
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import rmi.estimation as inf
import rmi.pca
import rmi.neuralnets as nn

if __name__ == '__main__':

    load_dir = './input'
    save_dir = './output'

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_name', default='', help='The name of csv file to load as data.')
    parser.add_argument('--y_nn', type=int, default=2, help='The dimension of generated neural network features.')
    parser.add_argument('--y_pca', type=int, default=5, help='The dimension of generated PCA features.')
    parser.add_argument('--n_hidden_layer', type=int, default=30, help='The number of hidder layer neurons.')
    parser.add_argument('--batch_size', type=int, default=512, help='The batch size for training.')
    parser.add_argument('--step', type=int, default=5000, help='The number of steps for training.')
    args = parser.parse_args()

    assert args.y_nn in [1, 2], "ERROR: Now only Support y_nn=1 or y_nn=2."

    # only support csv input file
    if not args.csv_name.lower().endswith('.csv'):
        args.csv_name = args.csv_name + '.csv'
    
    assert os.path.exists(os.path.join(load_dir, args.csv_name)), "ERROR: Input file not found!"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # df_data = pd.read_csv(os.path.join(load_dir, args.csv_name))
    df_data = pd.read_csv(os.path.join(load_dir, args.csv_name), index_col=['Date'], parse_dates=['Date'])
    df_data.drop(columns=['AP', 'sn', 'ni', 'SF', 'SM', 'bu', 'sc', 'fu', 'eg', 'eb', 'pg', 'SA', 'PF', 'ss'], inplace=True)
    df_data.dropna(inplace=True)

    ndarray_data = np.array(df_data)
    shape_N, shape_D = df_data.shape

    rmi_optimizer = nn.RMIOptimizer(
        layers=[
            nn.K.layers.Dense(args.n_hidden_layer, activation="relu",input_shape=(shape_D, )),
            nn.K.layers.Dense(args.y_nn)
    ])
    rmi_optimizer.compile(optimizer=nn.tf.optimizers.Adam(1e-3))
    print("Note: (shape_N, shape_D) = ", df_data.shape)
    rmi_optimizer.summary()

    rmi_net = nn.Net(rmi_optimizer)

    batchsize = 128
    def get_batch():
        return ndarray_data
    rmi_net.fit_generator(get_batch, args.step)

    nn_feature, nn_gradient = rmi_net.get_feature_and_grad(ndarray_data)
    nn_RMI = inf.RenormalizedMutualInformation(nn_feature, nn_gradient)
    print("Renormalized Mutual Information (x,f(x)) where f is neural network: %.2f" % nn_RMI)

    if args.y_nn == 1:
        df_nn_feature = pd.DataFrame(nn_feature, index=df_data.index, columns=['featureNN_y1'])
    elif args.y_nn == 2:
        df_nn_feature = pd.DataFrame(nn_feature, index=df_data.index, columns=['featureNN_y1', 'featureNN_y2'])

    g_pca = rmi.pca.pca(ndarray_data, args.y_pca)
    pca_feature = g_pca.transform(ndarray_data)
    pca_gradient = np.repeat(np.expand_dims(g_pca.w, axis=0), shape_N, axis=0)
    if args.y_pca <= 2:
        pca_RMI = inf.RenormalizedMutualInformation(pca_feature, pca_gradient)
        print("Renormalized Mutual Information (x,f(x)) where f is PCA: %.2f" % pca_RMI)
    else:
        print("y_pca is too big to calculate RMI.")

    pca_columns = ['featurePCA_y%s' % (i + 1, ) for i in range(args.y_pca)]
    df_pca_feature = pd.DataFrame(pca_feature, index=df_data.index, columns=pca_columns)

    df_save = pd.concat([df_data, df_nn_feature, df_pca_feature], axis=1)
    df_save.to_csv(os.path.join(save_dir, 'features_' + args.csv_name))
    print('Successfully saving features to %s.' % (os.path.join(save_dir, 'features_' + args.csv_name), ))
