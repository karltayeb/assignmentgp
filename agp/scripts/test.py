from agp.assignmentgp.stochastic_gpassignment import GaussianSMAGP, SMAGP, GaussianSAGP

import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import gpflow.training.monitor as mon

import tensorflow as tf
import numpy as np

import pickle
import os

from scipy.stats import multivariate_normal
import pandas as pd
import seaborn as sns

import tensorflow_probability as tfp
import datetime

from gpflow.likelihoods import Likelihood
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter
from gpflow import transforms, settings

class NegativeBinomial(Likelihood):
    def __init__(self, dispersion=1.0, **kwargs):
        """
        :param scale float: scale parameter
        :param df float: degrees of freedom
        """
        super().__init__(**kwargs)
        self.dispersion = Parameter(
            dispersion, transform=transforms.positive, dtype=settings.float_type)

    @params_as_tensors
    def logp(self, F, Y):
        mu = tf.exp(F)
        p = F -  tf.log(self.dispersion + mu)
        p = tf.clip_by_value(p, -100, 0)
        q = tf.log(1 - tf.exp(p))

        logp = tf.math.lgamma(Y + self.dispersion) - tf.math.lgamma(self.dispersion) - tf.lgamma(Y + 1)
        logp += (q * self.dispersion) + (p * Y)
        return logp
    
    def get_variance(self, mean):
        return mean + (1 / self.dispersion) * mean**2

def build_monitor(model, path=''):
    session = model.enquire_session()
    global_step = mon.create_global_step(session)

    # build monitor
    print_task = mon.PrintTimingsTask().with_name('print')\
        .with_condition(mon.PeriodicIterationCondition(500))\
        .with_exit_condition(True)

    sleep_task = mon.SleepTask(0.01).with_name('sleep').with_name('sleep')

    saver_task = mon.CheckpointTask('./monitor-saves/' + path + model.name).with_name('saver')\
        .with_condition(mon.PeriodicIterationCondition(500))\
        .with_exit_condition(True)

    file_writer = mon.LogdirWriter('./model-tensorboard/' + path + model.name)

    model_tboard_task = mon.ModelToTensorBoardTask(
        file_writer, model, only_scalars=False, parameters=list(model.trainable_parameters)).with_name('model_tboard')\
        .with_condition(mon.PeriodicIterationCondition(50))\
        .with_exit_condition(True)

    lml_tboard_task = mon.LmlToTensorBoardTask(file_writer, model).with_name('lml_tboard')\
        .with_condition(mon.PeriodicIterationCondition(50))\
        .with_exit_condition(True)

    monitor_tasks = [print_task, model_tboard_task, saver_task, sleep_task]
    return monitor_tasks, session, global_step, file_writer


name = 'test_script'
path = 'test_path/'

# load data
file_path='../../data/ipsc_expression_data/raw_counts.txt'
data_square = pd.read_csv(file_path, delim_whitespace=True, index_col=0)

file_path = '../../data/ipsc_expression_data/gene_attributes.csv'
gene_attributes = pd.read_csv(file_path, index_col=0)

data_square = data_square.loc[gene_attributes.ensembl_gene_id]

data = pd.melt(data_square.reset_index(), id_vars=['Gene_id'])

split_labels = data.variable.apply(lambda x: str(x).split('_'))
labels_df = pd.DataFrame([x for x in split_labels.values], columns=['line', 'time'])

data = pd.concat([labels_df, data], axis=1)
data = data.drop('variable', 1)

Y = data.value.values[:, None].astype(np.float64)
X = data.time.values[:, None].astype(np.float64)

genes, W2_idx = np.unique(data.Gene_id, return_inverse=True)
cell_lines, W1_idx = np.unique(data.line, return_inverse=True)

libsizes = data.groupby(['line', 'time']).value.sum()
d = {x: i for i, x in enumerate(zip(libsizes.reset_index().line.values, libsizes.reset_index().time.values))}
idx = [d[x] for x in zip(data.line, data.time)]

library_size = data.groupby(['line', 'time']).value.sum().values[idx]
gene_lengths = gene_attributes['size'].values[W2_idx]
gc_content = gene_attributes.percentage_gene_gc_content.values[W2_idx] / 100


# make feature
Zlibsize = np.quantile(library_size, np.linspace(0.01, .99, 200))
Zgenesize = np.quantile(gene_lengths, np.linspace(0.01, 0.99, 200))
Z = np.log(np.vstack([Zlibsize, Zgenesize]).T)

X_aug = np.log(np.stack([library_size.flatten(), gene_lengths.flatten()]).T)

# build model
gpflow.reset_default_graph_and_session()
with gpflow.defer_build():
    kernel = gpflow.kernels.RBF(1, active_dims=[0]) + gpflow.kernels.RBF(1, active_dims=[1])
    feature = gpflow.features.InducingPoints(Z)
    
    model = gpflow.models.SVGP(
        X_aug, Y, kernel, NegativeBinomial(), 
        feat=feature, minibatch_size=500, name=name)
model.compile()


# restore/create monitor session
lr = 0.01
monitor_tasks, session, global_step, file_writer = build_monitor(model, path)

optimiser = gpflow.train.AdamOptimizer(lr)
if os.path.isdir('./monitor-saves/' + path + model.name):
    try:
        mon.restore_session(session, './monitor-saves/' + path + model.name)
    except:
        pass
else:
    os.makedirs('./monitor-saves/' + path + model.name)

model.anchor(session)

# optimize
with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
    optimiser.minimize(model, step_callback=monitor, maxiter=1000, global_step=global_step)
file_writer.close()