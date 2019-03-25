import numpy as np
import tensorflow as tf

from gpflow.models import GPModel
from gpflow import features
from gpflow import likelihoods
from gpflow import settings
from gpflow.decors import autoflow
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.params import DataHolder, Parameter


class SGPRUpperMixin(object):
    """
    Upper bound for the GP regression marginal likelihood.
    It is implemented here as a Mixin class which works with SGPR and GPRFITC.
    Note that the same inducing points are used for calculating the upper bound,
    as are used for computing the likelihood approximation. This may not lead to
    the best upper bound. The upper bound can be tightened by optimising Z, just
    as just like the lower bound. This is especially important in FITC, as FITC
    is known to produce poor inducing point locations. An optimisable upper bound
    can be found in https://github.com/markvdw/gp_upper.
    The key reference isg
    ::
      @misc{titsias_2014,
        title={Variational Inference for Gaussian and Determinantal Point Processes},
        url={http://www2.aueb.gr/users/mtitsias/papers/titsiasNipsVar14.pdf},
        publisher={Workshop on Advances in Variational Inference (NIPS 2014)},
        author={Titsias, Michalis K.},
        year={2014},
        month={Dec}
      }
    """

    @autoflow()
    @params_as_tensors
    def compute_upper_bound(self):
        num_data = tf.cast(tf.shape(self.Y)[0], settings.float_type)

        Kdiag = self.kern.Kdiag(self.X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        Kuf = features.Kuf(self.feature, self.kern, self.X)

        L = tf.cholesky(Kuu)
        LB = tf.cholesky(Kuu + self.likelihood.variance ** -1.0 * tf.matmul(Kuf, Kuf, transpose_b=True))

        LinvKuf = tf.matrix_triangular_solve(L, Kuf, lower=True)
        # Using the Trace bound, from Titsias' presentation
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(LinvKuf ** 2.0)
        # Kff = self.kern.K(self.X)
        # Qff = tf.matmul(Kuf, LinvKuf, transpose_a=True)

        # Alternative bound on max eigenval:
        # c = tf.reduce_max(tf.reduce_sum(tf.abs(Kff - Qff), 0))
        corrected_noise = self.likelihood.variance + c

        const = -0.5 * num_data * tf.log(2 * np.pi * self.likelihood.variance)
        logdet = tf.reduce_sum(tf.log(tf.diag_part(L))) - tf.reduce_sum(tf.log(tf.diag_part(LB)))

        LC = tf.cholesky(Kuu + corrected_noise ** -1.0 * tf.matmul(Kuf, Kuf, transpose_b=True))
        v = tf.matrix_triangular_solve(LC, corrected_noise ** -1.0 * tf.matmul(Kuf, self.Y), lower=True)
        quad = -0.5 * corrected_noise ** -1.0 * tf.reduce_sum(self.Y ** 2.0) + 0.5 * tf.reduce_sum(v ** 2.0)

        return const + logdet + quad


class SGPR(GPModel, SGPRUpperMixin):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W, kern, feat=None, mean_function=None, Z=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects
        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X)
        Y = DataHolder(Y)
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_data = X.shape[0]

        self.W_prior = tf.ones(W.shape, dtype=settings.float_type) / W.shape[1]
        self.W = Parameter(W)
        self.num_inducing = Z.shape[0] * W.shape[1]


    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        ND = tf.cast(tf.size(self.Y), settings.float_type)
        D = tf.cast(tf.shape(self.Y)[1], settings.float_type)

        Kxu = self.kern.K(self.X, self.feature.Z)

        psi0 = self._psi0()
        psi1 = self._psi1(Kxu)
        psi2 = self._psi2(Kxu)

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = block_diagonal([L for _ in range(self.W.shape[1])])
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(self.num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        c = tf.matrix_triangular_solve(
            LB, tf.matmul(A, self.Y), lower=True) / sigma

        # KL[q(W) || p(W)]
        KL = tf.reduce_sum(self.Wnorm() * (tf.log(self.Wnorm()) - tf.log(self.W_prior)))

        # compute log marginal bound
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        bound -= KL
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        num_inducing = len(self.feature)
        err = self.Y - self.mean_function(self.X)
        Kuf = features.Kuf(self.feature, self.kern, self.X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)
        Kus = features.Kuf(self.feature, self.kern, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        B = tf.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            var = tf.tile(var[None, ...], [self.num_latent, 1, 1])  # P x N x N
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            var = tf.tile(var[:, None], [1, self.num_latent])
        return mean + self.mean_function(Xnew), var

    @autoflow()
    @params_as_tensors
    def compute_qu(self):
        """
        Computes the mean and variance of q(u), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, A
        """
        Kuf = features.Kuf(self.feature, self.kern, self.X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        Sig = Kuu + (self.likelihood.variance ** -1) * tf.matmul(Kuf, Kuf, transpose_b=True)
        Sig_sqrt = tf.cholesky(Sig)

        Sig_sqrt_Kuu = tf.matrix_triangular_solve(Sig_sqrt, Kuu)

        A = tf.matmul(Sig_sqrt_Kuu, Sig_sqrt_Kuu, transpose_a=True)
        mu = tf.matmul(Sig_sqrt_Kuu,
                       tf.matrix_triangular_solve(Sig_sqrt, tf.matmul(Kuf, self.Y - self.mean_function(self.X))),
                       transpose_a=True) * self.likelihood.variance ** -1.0

        return mu, A

    @params_as_tensors
    def _psi0(self):
        return self.num_data * self.kern.variance

    @params_as_tensors
    def _psi1(self, Kxu):
        psi1 = tf.reshape(
            Kxu[:, None, :] * self.Wnorm()[:, :, None], [self.num_data, -1])
        return psi1

    @params_as_tensors
    def _psi2(self, Kxu):
        psi2 = block_diagonal(tf.unstack(
            tf.einsum('nm,npk->kmp',Kxu, Kxu[:, :, None] * self.Wnorm()[:, None])))
        return psi2

    @params_as_tensors
    def Wnorm(self):
        return normalize(self.W)


class AssignmentGP(GPModel, SGPRUpperMixin):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W, kern, feat=None, mean_function=None, Z=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects
        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X)
        Y = DataHolder(Y, fix_shape=True)
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_data = X.shape[0]

        self.W_prior = tf.ones(W.shape, dtype=settings.float_type) / W.shape[1]
        self.W = Parameter(W)
        self.num_inducing = Z.shape[0]

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        ND = tf.cast(tf.size(self.Y), settings.float_type)
        D = tf.cast(tf.shape(self.Y)[1], settings.float_type)

        Kxu = self.kern.K(self.X, self.feature.Z)

        psi0 = self._psi0()
        psi1 = self._psi1(Kxu)  # K x N x M
        psi2 = self._psi2(Kxu)  # K x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = tf.tile(L[None], [self.W.shape[1], 1, 1])

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 2, 1]), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)

        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 2, 1]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None],
            [self.W.shape[1], 1, 1])
        LB = tf.cholesky(B)

        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))

        c = tf.matrix_triangular_solve(
            LB, tf.einsum('kmn,nd->kmd', A, self.Y), lower=True) / sigma

        # KL[q(W) || p(W)]
        KL = tf.reduce_sum(self.Wnorm() * (
            tf.log(self.Wnorm()) - tf.log(self.W_prior)))

        # compute log marginal bound
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        bound -= KL
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        num_inducing = len(self.feature)
        err = self.Y - self.mean_function(self.X)
        Kuf = features.Kuf(self.feature, self.kern, self.X)
        Kuu = features.Kuu(self.feature, self.kern,
                           jitter=settings.numerics.jitter_level)
        Kus = features.Kuf(self.feature, self.kern, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        B = tf.matmul(A, A, transpose_b=True) + \
            tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            var = tf.tile(var[None, ...], [self.num_latent, 1, 1])  # P x N x N
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            var = tf.tile(var[:, None], [1, self.num_latent])
        return mean + self.mean_function(Xnew), var

    @autoflow()
    @params_as_tensors
    def compute_qu(self):
        """
        Computes the mean and variance of q(u), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, A
        """
        Kux = features.Kuf(self.feature, self.kern, self.X)
        psi1 = self._psi1(tf.transpose(Kux))  # K x N x M
        psi2 = self._psi2(tf.transpose(Kux))  # K x M x M

        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        Kuu = tf.tile(Kuu[None], [self.W.shape[1], 1, 1])

        Sig = Kuu + (self.likelihood.variance ** -1) * psi2
        Sig_sqrt = tf.cholesky(Sig)

        Sig_sqrt_Kuu = tf.matrix_triangular_solve(Sig_sqrt, Kuu)

        A = tf.matmul(Sig_sqrt_Kuu, Sig_sqrt_Kuu, transpose_a=True)

        tmp = tf.matrix_triangular_solve(Sig_sqrt, tf.transpose(psi1, perm=[0, 2, 1]))
        P = tf.einsum('kmn,nd->kmd', tmp, self.Y)
        mu = tf.matmul(Sig_sqrt_Kuu, P, transpose_a=True) * self.likelihood.variance ** -1.0

        return mu, A

    @autoflow()
    @params_as_tensors
    def update_W(self):
        Kxu = self.kern.K(self.X, self.feature.Z)
        psi1 = self._psi1(Kxu)  # K x N x M
        psi2 = self._psi2(Kxu)  # K x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = tf.tile(L[None], [self.W.shape[1], 1, 1])

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.tile(tf.transpose(Kxu)[None],
                       [self.W.shape[1], 1, 1]), lower=True) / sigma

        Apsi = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 2, 1]), lower=True) / sigma

        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 2, 1]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None],
            [self.W.shape[1], 1, 1])
        LB = tf.cholesky(B)

        LBinvA = tf.matrix_triangular_solve(LB, A)  # K x M x N
        LBinvApsi = tf.matrix_triangular_solve(LB, Apsi)  # K x M x N

        err = self.Y[None] - tf.matmul(
            tf.transpose(LBinvA, perm=[0, 2, 1]),
            tf.einsum('kmn,nd->kmd', LBinvApsi, self.Y))  # K x N
        err = tf.squeeze(err)

        reg1 = tf.reduce_sum(tf.pow(LBinvA, 2), axis=1)  # K x N
        reg2 = self.kern.Kdiag(self.X) / sigma2
        reg2 = reg2 - tf.reduce_sum(tf.pow(A, 2), axis=1)  # K x N

        logW = -0.5 * tf.log(2 * np.pi * sigma2) \
            - 0.5 * tf.pow(err, 2) / sigma2 \
            - 0.5 * reg1 \
            - 0.5 * reg2 \
            + tf.transpose(tf.log(self.W_prior))

        logW = logW - tf.reduce_logsumexp(logW, axis=0, keepdims=True)
        return tf.transpose(logW)

    @params_as_tensors
    def _psi0(self):
        return self.num_data * self.kern.variance

    @params_as_tensors
    def _psi1(self, Kxu):
        psi1 = Kxu[None, :, :] * tf.transpose(self.Wnorm())[:, :, None]
        return psi1 # K x N x M

    @params_as_tensors
    def _psi2(self, Kxu):
        psi2 = tf.einsum('nm,npk->kmp',Kxu, Kxu[:, :, None] * self.Wnorm()[:, None])
        return psi2 # K x M x M

    @params_as_tensors
    def Wnorm(self):
        return normalize(self.W)


class MultipleAssignmentGP(GPModel, SGPRUpperMixin):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W1, W2, kern, feat=None, mean_function=None, Z=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        W1, size NxK
        W2, size NxL

        kern, mean_function are appropriate GPflow objects
        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X)
        Y = DataHolder(Y, fix_shape=True)
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_data = X.shape[0]

        self.W1_prior = Parameter(
            np.ones(W1.shape, dtype=settings.float_type) / W1.shape[1])
        self.W1 = Parameter(W1)
        self.K = W1.shape[1]

        self.W2_prior = Parameter(
            np.ones(W2.shape, dtype=settings.float_type) / W2.shape[1])
        self.W2 = Parameter(W2)
        self.L = W2.shape[1]

        self.num_inducing = Z.shape[0]

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        ND = tf.cast(tf.size(self.Y), settings.float_type)
        D = tf.cast(tf.shape(self.Y)[1], settings.float_type)

        Kxu = self.kern.K(self.X, self.feature.Z)

        psi0 = self._psi0() # scalar
        psi1 = self._psi1(Kxu) # K x L X N x M
        psi2 = self._psi2(Kxu) # K x L x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = tf.tile(L[None, None], [self.K, self.L, 1, 1])

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 1, 3, 2]), lower=True) / sigma

        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)

        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 1, 3, 2]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None, None],
            [self.K, self.L, 1, 1])
        LB = tf.cholesky(B)

        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))

        c = tf.matrix_triangular_solve(
            LB, tf.einsum('klmn,nd->klmd', A, self.Y), lower=True) / sigma

        # KL[q(W) || p(W)]
        W1norm = normalize(self.W1)
        W2norm = normalize(self.W2)

        KL = tf.reduce_sum(W1norm * (tf.log(W1norm) - tf.log(self.W1_prior)))
        KL += tf.reduce_sum(W2norm * (tf.log(W2norm) - tf.log(self.W2_prior)))

        # compute log marginal bound
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        bound -= KL
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        Kxu = self.kern.K(self.X, self.feature.Z)
        Ksu = self.kern.K(Xnew, self.feature.Z)
        Ksu = tf.tile(Ksu[None, None], [self.K, self.L, 1, 1])

        psi0 = self._psi0() # scalar
        psi1 = self._psi1(Kxu) # K x L X N x M
        psi2 = self._psi2(Kxu) # K x L x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = tf.tile(L[None, None], [self.K, self.L, 1, 1])

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(Ksu, perm=[0, 1, 3, 2]), lower=True) / sigma

        Apsi = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 1, 3, 2]), lower=True) / sigma


        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)

        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 1, 3, 2]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None, None],
            [self.K, self.L, 1, 1])
        LB = tf.cholesky(B)


        mu = tf.einsum(
            'klmn,nd->klmd',
            tf.matrix_triangular_solve(LB, Apsi), self.Y)
        mu = tf.matmul(
            tf.transpose(tf.matrix_triangular_solve(LB, A), perm=[0, 1, 3, 2]),
            mu)

        LBinvA = tf.matrix_triangular_solve(LB, A)

        if full_cov:
            var = self.kern.K(Xnew, Xnew)
            var -= tf.matmul(A, A, transpose_a=True) * sigma2
            var += tf.matmul(LBinvA, LBinvA, transpose_a=True) * sigma2

        else:
            var = self.kern.Kdiag(Xnew)[None, None]
            var -= tf.reduce_sum(A ** 2, axis=2) * sigma2
            var += tf.reduce_sum(LBinvA ** 2, axis=2) * sigma2

        return mu, var

    @autoflow()
    @params_as_tensors
    def compute_qu(self):
        """
        Computes the mean and variance of q(u), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, A
        """
        Kxu = self.kern.K(self.X, self.feature.Z)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        Kuu = tf.tile(Kuu[None, None], [self.K, self.L, 1, 1])

        psi1 = self._psi1(Kxu)  # K x L x N x M
        psi2 = self._psi2(Kxu)  # K x L x M x M

        Sig = Kuu + (self.likelihood.variance ** -1) * psi2
        Sig_sqrt = tf.cholesky(Sig)

        Sig_sqrt_Kuu = tf.matrix_triangular_solve(Sig_sqrt, Kuu)

        A = tf.matmul(Sig_sqrt_Kuu, Sig_sqrt_Kuu, transpose_a=True)

        mu = tf.einsum(
            'klmn,nd->klmd',
            tf.matrix_triangular_solve(Sig_sqrt, tf.transpose(psi1, perm=[0, 1, 3, 2])),
            self.Y)
        mu = tf.matmul(tf.transpose(Sig_sqrt_Kuu, perm=[0, 1, 3, 2]), mu) * (self.likelihood.variance ** -1)

        return mu, A

    @autoflow()
    @params_as_tensors
    def update_W(self):
        Kxu = self.kern.K(self.X, self.feature.Z)
        psi1 = self._psi1(Kxu)  # K x L x N x M
        psi2 = self._psi2(Kxu)  # K x L x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(Kxu), lower=True) / sigma

        # Kxu = tf.tile(Kxu[None, None], [self.K, self.L, 1, 1])
        L = tf.tile(L[None, None], [self.K, self.L, 1, 1]) # K x L x M x M
        A = tf.tile(A[None, None], [self.K, self.L, 1, 1])  # K x L X M x N

        Apsi = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 1, 3, 2]), lower=True) / sigma

        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 1, 3, 2]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None, None],
            [self.K, self.L, 1, 1])
        LB = tf.cholesky(B)

        LBinvA = tf.matrix_triangular_solve(LB, A)  # K x L x M x N
        LBinvApsi = tf.matrix_triangular_solve(LB, Apsi)  # K x L x M x N

        err = self.Y[None] - tf.matmul(
            tf.transpose(LBinvA, perm=[0, 1, 3, 2]),
            tf.einsum('klmn,nd->klmd', LBinvApsi, self.Y))  # K x L x N
        err = tf.squeeze(err)

        reg1 = tf.reduce_sum(tf.pow(LBinvA, 2), axis=2)  # K x L x N
        reg2 = self.kern.Kdiag(self.X) / sigma2
        reg2 = reg2 - tf.reduce_sum(tf.pow(A, 2), axis=2)  # K x L x N

        logW = -0.5 * tf.log(2 * np.pi * sigma2) \
            - 0.5 * tf.pow(err, 2) / sigma2 \
            - 0.5 * reg1 \
            - 0.5 * reg2 \
            + tf.transpose(tf.log(self.W1_prior))[:, None] \
            + tf.transpose(tf.log(self.W2_prior))[None]

        logW1 = tf.reduce_sum(
            logW * tf.transpose(normalize(self.W2))[None, :, :],
            axis=1)
        logW1 = logW1 - tf.reduce_logsumexp(logW1, axis=0, keepdims=True)

        logW2 = tf.reduce_sum(
            logW * tf.transpose(normalize(self.W1))[:, None, :],
            axis=0)
        logW2 = logW2 - tf.reduce_logsumexp(logW2, axis=0, keepdims=True)
        return tf.transpose(logW1), tf.transpose(logW2)

    @params_as_tensors
    def _psi0(self):
        return self.num_data * self.kern.variance

    @params_as_tensors
    def _psi1(self, Kxu):
        W1norm = normalize(self.W1)
        W2norm = normalize(self.W2)
        W = W1norm[:, :, None] * W2norm[:, None, :]
        psi1 = Kxu[:, None, None, :] * W[:, :, :, None]
        return tf.transpose(psi1, perm=[1, 2, 0, 3])

    @params_as_tensors
    def _psi2(self, Kxu):
        W1norm = normalize(self.W1)
        W2norm = normalize(self.W2)
        psi2 = tf.einsum(
            'nm,npkg->kgmp', Kxu,
            Kxu[:, :, None, None] *
            W1norm[:, None, :, None] *
            W2norm[:, None, None, :])
        return psi2

    @params_as_tensors
    def _psi2n(self, Kxu):
        W = normalize(self.W1)
        psi2 = tf.einsum(
            'nm,npk->nmp', Kxu,
            Kxu[:, :, None] *
            W1norm[:, None, :])
        return psi2


class GroupedMultipleAssignmentGP(GPModel, SGPRUpperMixin):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W1, W1_index, W2, W2_index, kern, feat=None, mean_function=None, Z=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        W1, size NxK
        W1_index PxL

        W2, size NxL
        W2_index PxL

        kern, mean_function are appropriate GPflow objects
        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X)
        Y = DataHolder(Y, fix_shape=True)
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_data = X.shape[0]

        self.W1_prior = Parameter(
            np.log(np.ones(W1.shape[1], dtype=settings.float_type) / W1.shape[1]),
            trainable=False)
        self.W1 = Parameter(W1)
        self.W1_index = DataHolder(W1_index, dtype=np.int32, fix_shape=True)
        self.K = W1.shape[1]

        self.W2_prior = Parameter(
            np.log(np.ones(W2.shape[1], dtype=settings.float_type) / W2.shape[1]),
            trainable=False)
        self.W2 = Parameter(W2)
        self.W2_index = DataHolder(W2_index, dtype=np.int32, fix_shape=True)
        self.L = W2.shape[1]

        self.num_inducing = Z.shape[0]

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        ND = tf.cast(tf.size(self.Y), settings.float_type)
        D = tf.cast(tf.shape(self.Y)[1], settings.float_type)

        Kxu = self.kern.K(self.X, self.feature.Z)

        psi0 = self._psi0() # scalar
        psi1 = self._psi1(Kxu) # K x L X N x M
        psi2 = self._psi2(Kxu) # K x L x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = tf.tile(L[None, None], [self.K, self.L, 1, 1])

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 1, 3, 2]), lower=True) / sigma

        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)

        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 1, 3, 2]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None, None],
            [self.K, self.L, 1, 1])
        LB = tf.cholesky(B)

        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))

        c = tf.matrix_triangular_solve(
            LB, tf.einsum('klmn,nd->klmd', A, self.Y), lower=True) / sigma

        # KL[q(W) || p(W)]
        W1norm = normalize(self.W1)
        W2norm = normalize(self.W2)

        KL = tf.reduce_sum(W1norm * (tf.log(W1norm) - self.W1_prior[None]))
        KL += tf.reduce_sum(W2norm * (tf.log(W2norm) - self.W2_prior[None]))

        # compute log marginal bound
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        bound -= KL
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        Kxu = self.kern.K(self.X, self.feature.Z)
        Ksu = self.kern.K(Xnew, self.feature.Z)
        Ksu = tf.tile(Ksu[None, None], [self.K, self.L, 1, 1])

        psi0 = self._psi0() # scalar
        psi1 = self._psi1(Kxu) # K x L X N x M
        psi2 = self._psi2(Kxu) # K x L x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = tf.tile(L[None, None], [self.K, self.L, 1, 1])

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(Ksu, perm=[0, 1, 3, 2]), lower=True) / sigma

        Apsi = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 1, 3, 2]), lower=True) / sigma


        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)

        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 1, 3, 2]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None, None],
            [self.K, self.L, 1, 1])
        LB = tf.cholesky(B)


        mu = tf.einsum(
            'klmn,nd->klmd',
            tf.matrix_triangular_solve(LB, Apsi), self.Y)
        mu = tf.matmul(
            tf.transpose(tf.matrix_triangular_solve(LB, A), perm=[0, 1, 3, 2]),
            mu)

        LBinvA = tf.matrix_triangular_solve(LB, A)

        if full_cov:
            var = self.kern.K(Xnew, Xnew)
            var -= tf.matmul(A, A, transpose_a=True) * sigma2
            var += tf.matmul(LBinvA, LBinvA, transpose_a=True) * sigma2

        else:
            var = self.kern.Kdiag(Xnew)[None, None]
            var -= tf.reduce_sum(A ** 2, axis=2) * sigma2
            var += tf.reduce_sum(LBinvA ** 2, axis=2) * sigma2

        return mu, var

    @autoflow()
    @params_as_tensors
    def compute_qu(self):
        """
        Computes the mean and variance of q(u), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, A
        """
        Kxu = self.kern.K(self.X, self.feature.Z)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        Kuu = tf.tile(Kuu[None, None], [self.K, self.L, 1, 1])

        psi1 = self._psi1(Kxu)  # K x L x N x M
        psi2 = self._psi2(Kxu)  # K x L x M x M

        Sig = Kuu + (self.likelihood.variance ** -1) * psi2
        Sig_sqrt = tf.cholesky(Sig)

        Sig_sqrt_Kuu = tf.matrix_triangular_solve(Sig_sqrt, Kuu)

        A = tf.matmul(Sig_sqrt_Kuu, Sig_sqrt_Kuu, transpose_a=True)

        mu = tf.einsum(
            'klmn,nd->klmd',
            tf.matrix_triangular_solve(Sig_sqrt, tf.transpose(psi1, perm=[0, 1, 3, 2])),
            self.Y)
        mu = tf.matmul(tf.transpose(Sig_sqrt_Kuu, perm=[0, 1, 3, 2]), mu) * (self.likelihood.variance ** -1)

        return mu, A

    @autoflow()
    @params_as_tensors
    def update_W(self):
        Kxu = self.kern.K(self.X, self.feature.Z)
        psi1 = self._psi1(Kxu)  # K x L x N x M
        psi2 = self._psi2(Kxu)  # K x L x M x M

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)
        L = tf.tile(L[None, None], [self.K, self.L, 1, 1]) # K x L x M x M

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        Kxu = tf.tile(Kxu[None, None], [self.K, self.L, 1, 1])

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(
            L, tf.transpose(Kxu, perm=[0, 1, 3, 2]), lower=True) / sigma

        Apsi = tf.matrix_triangular_solve(
            L, tf.transpose(psi1, perm=[0, 1, 3, 2]), lower=True) / sigma

        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(
            L, tf.transpose(tmp, perm=[0, 1, 3, 2]), lower=True) / sigma2
        B = AAT + tf.tile(tf.eye(
            self.num_inducing, dtype=settings.float_type)[None, None],
            [self.K, self.L, 1, 1])
        LB = tf.cholesky(B)

        LBinvA = tf.matrix_triangular_solve(LB, A)  # K x L x M x N
        LBinvApsi = tf.matrix_triangular_solve(LB, Apsi)  # K x L x M x N

        err = self.Y[None] - tf.matmul(
            tf.transpose(LBinvA, perm=[0, 1, 3, 2]),
            tf.einsum('klmn,nd->klmd', LBinvApsi, self.Y))  # K x L x N
        err = tf.squeeze(err)

        reg1 = tf.reduce_sum(tf.pow(LBinvA, 2), axis=2)  # K x L x N
        reg2 = self.kern.Kdiag(self.X) / sigma2
        reg2 = reg2 - tf.reduce_sum(tf.pow(A, 2), axis=2)  # K x L x N

        logW = -0.5 * tf.log(2 * np.pi * sigma2) \
            - 0.5 * tf.pow(err, 2) / sigma2 \
            - 0.5 * reg1 \
            - 0.5 * reg2 \
            #+ tf.transpose(tf.log(self.W1_prior))[:, None] \
            #+ tf.transpose(tf.log(self.W2_prior))[None]


        W1norm = tf.gather(normalize(self.W1), self.W1_index, axis=0)
        W2norm = tf.gather(normalize(self.W2), self.W2_index, axis=0)

        logW1 = tf.reduce_sum(
            logW * tf.transpose(W2norm)[None, :, :],
            axis=1) # K x N
        logW1_parts = tf.dynamic_partition(
            tf.transpose(logW1), self.W1_index, num_partitions=self.W1.shape[0])
        logW1 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW1_parts])
        logW1 = logW1 + self.W1_prior[None]
        logW1 = logW1 - tf.reduce_logsumexp(logW1, axis=1, keepdims=True)


        logW2 = tf.reduce_sum(
            logW * tf.transpose(W1norm)[:, None, :],
            axis=0) # L x N
        logW2_parts = tf.dynamic_partition(
            tf.transpose(logW2), self.W2_index, num_partitions=self.W2.shape[0])
        logW2 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW2_parts])

        logW2 = logW2 + self.W2_prior
        logW2 = logW2 - tf.reduce_logsumexp(logW2, axis=1, keepdims=True)

        return logW1, logW2

    @params_as_tensors
    def _psi0(self):
        return self.num_data * self.kern.variance

    @params_as_tensors
    def _psi1(self, Kxu):
        W1norm = tf.gather(normalize(self.W1), self.W1_index, axis=0)
        W2norm = tf.gather(normalize(self.W2), self.W2_index, axis=0)
        W = W1norm[:, :, None] * W2norm[:, None, :]
        psi1 = Kxu[:, None, None, :] * W[:, :, :, None]
        return tf.transpose(psi1, perm=[1, 2, 0, 3])

    @params_as_tensors
    def _psi2(self, Kxu):
        W1norm = tf.gather(normalize(self.W1), self.W1_index, axis=0)
        W2norm = tf.gather(normalize(self.W2), self.W2_index, axis=0)
        psi2 = tf.einsum(
            'nm,npkg->kgmp', Kxu,
            Kxu[:, :, None, None] *
            W1norm[:, None, :, None] *
            W2norm[:, None, None, :])
        return psi2


def normalize(W, epsilon=1e-3):
    expW = tf.exp(W) + epsilon
    return expW / tf.reduce_sum(expW, axis=1)[:, None]


def block_diagonal(matrices, dtype=settings.float_type):
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
        ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked
