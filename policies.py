import numpy as np
import tensorflow as tf
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from tensorflow.contrib import layers

#
class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x: x)
            vf = fc(h5, 'v', 1, act=lambda x: x)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x: x)
            vf = fc(h5, 'v', 1, act=lambda x: x)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x, init_scale=0.01)
            vf = fc(h4, 'v', 1, act=lambda x: x)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space=2, ac_space=2, nbatch=1, nsteps=None, reuse=False):  # pylint: disable=W0613
        # ob_shape = (nbatch,) + ob_space.shape
        # actdim = ac_space.shape[0]

        ob_shape = [nbatch, ob_space]
        actdim = ac_space

        X = tf.placeholder(tf.float32, shape=ob_shape, name='Ob')  # obs
        # regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        regularizer = layers.l2_regularizer(0.1)
        with tf.variable_scope("model", reuse=reuse, regularizer=regularizer):
            h1 = fc(X, 'pi_fc1', nh=50, init_scale=np.sqrt(2), act=tf.nn.relu)
            # h2 = fc(h1, 'pi_fc2', nh=50, init_scale=np.sqrt(2), act=tf.nn.relu)
            # h3 = fc(h2, 'pi_fc3', nh=50, act=tf.nn.relu)
            # h4 = fc(h3, 'pi_fc4', nh=50, act=tf.nn.relu)
            # h5 = fc(h4, 'pi_fc5', nh=50, act=tf.nn.relu)
            pi = fc(h1, 'pi', actdim, act=tf.nn.relu, init_scale=0.01)

            h1 = fc(X, 'vf_fc1', nh=50, init_scale=np.sqrt(2), act=tf.nn.relu)
            # h2 = fc(h1, 'vf_fc2', nh=50, init_scale=np.sqrt(2), act=tf.nn.relu)
            # h3 = fc(h2, 'vf_fc3', nh=50, act=tf.nn.relu)
            # h4 = fc(h3, 'vf_fc4', nh=50, act=tf.nn.relu)
            # h5 = fc(h4, 'vf_fc5', nh=50, act=tf.nn.relu)
            vf = fc(h1, 'vf', 1, act=tf.nn.relu)[:, 0]

            # h1 = fc(X, 'pi_fc1', nh=128, init_scale=np.sqrt(2), act=tf.tanh)
            # h2 = fc(h1, 'pi_fc2', nh=128, init_scale=np.sqrt(2), act=tf.tanh)
            # pi = fc(h2, 'pi', actdim, act=lambda x:x, init_scale=0.01)
            # h1 = fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2), act=tf.tanh)
            # h2 = fc(h1, 'vf_fc2', nh=128, init_scale=np.sqrt(2), act=tf.tanh)

            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        # pdparam = [pi, 0]
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = DiagGaussianPdType(ac_space)
        # self.pd => DiagGaussianPd
        self.pd = self.pdtype.pdfromflat(pdparam)
        pdtest = self.pd.sam - self.pd.mean

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        # def step(ob, *_args, **_kwargs):
        def step(ob):
            a, v, pdtes, neglogp = sess.run([a0, vf, pdtest, neglogp0], {X: ob})

            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


# 得到 h[1,nh]
def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value

        w = tf.get_variable("w", [nin, nh], initializer=tf.constant_initializer(0.0))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h


# def fc_2(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
#     with tf.variable_scope(scope):
#         nin = x.get_shape()[1].value
#
#         w = tf.get_variable("w", [nin, nh], initializer=tf.constant_initializer(0.0))
#         #b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
#         z = tf.matmul(x, w)+b
#         return z

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            # np.pro 所有的相乘
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        # a[1,flat_shape]
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])

    else:
        raise NotImplementedError


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        # logstd = 0; std = 1;
        self.std = tf.exp(logstd)
        self.sam = mean

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    # e^neglogp ，相当于下面的+改成*
    def neglogp(self, x):
        return 0.5 * Usum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + Usum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return Usum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (
                2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return Usum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        self.sam = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
        # self.sam = self.mean
        return self.sam

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def Usum(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)


class PIDPolicy(object):
    def __init__(self, sess, ob_space=2, ac_space=2, nbatch=1, nsteps=None, reuse=False):
        K_n = 1
        K_T = 1
        ob_shape = [nbatch, ob_space]
        actdim = ac_space
        X = tf.placeholder(tf.float32, shape=ob_shape, name='Ob')  # obs
        # regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # [I1_old D1_old I2_old D2_old]
        PID_old = tf.placeholder(dtype=tf.float32, shape=[nbatch, 4], name='PID_OLD')

        regularizer = layers.l2_regularizer(0.1)
        with tf.variable_scope("model", reuse=reuse, regularizer=regularizer):
            # const_1 = tf.Variable(-1, trainable=False, dtype=tf.float32)
            const_1 = tf.get_variable("const_1", dtype=tf.float32, initializer=[-1.0])
            # const_2 = tf.Variable(1, trainable=False, dtype=tf.float32)
            const_2 = tf.get_variable("const_2", dtype=tf.float32, initializer=[1.0])

            n_forward = K_n * X[:, 0]
            T_forward = K_T * X[:, 2]
            n_forward = tf.reshape(n_forward, shape=[nbatch,1])
            T_forward = tf.reshape(T_forward, shape=[nbatch,1])
            # n_set的P1,I1,D1
            w_P1 = tf.get_variable("w_P1", [2, 1], initializer=tf.constant_initializer(0.0))
            w_I1 = tf.get_variable("w_I1", [2, 1], initializer=tf.constant_initializer(0.0))
            w_D1 = tf.get_variable("w_D1", [2, 1], initializer=tf.constant_initializer(0.0))
            # h1[1,1]
            P1_in = tf.matmul(X[:, 0:2], w_P1)
            P1_out = P1_in
            P1_out = P1_out * tf.to_float(tf.greater(P1_out, const_1)) + const_1 * tf.to_float(
                tf.greater(const_1, P1_out))
            P1_out = P1_out * tf.to_float(tf.greater(const_2, P1_out)) + const_2 * tf.to_float(
                tf.greater(P1_out, const_2))

            I1_in = tf.matmul(X[:, 0:2], w_I1)
            I1_old = tf.reshape(PID_old[:, 0], shape=[nbatch, 1])
            I1_out = I1_in + I1_old
            I1_out = I1_out * tf.to_float(tf.greater(I1_out, const_1)) \
                     + const_1 * tf.to_float(tf.greater(const_1, I1_out))
            I1_out = I1_out * tf.to_float(tf.greater(const_2, I1_in)) \
                     + const_2 * tf.to_float(tf.greater(I1_out, const_2))

            D1_in = tf.matmul(X[:, 0:2], w_D1)
            D1_old = tf.reshape(PID_old[:, 1], shape=[nbatch, 1])
            D1_out = D1_in - D1_old
            D1_out = D1_out * tf.to_float(tf.greater(D1_out, const_1)) \
                     + const_1 * tf.to_float(tf.greater(const_1, D1_out))
            D1_out = D1_out * tf.to_float(tf.greater(const_2, D1_in)) \
                     + const_2 * tf.to_float(tf.greater(D1_out, const_2))

            # T_set的P2 I2 D2
            w_P2 = tf.get_variable("w_P2", [2, 1], initializer=tf.constant_initializer(0.0))
            w_I2 = tf.get_variable("w_I2", [2, 1], initializer=tf.constant_initializer(0.0))
            w_D2 = tf.get_variable("w_D2", [2, 1], initializer=tf.constant_initializer(0.0))

            P2_in = tf.matmul(X[:, 2:4], w_P2)
            P2_out = P2_in
            P2_out = P2_out * tf.to_float(tf.greater(P2_out, const_1)) + const_1 * tf.to_float(
                tf.greater(const_1, P2_out))
            P2_out = P2_out * tf.to_float(tf.greater(const_2, P2_in)) + const_2 * tf.to_float(
                tf.greater(P2_in, const_2))

            I2_in = tf.matmul(X[:, 2:4], w_I2)
            I2_old = tf.reshape(PID_old[:, 2], shape=[nbatch, 1])
            I2_out = I2_in + I2_old
            I2_out = I2_out * tf.to_float(tf.greater(I2_out, const_1)) \
                     + const_1 * tf.to_float(tf.greater(const_1, I2_out))
            I2_out = I2_out * tf.to_float(tf.greater(const_2, I2_in)) \
                     + const_2 * tf.to_float(tf.greater(I2_out, const_2))

            D2_in = tf.matmul(X[:, 0:2], w_D2)
            D2_old = tf.reshape(PID_old[:, 3], shape=[nbatch, 1])
            D2_out = D2_in - D2_old
            D2_out = D2_out * tf.to_float(tf.greater(D2_out, const_1)) \
                     + const_1 * tf.to_float(tf.greater(const_1, D2_out))
            D2_out = D2_out * tf.to_float(tf.greater(const_2, D2_in)) \
                     + const_2 * tf.to_float(tf.greater(D2_out, const_2))

            concat_PID = tf.concat([P1_out, I1_out, D1_out, P2_out, I2_out, D2_out], axis=1)

            w_hidden1 = fc(concat_PID, 'w_hidden1', nh=15, act=tf.nn.relu)
            w_hidden2 = fc(w_hidden1, 'w_hidden2', nh=30, act=tf.nn.relu)
            PID_out = fc(w_hidden2, 'w_pi', nh=2, act=tf.nn.relu)
            # w_U1 = tf.get_variable("w_U1", [6, 1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            # U1_in = tf.matmul(concat_PID, w_U1)
            # w_U2 = tf.get_variable("w_U2", [6, 1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            # U2_in = tf.matmul(concat_PID, w_U2)
            # pi = tf.concat([U1_in, U2_in], axis=1)

            forward_out = tf.concat([n_forward, T_forward], axis=1)
            pi = PID_out + forward_out

            PID0 = tf.concat([I1_out, D1_in, I2_out, D2_in], axis=0)

            logstd = tf.get_variable(name="logstd", shape=[1, 2],
                                     initializer=tf.zeros_initializer())

            t3 = tf.reshape(X[:, 0:4], shape=[nbatch, 4])

            h1_v = fc(t3, 'vf_fc1', nh=50, init_scale=np.sqrt(2), act=tf.nn.relu)
            # h2 = fc(h1, 'vf_fc2', nh=50, init_scale=np.sqrt(2),  act=tf.nn.relu)
            # h3 = fc(h2, 'vf_fc3', nh=50, act=tf.nn.relu)
            # h4 = fc(h3, 'vf_fc4', nh=50, act=tf.nn.relu)
            # h5 = fc(h4, 'vf_fc5', nh=50, act=tf.nn.relu)
            vf = fc(h1_v, 'vf', 1, act=tf.nn.relu)[:, 0]

            # 代表被试件的排量q
            t1 = tf.reshape(X[:, 5], shape=[nbatch, 1])
            # 代表系统压力p
            t2 = tf.reshape(X[:, 4], shape=[nbatch, 1])
            X_relief = tf.concat([t1, t2], axis=1)
            h1_relief = fc(X_relief, 'pi_fc1', nh=50, init_scale=np.sqrt(2), act=tf.nn.relu)
            # h2 = fc(h1, 'pi_fc2', nh=50, init_scale=np.sqrt(2), act=tf.nn.relu)
            # h3 = fc(h2, 'pi_fc3', nh=50, act=tf.nn.relu)
            # h4 = fc(h3, 'pi_fc4', nh=50, act=tf.nn.relu)
            # h5 = fc(h4, 'pi_fc5', nh=50, act=tf.nn.relu)
            q0_peishi = fc(h1_relief, 'pi', 1, act=tf.nn.relu, init_scale=0.01)

        # pdparam = [pi, 0]
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = DiagGaussianPdType(ac_space)
        # self.pd => DiagGaussianPd
        self.pd = self.pdtype.pdfromflat(pdparam)

        # a0[n_ctrol, p_relief, q_test]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        # neglogp0 = tf.concat([neglogp0,tf.Variable(1)], axis=1)
        self.initial_state = None
        PID0 = tf.concat([I1_out, D1_in, I2_out, D2_in], axis=1)

        # def step(ob, *_args, **_kwargs):
        def step(ob, pid):
            a, v, PID_sess, neglogp, q_peishi = sess.run([a0, vf, PID0, neglogp0, q0_peishi],
                                                         {X: ob, PID_old: pid})  # , PID_old:pid


            return a, v, PID_sess, neglogp, q_peishi

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.PID_old = PID_old
        self.q0_peishi = q0_peishi


def test():
    # b = tf.get_variable("x",dtype=tf.float32,
    #                     initializer=[1.0,2])
    # X = tf.get_variable(dtype=tf.float32, name='Ob', initializer=[[1.0, 2.0]])
    # X = tf.placeholder(tf.float32, shape=[1,3], name='Ob')
    X = tf.get_variable(dtype=tf.float32, name='Ob', shape=[2, 3], initializer=tf.constant_initializer(0.0))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(X)
        print(a)

        # a=MlpPolicy(sess)
        # sess.run(tf.global_variables_initializer())
        # X = sess.run(X)
        # c,d,e,f = a.step(X)
        # print(c)
        # print(d)
        # print(e)
        # print(f)
        # sess.run(tf.global_variables_initializer())
        # b0 = sess.run(b)
        # print(sess.run(X, feed_dict={X: b}))


if __name__ == '__main__':
    test()
