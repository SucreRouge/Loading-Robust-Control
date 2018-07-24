import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
import logger
from collections import deque
import matplotlib.pyplot as plt
import threading
from policies import fc


# 与env无关
class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act=1, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()
        # sess = tf.Session()
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)
        # train_model = act_model

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # OLD V Prediction
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # clip range
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        # p1/p2
        # -neglogpac-(-OLDNEGLOGPAC)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        # ent_coef = 0
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef  # + 0.1*regularization_loss
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, pidold, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.PID_old: pidold, train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            # keke
            # temp = sess.run([neglogpac, OLDNEGLOGPAC, ADV, loss, pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
            #                 td_map)
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        sess.run(tf.global_variables_initializer())
        tf.global_variables_initializer().run(session=sess) # pylint: disable=E1101

        # try:
        #
        # except:
        #     print("e")





#
class Runner(object):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        # nenv = env.num_envs ***
        nenv = 1
        # ***
        ob_shape = (1, self.env.ob_space)
        # self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs = np.zeros(ob_shape, dtype=np.float32)
        # self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.Tperiod = 0.1
        self.t1 = 10 * np.random.random()
        self.t2 = 10 * np.random.random()

    def run(self, tr=None, flag=None):
        self.t1 = 10 * np.random.random()
        self.t2 = 10 * np.random.random()
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_pidold = [], [], [], [], [], [], []
        out_nset, out_Tset = [], []
        mb_states = self.states
        epinfos = []
        n_real = []
        T_test = []

        PID_old = np.array([[0, 0, 0, 0.0]])

        # nsteps是一次epoch的step值
        for _t in range(self.nsteps):
            # tr是true_env的意思
            # if flag == True:
            #     tr.render()
            # model.step 从policy类中得到，由神经网络计算得到的model_based
            # actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            actions, values, PID_sess, neglogpacs, q_peishi = self.model.step(self.obs, PID_old)
            X = self.obs
            PID_old = PID_sess

            #actions = np.concatenate([actions, q_peishi], axis=1)

            if actions[0][0] < 300:
                actions[0][0] = 0
            if actions[0][0] > 3000:
                actions[0][0] = 3000

            if actions[0][1] < 0:
                actions[0][1] = 0
            if actions[0][1] > 30:
                actions[0][1] = 30

            if q_peishi < 0:
                q_peishi = 0
            if q_peishi > 125:
                q_peishi = 125

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_pidold.append(PID_old)

            n_set = 500 * np.sin(0.1 * 2 * np.pi * self.Tperiod * _t + self.t1) + 1000
            # p_set = 10
            T_set = 25 * np.sin(0.1 * 2 * np.pi * self.Tperiod * _t + self.t2) + 75
            out_nset.append(n_set)
            out_Tset.append(T_set)

            out_actions = [n_set, T_set]
            # env.step(actions)
            self.obs[:], rewards, self.dones, infos, scope = self.env.runsampletime(0.1, actions, q_peishi, out_actions)
            n_real.append(scope[0])
            T_test.append(scope[1])

            # for info in infos:
            #     maybeepinfo = info.get('episode')
            #     if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # *********
        if flag == True:
            plt.clf()
            # def threadplot():
            # plt.figure()
            # figure分成3行3列, 取得第一个子图的句柄, 第一个子图跨度为1行3列, 起点是表格(0, 0)
            plt.subplot(211)
            plt.plot(np.array(range(self.nsteps)) * 0.1, out_nset)
            plt.plot(np.array(range(self.nsteps)) * 0.1, n_real)
            plt.title("q_motor&q_trace; Q_relief; out_pset&p_sys")
            plt.subplot(212)
            plt.plot(np.array(range(self.nsteps)) * 0.1, out_Tset)
            plt.plot(np.array(range(self.nsteps)) * 0.1, T_test)

            plt.pause(0.3)
            # thread1 = threading.Thread(target=threadplot,args=())
            # thread1.start()

            # plt.close()
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_pidold = np.asarray(mb_pidold, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            # 单次t时刻的adv, adv = R + gamma*V[T+1] - V[t]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            # advt = adv + 下一个时刻的advt。开始的advt包含了后面的所有信息
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        # mb_returns,mb_dones
        # *不能单独使用 a=[1,2],b=[3,4]
        # *a错误,*a,b正确

        return (*map(sf01, (mb_pidold, mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    # arr.shape不能单独使用,要结合shape[0]等
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val

    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, tr=None):
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # nenvs = env.num_envs
    nenvs = 1
    # *****
    ob_space = env.ob_space
    ac_space = env.ac_space
    nbatch = nenvs * nsteps
    # nsteps=128, nminibatches=4
    nbatch_train = nbatch // nminibatches

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)
    # if save_interval and logger.get_dir():
    #     import cloudpickle
    #     with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
    #         fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps // nbatch
    for update in range(1, nupdates + 1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        print(lrnow)
        cliprangenow = cliprange(frac)

        # 决定是否render出来
        flagrender = False
        if (update % 2 == 0):
            print(update)
            flagrender = True
        # print(flagrender)
        pidolds, obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(tr=tr,
                                                                                       flag=flagrender)  # pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None:  # nonrecurrent version <--
            inds = np.arange(nbatch)  # <--
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                # start = 0, stop = nenvs, interval = envsperbatch
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (pidolds, obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (pidolds, obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # ev = explained_variance(values, returns)
            # logger.logkv is used together with logger.dumpkvs
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            # logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    # env.close()
    print("Training is over.")


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
