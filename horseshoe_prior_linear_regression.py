from jax.lib import xla_bridge
from numpy import random
import numpy as np
import random
import optax
import jax
from functools import partial
from jax import jit
from scipy.special import *
import math
import jax.numpy as jnp

np.random.seed(0)
random.seed(0)
print("Jax on", xla_bridge.get_backend().platform)


# horseshoe prior linear regression
class HS:

    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.num = X.shape[1]

        self.ln_s0 = 0.0
        self.bg = 1e-5
        self.b0 = 1e-5
        self.PI = math.pi
        self.ONE = 1.0
        self.HALF = 0.5

    def update_aux(self, params_HS):
        u_v = params_HS['u_v']
        ln_s_v = params_HS['ln_s_v']
        M_mu = params_HS['M_mu']
        M_U = params_HS['M_U']
        _tau_p = jnp.exp(-M_mu[self.num:, 0] + 0.5 * (jnp.diag(M_U)[self.num:])**2) + 1 / self.b0**2
        _v_p = jnp.exp(-u_v + 0.5 * jnp.exp(ln_s_v)) + 1 / self.bg**2
        return _tau_p, _v_p

    def entropy_post(self, params_HS):
        u_v = params_HS['u_v']
        ln_s_v = params_HS['ln_s_v']
        M_U = params_HS['M_U']
        U = jnp.matmul(jnp.tril(M_U), jnp.tril(M_U).T)
        ent_v = (0.5 * ln_s_v + u_v).sum()
        ent_M = 0.5 * jnp.linalg.slogdet(U)[1]
        return ent_v + ent_M

    def exp_prior(self, params_HS, _tau_p, _v_p):
        u_v = params_HS['u_v']
        ln_s_v = params_HS['ln_s_v']
        M_mu = params_HS['M_mu']
        M_U = params_HS['M_U']
        U = jnp.matmul(jnp.tril(M_U), jnp.tril(M_U).T)
        exp_tau = (-1.5 * M_mu[self.num:, 0] - self.ONE / _tau_p * (jnp.exp(-M_mu[self.num:, 0] + 0.5 * (jnp.diag(M_U)[self.num:]**2)))).sum()
        exp_v = (-1.5 * u_v - self.ONE / _v_p * (jnp.exp(-u_v + 0.5 * jnp.exp(ln_s_v)))).sum()
        exp_w = (-0.5 * (M_mu[:self.num, 0]**2) / jnp.exp(self.ln_s0)).sum() + (-0.5 * (jnp.trace(U[:self.num, :self.num])) / jnp.exp(self.ln_s0)).sum()
        return exp_tau + exp_v + exp_w

    def loss_HS(self, params_HS, _tau_p, _v_p, key, X, y):
        key, sub_key1 = jax.random.split(key)
        key, sub_key2 = jax.random.split(key)
        u_v = params_HS['u_v']
        ln_s_v = params_HS['ln_s_v']
        M_mu = params_HS['M_mu']
        M_U = params_HS['M_U']
        L = jnp.tril(M_U)
        s_M = M_mu + jnp.matmul(L, jax.random.normal(sub_key1, shape=(self.num * 2, 1)))
        s_tau = jnp.exp(s_M[self.num:, 0]).reshape(1, -1)
        s_w = s_M[:self.num, ].reshape(1, -1)
        s_v = jnp.exp(u_v + jax.random.normal(sub_key2, shape=(1, )) * jnp.exp(ln_s_v * 0.5))
        weights = s_tau**0.5 * s_w * s_v**0.5
        llh = -10 * jnp.sum(((weights * X).sum(axis=-1, keepdims=True) - y)**2)
        elbo = self.entropy_post(params_HS) + self.exp_prior(params_HS, _tau_p, _v_p) + llh
        return -elbo.sum()

    @partial(jit, static_argnums=(0, 1))
    def step_HS(self, optimizer, params_HS, opt_state, _tau_p, _v_p, key, X, y):
        loss, d_params = jax.value_and_grad(self.loss_HS)(params_HS, _tau_p, _v_p, key, X, y)
        updates, opt_state = optimizer.update(d_params, opt_state, params_HS)
        params_HS = optax.apply_updates(params_HS, updates)
        return params_HS, opt_state, loss

    def train(self):
        key = jax.random.PRNGKey(0)
        params_HS = {
            "u_v": np.array([0.0]),
            "ln_s_v": np.array([0.0]),
            'M_mu': np.zeros((self.num * 2, 1)),
            'M_U': np.eye(self.num * 2),
        }
        optimizer_HS = optax.adam(1e-3)
        opt_state_HS = optimizer_HS.init(params_HS)
        for _ in range(1):
            for i in range(1000000):
                key, sub_key = jax.random.split(key)
                _tau_p, _v_p = self.update_aux(params_HS)
                params_HS, opt_state_HS, loss = self.step_HS(optimizer_HS, params_HS, opt_state_HS, _tau_p, _v_p, sub_key, self.X, self.y)
                u_v = params_HS['u_v']
                M_mu = params_HS['M_mu']
                s_M = M_mu
                s_tau = jnp.exp(s_M[self.num:, 0]).reshape(1, -1)
                s_w = s_M[:self.num, ].reshape(1, -1)
                s_v = jnp.exp(u_v)
                weights = s_tau**0.5 * s_w * s_v**0.5
                weights = weights.reshape(-1)
                if (i + 1) % 2000 == 0:
                    print("Loss ", loss, " all weights ", weights, "\n selected weights ", weights[np.where(abs(weights) > 1e-3)[0]], "\n indices ", np.where(abs(weights) > 1e-3)[0])


def test_simulation():
    n = 40
    d = 100
    w = np.zeros([d, 1])
    X = np.random.randn(n, d)
    noise_std = 0.3
    w[0] = 1.0
    w[1] = 1.0
    w[2] = -1.0
    w[3] = -1.0
    w[4] = 0.5
    print("True weights ", w[:5].reshape(1, -1))
    w = w.reshape(1, -1)
    y = (X * w).sum(axis=-1, keepdims=True) + noise_std * np.random.randn(n, 1)
    y = y.reshape([-1])
    hs = HS(X, y)
    hs.train()


test_simulation()