from posteriordb import PosteriorDatabase
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax

def get_target(data):
    k = data["K"]
    y = jnp.array(data["Y"])
    x = jnp.array(data["X"])
    x = (x - jnp.mean(x, axis=0))[:, 1:] # mean center, don't include bias

    @jax.vmap
    def unormalised_posterior(params):
        intercept = params[0]
        beta = params[1:k]
        sigma = jnp.exp(params[-1])
        means = jnp.dot(beta, x.T) # (N,)

        log_likelihoods = jax.vmap(lambda y, mu: jstats.norm.logpdf(y, loc=mu, scale=sigma))(y, means)

        return (
            jnp.sum(jstats.norm.logpdf(beta)) +
            jstats.t.logpdf(intercept, df=3., loc=8., scale=10.) +
            jstats.t.logpdf(sigma, df=3., loc=0., scale=10.) - jnp.log(jnp.array(0.5)) + # folded studentT
            jnp.sum(log_likelihoods)
        )
        
    return unormalised_posterior, k + 2