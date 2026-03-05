__all__ = ['sample_elements', 'sample_xv', 'model_relative_astrometry_radec_nocorr',
           'sample_xv_via_period', 'get_init_elements']

import numpyro
import numpyro.distributions as dist
from jnkepler.jaxttv.conversion import G
from jnkepler.keplerian import xv_to_elements, elements_to_xv
import jax.numpy as jnp
from jnkepler.keplerian.jacobian import det_jkep_pm


def model_relative_astrometry_radec_nocorr(t, ra_obs, ra_err, dec_obs, dec_err, mass_obs, mass_err, parallax_obs, parallax_err,
                                           t_ref=None, log_period=True, pmin=1e2, pmax=1e6, sample_method='elements', ignore_obs=False):
    """NumPyro model for relative astrometry without RA/Dec covariance.
    Args:
        t: Observation times.
        ra_obs: Observed relative RA offsets.
        ra_err: Uncertainties in ra_obs.
        dec_obs: Observed relative Dec offsets.
        dec_err: Uncertainties in dec_obs.
        mass_obs: Gaussian prior mean for total mass.
        mass_err: Gaussian prior std for total mass.
        parallax_obs: Gaussian prior mean for parallax (mas).
        parallax_err: Gaussian prior std for parallax (mas).
        t_ref: Reference time for orbital parameterization. If None, use t[0].
        log_period: If True, sample log(period) when applicable.
        pmin: Minimum period prior bound (days).
        pmax: Maximum period prior bound (days).
        sample_method: Orbital sampling method; one of 'elements', 'xv', or 'xvp'.
        ignore_obs: If True, skip conditioning on astrometric observations.

    Returns:
        None
    """
    if t_ref is None:
        t_ref = t[0]

    mass = numpyro.sample("mass", dist.TruncatedNormal(
        mass_obs, mass_err, low=0.))
    parallax = numpyro.sample("parallax", dist.TruncatedNormal(
        parallax_obs, parallax_err, low=0.))  # mas

    if sample_method == 'elements':
        params = sample_elements(t_ref, pmin=pmin, pmax=pmax)
    else:
        if sample_method == 'xv':
            params = sample_xv(t_ref, mass, pmin=pmin, pmax=pmax)
        elif sample_method == 'xvp':
            params = sample_xv_via_period(
                t_ref, mass, log_period=log_period, pmin=pmin, pmax=pmax)
        else:
            raise ValueError(
                f"Unknown sample_method '{sample_method}'. "
                "Expected one of ['elements', 'xv', 'xvp']."
            )

        # prior correction
        Jkep = det_jkep_pm(params) / jnp.sin(params['inc'])
        if (sample_method == 'xv') and log_period:
            Jkep *= params['period']
        numpyro.factor("logjac", -jnp.log(jnp.abs(Jkep)))

    params['mass'] = mass

    out = elements_to_xv(t, params)  # shape (len(t), 3)
    x_vect, v_vect = out['x'], out['v']
    numpyro.deterministic("xref", x_vect[0])
    numpyro.deterministic("vref", v_vect[0])

    x_au, y_au = x_vect[:, 0], x_vect[:, 1]
    x_mas = numpyro.deterministic("x_mas", x_au * parallax)
    y_mas = numpyro.deterministic("y_mas", y_au * parallax)

    # (+x, +y) corresponds to (+Dec, +RA), respectively. Thus +z points toward the observer.
    if not ignore_obs:
        numpyro.sample("dec", dist.Normal(
            loc=x_mas, scale=dec_err), obs=dec_obs)
        numpyro.sample("ra", dist.Normal(loc=y_mas, scale=ra_err), obs=ra_obs)


def sample_elements(t_ref, pmin=1e2, pmax=1e6, emax=0.99, log_period=True):
    """Sample orbital elements in a NumPyro model.

    Args:
        t_ref: Reference time.
        pmin: Minimum period prior bound (days).
        pmax: Maximum period prior bound (days).
        emax: Maximum eccentricity prior bound.
        log_period: If True, sample log(period) instead of period.

    Returns:
        dict: Sampled orbital parameters.
    """
    if log_period:
        logperiod = numpyro.sample(
            "logperiod", dist.Uniform(jnp.log(pmin), jnp.log(pmax)))
        period = numpyro.deterministic("period", jnp.exp(logperiod))
    else:
        period = numpyro.sample("period", dist.Uniform(pmin, pmax))
        logperiod = numpyro.deterministic("logperiod", jnp.log(period))
    n = 2 * jnp.pi / period  # mean motion

    ecc = numpyro.sample("ecc", dist.Uniform(0, emax))

    cosi = numpyro.sample("cosi", dist.Uniform(-1, 1))
    inc = numpyro.deterministic("inc", jnp.arccos(cosi))

    cosL = numpyro.sample("cosL", dist.Normal())
    sinL = numpyro.sample("sinL", dist.Normal())
    lnode = numpyro.deterministic("lnode", jnp.arctan2(sinL, cosL))

    cosw = numpyro.sample("cosw", dist.Normal())
    sinw = numpyro.sample("sinw", dist.Normal())
    omega = numpyro.deterministic("omega", jnp.arctan2(sinw, cosw))

    cosm = numpyro.sample("cosm", dist.Normal())
    sinm = numpyro.sample("sinm", dist.Normal())
    mean_anom = numpyro.deterministic("mean_anom", jnp.arctan2(sinm, cosm))
    tau = numpyro.deterministic("tau", t_ref - mean_anom / n)
    numpyro.deterministic("M", n * (t_ref - tau))  # bookkeeping

    params = dict(period=period, ecc=ecc, inc=inc,
                  omega=omega, lnode=lnode, tau=tau, cosi=cosi)

    return params


def p_to_a(p, mu):
    """Convert orbital period to semi-major axis.

    Args:
        p: Orbital period (days).
        mu: Gravitational parameter G * mass (solar units).

    Returns:
        float: Semi-major axis (AU).
    """
    n = 2 * jnp.pi / p
    a = (mu / n**2)**(1./3.)
    return a


def sample_radius_volume_uniform(prefix, rmax, rmin=0.):
    """Sample a radius with probability density proportional to r^2.

    Args:
        prefix: Prefix for NumPyro site names.
        rmax: Maximum radius.
        rmin: Minimum radius.

    Returns:
        float: Sampled radius.
    """
    u = numpyro.sample(f"{prefix}_u", dist.Uniform(0., 1.))
    r = (rmin**3 + u * (rmax**3 - rmin**3))**(1./3.)
    r = numpyro.deterministic(prefix, r)
    return r


def sample_unit_vector(prefix, flip_z_sign=False):
    """Sample a unit vector isotropically on the sphere.

    Args:
        prefix: Prefix for NumPyro site names.
        flip_z_sign: If True, flip the sign of the z component.

    Returns:
        array: Sampled unit vector with shape (3,).
    """
    d_raw = numpyro.sample(
        f"{prefix}_dir_raw",
        dist.Normal(0., 1.).expand((3,))
    )
    d_hat = d_raw / jnp.linalg.norm(d_raw)

    if flip_z_sign:
        d_hat = d_hat * jnp.array([1., 1., -1.])

    d_hat = numpyro.deterministic(prefix, d_hat)
    return d_hat


def sample_vector_volume_uniform(prefix, rmax, rmin=0.):
    """Sample a 3D vector uniformly in spherical volume.

    Args:
        prefix: Prefix for NumPyro site names.
        rmax: Outer radius.
        rmin: Inner radius.

    Returns:
        tuple: Sampled radius and 3D vector.
    """
    r = sample_radius_volume_uniform(f"{prefix}_r", rmax, rmin)
    r_hat = sample_unit_vector(f"{prefix}_hat")
    vec = numpyro.deterministic(prefix, r * r_hat)
    return r, vec


def sample_xv(t_ref, mass, pmin=1e2, pmax=1e6, eps=1e-3):
    """Sample Cartesian position and velocity, then convert to elements.

    Args:
        t_ref: Reference time.
        mass: Total mass.
        pmin: Minimum period prior bound (days).
        pmax: Maximum period prior bound (days).
        eps: Small offset to keep the sampled orbit bound.

    Returns:
        dict: Orbital parameters inferred from sampled x and v.
    """
    mu = G * mass
    amin, amax = p_to_a(pmin, mu), p_to_a(pmax, mu)
    rmin = amin  # actual amin becomes slightly smaller than this
    rmax = amax * (2. - eps)

    x_norm, x = sample_vector_volume_uniform("x", rmax=rmax, rmin=rmin)

    vmax_given_x = numpyro.deterministic(
        "vmax_given_x", jnp.sqrt(mu * (2. / x_norm - 1. / amax)))
    v_norm, v = sample_vector_volume_uniform("v", rmax=vmax_given_x)

    # correction to enforece p(x,v)=uniform;
    # above sampling leads to: p(x) = unif, p(v|x) \propto 1/vmax(x)**3, so p(x,v) \propto 1/vmax(x)**3/
    numpyro.factor("phase_space_volume_factor", 3 * jnp.log(vmax_given_x))

    params = xv_to_elements(x, v, mass, t_ref=t_ref)
    for key in params.keys():
        if key != 'mass':
            numpyro.deterministic(key, params[key])
    numpyro.deterministic("logperiod", jnp.log(params['period']))
    numpyro.deterministic("cosi", jnp.cos(params['inc']))

    return params


def sample_xv_via_period(t_ref, mass, pmin=1e2, pmax=1e6, eps=1e-3, log_period=True):
    """Sample Cartesian position and velocity via a sampled period.

    Args:
        t_ref: Reference time.
        mass: Total mass.
        pmin: Minimum period prior bound (days).
        pmax: Maximum period prior bound (days).
        eps: Small offset to keep the sampled orbit bound.
        log_period: If True, sample log(period) instead of period.

    Returns:
        dict: Orbital parameters inferred from sampled x and v.
    """
    if log_period:
        logperiod = numpyro.sample(
            "logperiod", dist.Uniform(jnp.log(pmin), jnp.log(pmax)))
        period = numpyro.deterministic("period", jnp.exp(logperiod))
    else:
        period = numpyro.sample("period", dist.Uniform(pmin, pmax))
        logperiod = numpyro.deterministic("logperiod", jnp.log(period))

    return sample_xv_given_P(t_ref, mass, period, eps=eps)


def sample_xv_given_P(
    t_ref,
    mass,
    period,
    eps=1e-3,
    rmin=None,
    rmin_frac=1e-3,
):
    """Sample Cartesian position and velocity at fixed period.

    Args:
        t_ref: Reference time.
        mass: Total mass.
        period: Orbital period (days).
        eps: Small offset to keep the sampled orbit bound.
        rmin: Minimum radius. If None, use rmin_frac * a.
        rmin_frac: Minimum radius as a fraction of semi-major axis when rmin is None.

    Returns:
        dict: Orbital parameters inferred from sampled x and v.
    """
    mu = G * mass
    a = p_to_a(period, mu)

    # r-range
    if rmin is None:
        rmin = rmin_frac * a
    rmax = (2. - eps) * a

    # position: uniform in volume
    x_norm, x = sample_vector_volume_uniform("x", rmax=rmax, rmin=rmin)

    # speed from vis-viva (E fixed by P)
    v_norm = numpyro.deterministic(
        "v_norm", jnp.sqrt(mu * (2. / x_norm - 1. / a)))

    # direction: isotropic
    v_hat = sample_unit_vector("v_hat")
    v = numpyro.deterministic("v", v_norm * v_hat)

    # correction to enforece p(x,v)=uniform
    numpyro.factor("phase_space_volume_factor", 3 *
                   jnp.log(rmax) + jnp.log(1 - (rmin/rmax)**3))
    numpyro.factor("jacobian", jnp.log(mu) + jnp.log(v_norm) -
                   jnp.log(a) - jnp.log(period) - jnp.log(3))

    # elements + bookkeeping
    params = xv_to_elements(x, v, mass, t_ref=t_ref)
    for key in params.keys():
        if key != "mass" and key != 'period':
            numpyro.deterministic(key, params[key])
    # numpyro.deterministic("logperiod", jnp.log(params["period"]))
    numpyro.deterministic("cosi", jnp.cos(params["inc"]))

    return params


def get_init_elements(p, t_ref):
    """Construct initial values in the element parameterization.

    Args:
        p: Dictionary of orbital parameters.
        t_ref: Reference time.

    Returns:
        dict: Initial values for NumPyro sampling.
    """
    year = 2 * jnp.pi / jnp.sqrt(G)
    period_day = p['period_yr'] * year
    tau = p['tau_mjd']
    Mref = 2 * jnp.pi * (t_ref - tau) / period_day
    inc = p['inc_deg'] * jnp.pi / 180.
    lnode = p['lnode_deg'] * jnp.pi / 180.
    omega = p['omega_deg'] * jnp.pi / 180.

    params = dict(mass=p['mass'],
                  parallax=p['parallax'],
                  logperiod=jnp.log(period_day),
                  period=period_day,
                  ecc=p['ecc'],
                  cosi=jnp.cos(inc),
                  cosL=jnp.cos(lnode),
                  sinL=jnp.sin(lnode),
                  cosw=jnp.cos(omega),
                  sinw=jnp.sin(omega),
                  cosm=jnp.cos(Mref),
                  sinm=jnp.sin(Mref),
                  lnode=lnode,
                  omega=omega,
                  tau=tau,
                  M=Mref,
                  )

    return params
