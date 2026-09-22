"""Check cascade goal tracking, episode endings, and batched env behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import pytest

from swarm.envs import a_to_b, chase, core, flat
from swarm.sim import control

PRESETS = ("hover", "default")


def cascade_policy(params):
    gains = control.cascade_gains(params.model)
    pids = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (params.n_drones,) + x.shape),
        control.cascade_init(),
    )

    def act(state, pids):
        def one(drone, goal, pid):
            (throttle, rate_ref), pid = control.cascade_outer(
                drone, goal, 0.0, pid, gains, params.model, params.policy_dt
            )
            return core.command_to_action(throttle, rate_ref, params), pid

        return jax.vmap(one)(state.drone, state.goal, pids)

    return act, pids


def replay(state, actions):
    return actions[state.time], actions


def rollout(params, key, act, carry):
    _, state = a_to_b.reset(key, params)

    def body(both, _):
        state, carry = both
        action, carry = act(state, carry)
        _, state, reward, done, info = a_to_b.step(key, state, action, params)
        return (state, carry), (reward, done, info["end"]["distance"], info["alive"])

    return jax.lax.scan(body, (state, carry), None, length=params.max_steps)


@pytest.mark.parametrize("preset", PRESETS)
def test_cascade_arrives_and_stays(preset):
    params = a_to_b.PRESETS[preset]
    act, pids = cascade_policy(params)
    keys = jax.random.split(jax.random.PRNGKey(0), 32)

    _, (_, _, distance, _) = jax.vmap(rollout, in_axes=(None, 0, None, None))(
        params, keys, act, pids
    )

    settled = distance[:, -int(1.0 / params.policy_dt) :]
    assert jnp.all(settled < 0.2), f"worst settled distance {settled.max():.3f} m"


def test_shapes_and_dtypes_survive_a_step():
    params = a_to_b.PRESETS["default"]
    n, k = params.n_drones, params.n_visible
    obs, state = a_to_b.reset(jax.random.PRNGKey(0), params)

    assert obs.own.shape == (n, 20)
    assert obs.others.shape == (n, k, core.OTHER_FEATURES)
    assert obs.target.shape == (n, 6)
    assert flat(obs).shape == (n, 20 + k * core.OTHER_FEATURES + k + 6)

    action = jnp.zeros((n, a_to_b.NUM_ACTIONS))
    obs2, state2, reward, done, info = a_to_b.step(jax.random.PRNGKey(1), state, action, params)

    assert reward.shape == (n,) and done.shape == ()
    assert info["alive"].shape == (n,) and info["truncated"].shape == ()
    assert jax.tree.structure(obs) == jax.tree.structure(obs2)

    for field, before in jax.tree_util.tree_leaves_with_path(state):
        after = dict(jax.tree_util.tree_leaves_with_path(state2))[field]
        assert before.dtype == after.dtype, f"dtype drift in {field}"


def test_random_actions_stay_finite_and_terminate():
    params = a_to_b.PRESETS["hover"]
    key = jax.random.PRNGKey(0)
    actions = jax.random.uniform(
        key, (params.max_steps, params.n_drones, a_to_b.NUM_ACTIONS), minval=-1, maxval=1
    )
    _, (reward, done, distance, _) = rollout(params, key, replay, actions)

    assert jnp.all(jnp.isfinite(reward)) and jnp.all(jnp.isfinite(distance))
    assert jnp.any(done), "random actions never ended an episode"


def test_dead_drone_stays_dead():
    params = a_to_b.PRESETS["hover"]
    key = jax.random.PRNGKey(0)
    actions = jnp.tile(
        jnp.array([-1.0, 0.0, 0.0, 0.0]), (params.max_steps, params.n_drones, 1)
    )
    _, (reward, _, _, alive) = rollout(params, key, replay, actions)

    death = int(jnp.argmin(alive[:, 0]))
    assert 0 < death < params.max_steps, "motors off and the drone never died"
    assert not jnp.any(alive[death:]), "a dead drone came back"
    assert reward[death] <= -params.reward.crash, "no crash penalty on the fatal step"
    assert jnp.allclose(reward[death + 1 :], 0.0), "a dead drone kept scoring"


def test_timeout_truncates_without_terminating():
    params = a_to_b.PRESETS["hover"]
    _, state = a_to_b.reset(jax.random.PRNGKey(0), params)
    state = state.replace(time=jnp.int32(params.max_steps - 1))
    action = jnp.zeros((params.n_drones, a_to_b.NUM_ACTIONS))

    _, _, _, done, info = a_to_b.step(jax.random.PRNGKey(1), state, action, params)
    assert bool(done) and bool(info["truncated"]) and bool(jnp.all(info["alive"]))


@pytest.mark.parametrize("pursuers", (1, 3))
def test_chase_reads_its_drone_count_from_roles(pursuers):
    params = chase.EnvParams(roles=("pursuer",) * pursuers + ("evader",))
    n, k = params.n_drones, params.n_visible
    obs, state = chase.reset(jax.random.PRNGKey(0), params)

    action = jnp.zeros((n, chase.NUM_ACTIONS))
    _, _, reward, done, info = chase.step(jax.random.PRNGKey(1), state, action, params)

    # chase flies at a drone, never at a fixed point, so it has no target block.
    assert obs.target.shape == (n, 0)
    assert flat(obs).shape == (n, 20 + k * core.OTHER_FEATURES + k)
    assert reward.shape == (n,) and done.shape == () and info["end"]["caught"].shape == ()
    assert chase.reference(state, params).shape == (n, 3)
    assert chase.is_caught(state.drone, params).shape == (params.roles.count("pursuer"),)

    # Every other drone has a seat, and the flag says which are the same role.
    role = jnp.array([params.roles[i] == "pursuer" for i in range(n)])
    assert jnp.all(obs.others_mask == 1.0), "a drone in range lost its seat"
    for i in range(n):
        same = obs.others[i, :, -1]
        assert int(same.sum()) == int(jnp.sum(role == role[i])) - 1


def test_one_catch_pays_every_pursuer():
    params = chase.EnvParams(roles=("pursuer",) * 3 + ("evader",))
    _, state = chase.reset(jax.random.PRNGKey(0), params)
    pursuers, evader = chase._sides(params)

    # The evader touches the net of pursuer 1, 0.9 m under it. The others are clear.
    x = jnp.array([[-6.0, 0.0, 8.0], [0.0, 0.0, 8.0], [6.0, 0.0, 8.0], [0.2, 0.0, 7.1]])
    drone = state.drone.replace(x=x, R=jnp.stack([jnp.eye(3)] * params.n_drones))

    caught = chase.is_caught(drone, params)
    assert list(caught) == [False, True, False]

    reward = chase.compute_reward(
        state.replace(drone=drone),
        jnp.zeros((params.n_drones, 3)),
        jnp.ones(params.n_drones, bool),
        jnp.zeros(params.n_drones, bool),
        jnp.zeros(params.n_drones, bool),
        jnp.any(caught),
        params,
    )
    assert jnp.all(reward[pursuers] > 0.9 * params.reward.catch), "a pursuer went unpaid"
    assert reward[evader] < -0.9 * params.reward.catch, "the evader was not billed"


def test_a_net_on_another_pursuer_kills_both():
    params = chase.EnvParams(roles=("pursuer",) * 3 + ("evader",))
    _, state = chase.reset(jax.random.PRNGKey(0), params)

    # Pursuer 0 touches pursuer 1's net 0.9 m under it, 1.03 m from its centre.
    # That is outside the 0.8 m kill ball, so only the net can end this.
    x = jnp.array([[0.3, 0.4, 7.1], [0.0, 0.0, 8.0], [6.0, 0.0, 8.0], [-6.0, 0.0, 8.0]])
    drone = state.drone.replace(x=x, R=jnp.stack([jnp.eye(3)] * params.n_drones))

    assert not jnp.any(core.hits(drone, params)), "the bodies must not touch here"
    assert not jnp.any(chase.is_caught(drone, params))
    # Pursuer 2 is clear, and no net catches the drone it hangs under.
    assert list(chase.net_collision(drone, params)) == [True, True, False, False]
    assert list(chase.collPP(drone, params)) == [True, True, False, False]
    assert not jnp.any(chase.crashed(drone, params)), "nobody hit a wall"


def test_scripted_decides_who_flies_the_evader():
    params = chase.PRESETS["default"]
    _, evader = chase._sides(params)

    def evader_motors(env_params, throttle):
        """One step is 10 ms, so the motors move long before the position does."""

        _, state = chase.reset(jax.random.PRNGKey(0), env_params)
        action = jnp.zeros((env_params.n_drones, chase.NUM_ACTIONS))
        action = action.at[evader, 0].set(throttle)
        _, out, _, _, _ = chase.step(jax.random.PRNGKey(1), state, action, env_params)
        return out.drone.rpm[evader]

    scripted = params.replace(scripted=("evader",))
    assert jnp.array_equal(evader_motors(scripted, -1.0), evader_motors(scripted, 1.0)), (
        "the cascade should fly a scripted evader, whatever the network asked for"
    )

    free = params.replace(scripted=())
    idle, full = evader_motors(free, -1.0), evader_motors(free, 1.0)
    assert jnp.all(full > 1.05 * idle), (
        "with nothing scripted the network's action must reach the motors"
    )


def test_vmap_matches_python_loop():
    params = a_to_b.PRESETS["default"]
    act, pids = cascade_policy(params)
    keys = jax.random.split(jax.random.PRNGKey(3), 4)

    batched = jax.vmap(rollout, in_axes=(None, 0, None, None))(params, keys, act, pids)[1][2]
    looped = jnp.stack([rollout(params, k, act, pids)[1][2] for k in keys])
    assert jnp.allclose(batched, looped)
