"""Check the rules of avoidance: walls, contact, arrival, the 10 s rule, sight, and the
Other-Play map."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from swarm.envs import avoidance, core
from swarm.learn import evaluate

PARAMS = avoidance.PRESETS["d_inf"]
KEY = jax.random.PRNGKey(0)
STILL = jnp.zeros((2, avoidance.NUM_ACTIONS))
GOALS = [[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]]
step = jax.jit(avoidance.step)


def actor_input(obs, drone):
    return obs.actor["drone"][drone]


def scene(x, params=PARAMS):
    """The 5 m tube along the world x axis, the drones at x in the tube frame, no map."""

    _, state = avoidance.reset(KEY, params)
    middle = avoidance.middle(params)
    return state.replace(
        drone=state.drone.replace(x=middle + jnp.array(x)),
        goal=middle + jnp.array(GOALS),
        heading=jnp.zeros(()),
        mirror=jnp.zeros(2, bool),
        turn=jnp.zeros(2),
    )


def once(x):
    """One step with the motors near hover, from the drones at x."""

    _, state, reward, done, info = step(KEY, scene(x), STILL, PARAMS)
    return state, reward, done, info


def test_a_drone_whose_ball_meets_a_wall_dies_and_ends_the_episode():
    # The side wall is 1.5 m from the middle, so a 0.4 m ball touches it from 1.1 m on.
    _, _, done, info = once([[-1.5, 1.0, 0.0], [1.5, 0.0, 0.0]])
    assert not bool(done) and bool(jnp.all(info["alive"])), "a drone 0.5 m from the wall died"

    _, reward, done, info = once([[-1.5, 1.2, 0.0], [1.5, 0.0, 0.0]])
    assert list(info["alive"]) == [False, True] and list(reward) == [-1.0, 0.0]
    assert bool(info["end"]["wall"])
    assert bool(done) and not bool(jnp.any(info["bootstrap"])), "a wall hit is terminal"


def test_contact_kills_both_drones():
    _, _, done, info = once([[-0.45, 0.0, 0.0], [0.45, 0.0, 0.0]])
    assert not bool(done) and bool(jnp.all(info["alive"])), "drones 0.9 m apart touched"

    _, reward, done, info = once([[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]])
    assert not bool(jnp.any(info["alive"])) and list(reward) == [-1.0, -1.0]
    assert bool(info["end"]["contact"])
    assert bool(done) and not bool(jnp.any(info["bootstrap"])), "a contact is terminal"


def test_an_arrived_drone_is_paid_once_freezes_and_cannot_be_touched():
    # Drone 0 is 0.5 m short of its goal, so it arrives in this step.
    state, reward, done, info = once([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert list(reward) == [1.0, 0.0] and list(info["flying"]) == [False, True]
    assert not bool(done), "the other drone still flies"

    frozen = state.drone.x[0]
    state = state.replace(
        drone=state.drone.replace(x=state.drone.x.at[1].set(frozen + jnp.array([0.0, 0.3, 0.0])))
    )
    full = jnp.ones((2, avoidance.NUM_ACTIONS))
    _, state, reward, _, info = step(KEY, state, full, PARAMS)
    assert jnp.array_equal(state.drone.x[0], frozen), "an arrived drone moved"
    assert float(reward[0]) == 0.0, "an arrived drone was paid again"
    assert bool(jnp.all(info["alive"])), "a contact counted an arrived drone"


def test_the_timeout_ends_the_episode():
    # Between steps 19 and 20, so a rounding error cannot move the end.
    params = PARAMS.replace(timeout=0.195)

    def body(state, _):
        _, state, _, done, info = avoidance.step(KEY, state, STILL, params)
        return state, (done, info)

    _, (done, info) = jax.lax.scan(body, scene(GOALS[::-1], params), None, length=25)
    end = int(jnp.argmax(done))
    assert end == 19, f"the episode ended at step {end + 1}, not at step 20"
    assert bool(info["end"]["timeout"][end]) and bool(jnp.all(info["alive"][end]))
    assert not bool(jnp.any(info["bootstrap"][end])), "the 10 s rule is terminal"


def test_a_drone_sees_another_only_in_range_and_in_flight():
    params = avoidance.PRESETS["d3"]

    def view(other, arrived=False):
        state = scene([[-1.5, 0.0, 0.0], other], params)
        state = state.replace(arrived=jnp.array([False, arrived]))
        return actor_input(avoidance.get_obs(state, params), 0)

    far = view([2.0, 0.0, 0.0])
    assert jnp.array_equal(far, view([2.0, 0.5, 0.3])), "a drone out of range left a trace"
    assert not jnp.array_equal(far, view([0.5, 0.0, 0.0])), "a drone 2 m away was not seen"
    assert jnp.array_equal(far, view([0.5, 0.0, 0.0], arrived=True)), "an arrived drone was seen"


def test_the_cascade_flies_a_mapped_drone_on_the_same_path():
    """The cascade reads the true world, so drone 1's map must not change any flight."""

    act, start = avoidance.PILOTS["cascade"](PARAMS, jnp.arange(2))

    def path(mirror, turn):
        mapped = lambda state: start(state.replace(mirror=jnp.array(mirror), turn=jnp.array(turn)))
        return evaluate.rollout(KEY, avoidance, PARAMS, act, mapped)[0].x

    plain = path([False, False], [0.0, 0.0])
    assert jnp.allclose(plain, path([False, True], [0.0, 2.3])), "the map changed the flight"


def test_the_other_play_map_is_a_symmetry():
    """Drone 1 sees and acts through the map F = R_z(turn) S. The same flight in the
    world seen through F, where nobody has a map, gives drone 1 the same input."""

    turn, S = 2.3, jnp.diag(jnp.array([1.0, -1.0, 1.0]))
    _, state = avoidance.reset(jax.random.PRNGKey(1), PARAMS)
    state = state.replace(mirror=jnp.array([False, True]), turn=jnp.array([0.0, turn]))

    # At the start the rates are 0 and every motor turns at hover, so only x, v and R move.
    F = core.yaw(turn) @ S
    d = state.drone
    image = state.replace(
        drone=d.replace(x=d.x @ F.T, v=d.v @ F.T, R=F @ d.R @ S),
        heading=turn - state.heading,
        goal=state.goal @ F.T,
        mirror=jnp.zeros(2, bool),
        turn=jnp.zeros(2),
    )

    actions = jax.random.uniform(KEY, (200, 2, 4), minval=-0.3, maxval=0.3)
    actions = actions.at[..., 0].add(-0.07)
    # Drone 0 has no map, so its image flies the mirrored roll and yaw.
    mirrored = actions.at[:, 0].multiply(jnp.array([1.0, -1.0, 1.0, -1.0]))

    def fly(state, actions):
        def body(state, action):
            obs, state, reward, done, _ = avoidance.step(KEY, state, action, PARAMS)
            return state, (actor_input(obs, 1), reward, done)

        return jax.lax.scan(body, state, actions)[1]

    first = lambda s: actor_input(avoidance.get_obs(s, PARAMS), 1)
    assert jnp.allclose(first(state), first(image), atol=1e-9)
    seen, reward, done = fly(state, actions)
    seen_image, reward_image, done_image = fly(image, mirrored)
    assert jnp.allclose(seen, seen_image, atol=1e-9), "drone 1 told the two worlds apart"
    assert jnp.array_equal(reward, reward_image) and jnp.array_equal(done, done_image)
