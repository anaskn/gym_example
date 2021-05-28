from gym.envs.registration import register

register(
    id="example-v0",
    entry_point="gym_example.envs:Example_v0",
)


register(
    id="caching-v0",
    entry_point="gym_example.envs:Caching_v0",
)

register(
    id="caching-v020",
    entry_point="gym_example.envs:Caching_v020",
)

register(
    id="fail-v1",
    entry_point="gym_example.envs:Fail_v1",
)
