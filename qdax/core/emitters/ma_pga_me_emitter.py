from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.ma_standard_emitters import NaiveMultiAgentMixingEmitter, MultiAgentEmitter
from qdax.custom_types import Params, RNGKey
from qdax.environments.base_wrappers import QDEnv


@dataclass
class MultiAgentPGAMEConfig:
    """Configuration for PGAME Algorithm"""

    env_batch_size: int = 100
    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2

    # MultiAgent params
    variation_percentage: float = 1.0
    crossplay_percentage: float = 0.5
    num_agents: int = 2
    agents_to_mutate: int = -1
    role_preserving: bool = True
    parameter_sharing: bool = False
    emitter_type: str = "naive"

class MultiAgentPGAMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: MultiAgentPGAMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
        mutation_fn: Callable[[Params, RNGKey], Tuple[Params, RNGKey]]
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        qpg_batch_size = config.env_batch_size - ga_batch_size

        qpg_config = QualityPGConfig(
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
            id=0,
        )

        q_emitters = {}
        i = 0;

        if config.parameter_sharing:
            q_emitters["0"] = QualityPGEmitter(
                config=qpg_config, policy_network=policy_network, env=env
            ) 
        else:
            for key, policy in policy_network.items():
                q_emitters[key] = QualityPGEmitter(
                    config=qpg_config, policy_network=policy, env=env
                )
                i += 1
                qpg_config.id = i
                

        if config.parameter_sharing:
            # define the mixing emitter
            ga_emitter = MixingEmitter(
                mutation_fn=mutation_fn,
                variation_fn=variation_fn,
                variation_percentage=config.variation_percentage,
                batch_size=ga_batch_size,
            )
        else:
            if config.emitter_type == "naive":
                ga_emitter = NaiveMultiAgentMixingEmitter(
                    mutation_fn=mutation_fn,
                    variation_fn=variation_fn,
                    variation_percentage=config.variation_percentage,
                    batch_size=ga_batch_size,
                    num_agents=config.num_agents,
                    agents_to_mutate=config.agents_to_mutate,
                )
            else:
                ga_emitter = MultiAgentEmitter(
                    mutation_fn=mutation_fn,
                    variation_fn=variation_fn,
                    variation_percentage=config.variation_percentage,
                    crossplay_percentage=config.crossplay_percentage,
                    batch_size=ga_batch_size,
                    num_agents=config.num_agents,
                    role_preserving=config.role_preserving,
                    agents_to_mutate=config.agents_to_mutate,
                )

        super().__init__(emitters=(tuple(q_emitters.values()) + (ga_emitter,)))
