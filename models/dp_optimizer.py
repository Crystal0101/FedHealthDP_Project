from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer # type: ignore
from tensorflow_privacy.privacy.dp_query.gaussian_query import GaussianSumQuery # type: ignore

def create_dp_optimizer(l2_norm_clip, noise_multiplier, num_microbatches, learning_rate):
    dp_query = GaussianSumQuery(l2_norm_clip, l2_norm_clip * noise_multiplier)
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate,
        dp_average_query=dp_query)
    return optimizer
