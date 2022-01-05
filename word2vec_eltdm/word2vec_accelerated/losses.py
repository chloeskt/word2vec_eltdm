import torch
from torch import nn


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(
        input_vector: torch.Tensor,
        output_vector: torch.Tensor,
        noise_vectors: torch.Tensor,
    ) -> float:
        batch_size, embed_size = input_vector.shape

        input_vector = input_vector.view(batch_size, embed_size, 1)

        output_vector = output_vector.view(batch_size, 1, embed_size)

        out_loss = torch.bmm(output_vector, input_vector).sigmoid().log()
        out_loss = out_loss.squeeze()

        noise_loss = torch.bmm(noise_vectors.neg(), input_vector).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)

        return -(out_loss + noise_loss).mean()
