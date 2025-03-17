import torch
from TFModel import TFModel


class ConvTRNet(torch.nn.Module):
    def __init__(self, config) -> None:
        super(ConvTRNet, self).__init__()

        # load configuration
        self.config = config
        self.num_users = config["num_users"]
        self.num_services = config["num_services"]
        self.num_timeslices = config["num_timeslices"]
        self.latent_dim = config["latent_dim"]

        # Embedding layers
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users,
                                                 embedding_dim=self.latent_dim * self.latent_dim)
        self.embedding_service = torch.nn.Embedding(num_embeddings=self.num_services,
                                                    embedding_dim=self.latent_dim * self.latent_dim)
        self.embedding_timeslice = torch.nn.Embedding(num_embeddings=self.num_timeslices,
                                                      embedding_dim=self.latent_dim * self.latent_dim)

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, (10, 10), stride=10),
            torch.nn.BatchNorm2d(20, eps=1e-08),
            torch.nn.Dropout2d(0.1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (1, 3)),
            torch.nn.Dropout2d(0.1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, (1, 3)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )
        self.init_weight()

    def forward(self, user_indices, service_indices, timeslice_indices):
        # Forward pass
        user_embedding = self.embedding_user(user_indices)
        service_embedding = self.embedding_service(service_indices)
        timeslice_embedding = self.embedding_timeslice(timeslice_indices)
        user_slice = user_embedding.reshape(user_indices.shape[0], 20, 20)
        service_slice = service_embedding.reshape(service_indices.shape[0], 20, 20)
        timeslice_slice = timeslice_embedding.reshape(timeslice_indices.shape[0], 20, 20)
        rating = torch.cat((user_slice, service_slice, timeslice_slice), dim=2)
        rating = rating.unsqueeze(1)
        rating = self.net(rating)
        return rating

    def init_weight(self):
        """Initialize weights with specific ranges or distributions"""
        # Embedding layers
        torch.nn.init.uniform_(self.embedding_user.weight, 0.0, 0.004)
        torch.nn.init.uniform_(self.embedding_service.weight, 0.0, 0.004)
        torch.nn.init.uniform_(self.embedding_timeslice.weight, 0.0, 0.004)

class ConvTR(TFModel):
    """Engine for training & evaluating MLP model"""

    def __init__(self, config):

        self.model = ConvTRNet(config)

        # 检查是否要使用CUDA，并且CUDA是否可用
        if config.get("use_cuda", False) and torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            self.model.to(cuda_device)
        else:
            # 如果没有可用GPU，使用CPU
            self.model.to(torch.device("cpu"))
        super(ConvTR, self).__init__(config)
