import torch 
import torch.nn as nn
from torch import distributions as dist
import numpy as np
from torch.nn import functional as F

class Bernoulli(nn.Module):
    """Bernoulli layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, in_features: int, out_channels: int):
        """Creat a gaussian layer.
        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.
        """
        super().__init__()
        # Create bernoulli parameters
        self.probs = nn.Parameter(torch.randn(in_features, out_channels))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        probs_ratio = torch.sigmoid(self.probs)
        return dist.Bernoulli(probs=probs_ratio)

    def forward(self, x):
        d = self._get_base_distribution()
        return d.log_prob(x)

class Sum(nn.Module):
    def __init__(
        self, in_channels: int, in_features: int, out_channels: int):
        """
        Create a Sum layer.
        Input is expected to be of shape [n, d, ic, r].
        Output will be of shape [n, d, oc, r].
        Args:
            in_channels (int): Number of output channels from the previous layer.
            in_features (int): Number of input features.
            out_channels (int): Multiplicity of a sum node for a given scope set.
            num_repetitions(int): Number of layer repetitions in parallel.
            dropout (float, optional): Dropout percentage.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        # self.dropout = nn.Parameter(torch.tensor(check_valid(dropout, float, 0.0, 1.0)), requires_grad=False)

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(self.in_features, self.in_channels, self.out_channels)
        self.weights = nn.Parameter(ws)
        # self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        self.out_shape = f"(N, {self.in_features}, {self.out_channels})"

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache = None

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache = None

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.weights.device

    def forward(self, x: torch.Tensor):
        """
        Sum layer foward pass.
        Args:
            x: Input of shape [batch, in_features, in_channels].
        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.clone()

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        # if self.dropout > 0.0 and self.training:
        #     dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
        #     x[dropout_indices] = np.NINF

        # Dimensions
        n, d, ic = x.size()
        oc = self.weights.size(2)

        x = x.unsqueeze(3)  # Shape: [n, d, ic, 1, r]

        # Normalize weights in log-space along in_channel dimension
        # Weights is of shape [d, ic, oc, r]
        logweights = F.log_softmax(self.weights, dim=1)
        # print("logweights", logweights.shape)
        # print("x", x.shape)

        # Multiply (add in log-space) input features and weights
        x = x + logweights  # Shape: [n, d, ic, oc, r]

        # Compute sum via logsumexp along in_channels dimension
        x = torch.logsumexp(x, dim=2)  # Shape: [n, d, oc, r]

        # Assert correct dimensions
        assert x.size() == (n, d, oc)

        return x

class CrossProduct(nn.Module):
    """
    Layerwise implementation of a RAT Product node.
    Builds the the combination of all children in two regions:
    res = []
    for n1 in R1, n2 in R2:
        res += [n1 * n2]
    TODO: Generalize to k regions (cardinality = k).
    """

    def __init__(self, in_features: int, in_channels: int):
        """
        Create a rat product node layer.
        Args:
            in_features (int): Number of input features.
            in_channels (int): Number of input channels. This is only needed for the sampling pass.
        """

        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = 2
        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self._pad = 0

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(self.cardinality)]

        # Create sequence of scopes
        scopes = np.arange(self.in_features)

        # For two consecutive scopes
        for i in range(0, self.in_features, self.cardinality):
            for j in range(cardinality):
                if i + j < in_features:
                    self._scopes[j].append(scopes[i + j])
                else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    self._scopes[j].append(self.in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            torch.tensor([(i, j) for i in range(self.in_channels) for j in range(self.in_channels)]),
            requires_grad=False,
        )

        self.out_shape = f"(N, {self._out_features}, {self.in_channels ** 2})"

    def forward(self, x: torch.Tensor):
        """
        Product layer forward pass.
        Args:
            x: Input of shape [batch, in_features, channel].
        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """
        # Check if padding to next power of 2 is necessary
        if self.in_features != x.shape[1]:
            # Compute necessary padding to the next power of 2
            self._pad = 2 ** np.ceil(np.log2(x.shape[1])).astype(np.int) - x.shape[1]

            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

        # Dimensions
        # print(x.size())
        # print("hi")
        n, d, c = x.size()
        d_out = d // self.cardinality

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor dimensions:
        # left: [n, d/2, c, r] -> [n, d/2, c, 1, r]
        # right: [n, d/2, c, r] -> [n, d/2, 1, c, r]
        left = x[:, self._scopes[0, :], :].unsqueeze(3)
        right = x[:, self._scopes[1, :], :].unsqueeze(2)

        # left + right with broadcasting: [n, d/2, c, 1, r] + [n, d/2, 1, c, r] -> [n, d/2, c, c, r]
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c, r] -> [n, d/2, c * c, r]
        result = result.view(n, d_out, c * c)

        assert result.size() == (n, d_out, c * c)
        return result

# def decode_spn(action_indices):
    

# x1 = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
# x2 = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])

# x1 = torch.unsqueeze(x1, dim = 1)
# x2 = torch.unsqueeze(x2, dim = 1)
# print(x1.shape)
# b1 = Bernoulli(in_features = 1, out_channels = 2)
# b2 = Bernoulli(in_features = 1, out_channels = 2)
# out1 = b1.forward(x1)
# out2 = b2.forward(x2)
# out1 = torch.unsqueeze(out1, axis = 1)
# out2 = torch.unsqueeze(out2, axis = 1)
# print(out1.shape)
# print(out2.shape)
# out = torch.cat([out1, out2], axis = 1)
# print(out.shape)
# cp = CrossProduct(in_features = 2, in_channels = 2)
# o = cp(out)
# print(o.shape)
# s = Sum(in_channels = 4, in_features = 1, out_channels = 2)
# final = s(o)
# print(final.shape)
# out = b.forward(x)
# print(out.shape)

