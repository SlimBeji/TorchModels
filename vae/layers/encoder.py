import torch


class VAEEncoder(torch.nn.Sequential):
    def __init__(
        self,
        image_shape: tuple[int, int, int],  # CWH
        latent_space_dim: int,
        conv_layers: list[int],
        perceptron_layers: list[int],
    ):
        # Initialization
        self.image_shape = image_shape
        self.latent_space_dim = latent_space_dim
        self.conv_layers = conv_layers
        self.perceptron_layers = perceptron_layers
        self._layers = []

        # Build
        self._build_conv_layers()
        self._build_reshaping_layers()
        self._build_perceptron_layers()
        super().__init__(*self._layers)

    @property
    def color_channels(self) -> int:
        return self.image_shape[0]

    @property
    def input_width(self) -> int:
        return self.image_shape[1]

    @property
    def input_height(self) -> int:
        return self.image_shape[2]

    @property
    def conv_size_reduction(self) -> int:
        return 2 ** len(self.conv_layers)

    @property
    def width_after_conv(self) -> int:
        return self.input_width // self.conv_size_reduction

    @property
    def height_after_conv(self) -> int:
        return self.input_height // self.conv_size_reduction

    @property
    def perceptron_input_size(self) -> int:
        return self.conv_layers[-1][-1]

    def _build_conv_layers(self):
        error_message = "For {} convolution layers, size reduction is {}. The input shapes {}x{} must be multiple of {}"
        if (self.width_after_conv == 0) or (self.height_after_conv == 0):
            raise ValueError(
                error_message.format(
                    len(self.conv_layers),
                    self.conv_size_reduction,
                    self.input_width,
                    self.input_height,
                    self.conv_size_reduction,
                )
            )

        last_out = self.color_channels
        error_message = "Last layer output features: {} does not match with next layer input features {}"
        for in_features, out_features in self.conv_layers:
            if last_out != in_features:
                raise ValueError(error_message.format(last_out, in_features))
            last_out = out_features

            # Append the layers
            self._layers.append(
                torch.nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
            )
            self._layers.append(torch.nn.BatchNorm2d(out_features))
            self._layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

    def _build_reshaping_layers(self):
        self._layers.append(
            torch.nn.AvgPool2d(
                kernel_size=(self.width_after_conv, self.height_after_conv)
            )
        )
        self._layers.append(torch.nn.Flatten())

    def _build_perceptron_layers(self):
        last_out = self.perceptron_input_size
        error_message = "Last layer output features: {} does not match with next layer input features {}"
        for in_features, out_features in self.perceptron_layers:
            if last_out != in_features:
                raise ValueError(error_message.format(last_out, in_features))
            last_out = out_features

            # Append the layers
            self._layers.append(torch.nn.Linear(in_features, out_features, bias=True))
            self._layers.append(torch.nn.ReLU())

        # Projection to Latent Space
        self._layers.append(torch.nn.Linear(last_out, 2 * self.latent_space_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self(x)
        mean = y[:, : self.latent_space_dim]
        logvar = y[:, self.latent_space_dim :]
        return mean, logvar
