import torch

from vae.layers.reshape import Reshape
from vae.layers.sample import Sample


class VAEDecoder(torch.nn.Sequential):
    def __init__(
        self,
        output_shape: tuple[int, int, int],  # CWH
        latent_space_dim: int,
        conv_layers: list[int],
        perceptron_layers: list[int],
    ):
        # Initialization
        self.image_shape = output_shape
        self.latent_space_dim = latent_space_dim
        self.conv_layers = conv_layers
        self.perceptron_layers = perceptron_layers
        self._layers = []

        # Build
        self._build_sampler()
        self._build_perceptron_layers()
        self._build_reshaping_layers()
        self._build_conv_layers()
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
    def perceptron_output_size(self) -> int:
        return self.perceptron_layers[-1][-1]

    @property
    def conv_size_augmentation(self) -> int:
        return 2 ** len(self.conv_layers)

    @property
    def width_before_conv(self) -> int:
        return self.input_width // self.conv_size_augmentation

    @property
    def height_before_conv(self) -> int:
        return self.input_height // self.conv_size_augmentation

    def _build_sampler(self):
        self._layers.append(Sample(self.latent_space_dim))

    def _build_perceptron_layers(self):
        last_out = self.latent_space_dim
        error_message = "Last layer output features: {} does not match with next layer input features {}"
        for in_features, out_features in self.perceptron_layers:
            if last_out != in_features:
                raise ValueError(error_message.format(last_out, in_features))
            last_out = out_features

            # Append the layers
            self._layers.append(torch.nn.Linear(in_features, out_features, bias=True))
            self._layers.append(torch.nn.ReLU())

    def _build_reshaping_layers(self):
        # Upsampling with linear projection
        self._layers.append(
            torch.nn.Linear(
                self.perceptron_output_size,
                self.perceptron_output_size
                * self.width_before_conv
                * self.height_before_conv,
            )
        )
        # Reshaping
        self._layers.append(
            Reshape(
                1,
                (
                    self.perceptron_output_size,
                    self.width_before_conv,
                    self.height_before_conv,
                ),
            )
        )

    def _build_conv_layers(self):
        error_message = "For {} convolution layers, size augmentation is {}. The input shapes {}x{} must be multiple of {}"
        if (self.width_before_conv == 0) or (self.height_before_conv == 0):
            raise ValueError(
                error_message.format(
                    len(self.conv_layers),
                    self.conv_size_augmentation,
                    self.input_width,
                    self.input_height,
                    self.conv_size_augmentation,
                )
            )

        last_out = self.perceptron_output_size
        error_message = "Last layer output features: {} does not match with next layer input features {}"
        for in_features, out_features in self.conv_layers:
            if last_out != in_features:
                raise ValueError(error_message.format(last_out, in_features))
            last_out = out_features

            # Append the layers
            self._layers.append(torch.nn.Upsample(scale_factor=2, mode="nearest"))
            self._layers.append(
                torch.nn.ConvTranspose2d(
                    in_features, out_features, kernel_size=3, padding=1
                )
            )
            self._layers.append(torch.nn.BatchNorm2d(out_features))

    def decode(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if mean.ndim != 2 or logvar.ndim != 2:
            raise ValueError("Mean and Logvar batch tensors must be 2D")
        if mean.shape != logvar.shape:
            raise ValueError("Inconsistencies in Mean and Logvar shapes")
        return self(torch.cat((mean, logvar), dim=1))
