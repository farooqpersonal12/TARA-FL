import os
import tempfile
import pytest
import torch
import torch.nn as nn

from model.base_model import BaseFLModel
from model.model import (
    MNISTModel,
    FashionMNISTModel,
    CIFAR10ConvNet,
    ResNet18Small,
    MLPModel,
    ModelFactory,
    get_model,
    list_available_models
)


class TestModelArchitectures:
    """Test suite for all model architectures and BaseFLModel features."""

    def test_mnist_model_forward(self):
        model = MNISTModel(num_classes=10, in_channels=1)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)
        assert model.count_parameters() > 0

    def test_fashion_mnist_model_forward(self):
        model = FashionMNISTModel(num_classes=10, in_channels=1)
        x = torch.randn(4, 1, 28, 28)
        model.eval()
        out = model(x)
        assert out.shape == (4, 10)

    def test_cifar10_convnet_forward(self):
        model = CIFAR10ConvNet(num_classes=10, in_channels=3)
        x = torch.randn(4, 3, 32, 32)
        model.eval()
        out = model(x)
        assert out.shape == (4, 10)

    def test_resnet18_small_forward(self):
        model = ResNet18Small(num_classes=10, in_channels=3, base_planes=16)
        x = torch.randn(4, 3, 32, 32)
        model.eval()
        out = model(x)
        assert out.shape == (4, 10)

    def test_mlp_model_forward_2d_and_4d(self):
        model = MLPModel(input_dim=784, hidden_dims=[128, 64], num_classes=10)
        model.eval()
        # 2D flattened input
        x2d = torch.randn(8, 784)
        out2d = model(x2d)
        assert out2d.shape == (8, 10)

        # 4D image input
        x4d = torch.randn(8, 1, 28, 28)
        out4d = model(x4d)
        assert out4d.shape == (8, 10)

    def test_parameter_vector_roundtrip(self):
        model = MNISTModel()
        param_vec = model.get_parameter_vector()
        assert param_vec.dim() == 1
        assert param_vec.numel() == model.count_parameters()

        # Modify vector and load back
        modified_vec = param_vec + 1.5
        model.set_parameter_vector(modified_vec)
        recovered_vec = model.get_parameter_vector()
        assert torch.allclose(modified_vec, recovered_vec, atol=1e-6)

    def test_gradient_vector_extraction(self):
        model = MNISTModel()
        x = torch.randn(2, 1, 28, 28)
        target = torch.tensor([1, 2])
        loss = nn.CrossEntropyLoss()(model(x), target)
        loss.backward()

        grad_vec = model.get_gradient_vector()
        assert grad_vec.dim() == 1
        assert grad_vec.numel() == model.count_parameters()
        assert not torch.all(grad_vec == 0)

    def test_weights_get_and_set(self):
        model1 = MNISTModel()
        model2 = MNISTModel()

        # Set model2 with random parameters
        with torch.no_grad():
            for p in model2.parameters():
                p.add_(torch.randn_like(p))

        weights2 = model2.get_weights()
        model1.set_weights(weights2)

        for k in weights2:
            assert torch.equal(model1.state_dict()[k], model2.state_dict()[k])

    def test_checkpoint_save_and_load(self):
        model = CIFAR10ConvNet(num_classes=10, in_channels=3)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            meta = {"round": 5, "accuracy": 0.89}
            model.save_checkpoint(temp_path, extra_meta=meta)

            new_model = CIFAR10ConvNet(num_classes=10, in_channels=3)
            loaded_meta = new_model.load_checkpoint(temp_path)

            assert loaded_meta == meta
            for k in model.state_dict():
                assert torch.equal(model.state_dict()[k], new_model.state_dict()[k])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestModelFactory:
    """Test suite for ModelFactory and get_model."""

    def test_list_available_models(self):
        models = list_available_models()
        assert "mnist" in models
        assert "cifar10" in models
        assert "resnet18" in models
        assert "mlp" in models

    def test_get_model_defaults(self):
        mnist = get_model("mnist")
        assert isinstance(mnist, MNISTModel)

        cifar = get_model("cifar10")
        assert isinstance(cifar, CIFAR10ConvNet)

        resnet = get_model("resnet18")
        assert isinstance(resnet, ResNet18Small)

        mlp = get_model("mlp")
        assert isinstance(mlp, MLPModel)

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            get_model("non_existent_model_architecture")
        assert "Unknown model name" in str(excinfo.value)
