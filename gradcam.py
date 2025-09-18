import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: trained CNN
        target_layer: string, name of the convolutional layer to visualize
        """
        self.model = model
        self.model.eval()
        self.target_layer = dict([*self.model.named_modules()])[target_layer]
        self.gradients = None
        # Hook the gradients
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)
        self.activations = None

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, x, class_idx=None):
        """
        x: input tensor (1,1,H,W)
        class_idx: which class to visualize. If None, uses top predicted class
        """
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()

        # Global average pooling of gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        # Weight the channels by corresponding gradients
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap
