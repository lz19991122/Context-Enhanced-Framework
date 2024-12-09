import torch
import torch.nn as nn
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim=1024, n=114, e=256):
        super(FeatureTransformer, self).__init__()
        self.n = n  # Number of labels (n)
        self.e = e  # Embedding dimension (e)
        # Linear transformation to map 1024-dimensional features to (n * e)
        self.img_features = nn.Linear(input_dim, n * e)
    def forward(self, img_features):
        # Transform and reshape the features
        transformed_features = self.img_features(img_features)  # (B, 1024) → (B, n * e)
        reshaped_features = transformed_features.view(img_features.shape[0], self.n, self.e)  # (B, n * e) → (B, n, e)
        return reshaped_features

# Demo example
if __name__ == "__main__":
    # Example: Batch of 16 images with 1024-dimensional extracted features
    batch_size = 16
    input_dim = 1024
    n = 114
    e = 256
    # Simulate extracted image features
    img_features = torch.rand(batch_size, input_dim)  # (B, 1024)
    # Initialize the transformer
    feature_transformer = FeatureTransformer(input_dim=input_dim, n=n, e=e)
    # Transform the features
    transformed_features = feature_transformer(img_features)  # (B, n, e)
    # Print the output shape to verify
    print(f"Input shape: {img_features.shape}")  # (4, 1024)
    print(f"Output shape: {transformed_features.shape}")  # (4, 114, 256)
