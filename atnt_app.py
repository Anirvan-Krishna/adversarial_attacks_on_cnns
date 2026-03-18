import os
import cv2
import torch
import numpy as np
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset

# --- Page Config ---
st.set_page_config(page_title="CNN Feature Visualizer", layout="wide")
st.title("🧠 What does the CNN see?")
st.markdown(
    "Step through the network to see how filters extract features from faces.")

# --- Setup & Constants ---
TRAIN_DIR = "Face-Recognition/Datasets/att_faces/Training"
TEST_DIR = "Face-Recognition/Datasets/att_faces/Testing"

# Removed caching to stop Streamlit from remembering old, broken states


def get_subjects_and_maps():
    if not os.path.exists(TRAIN_DIR):
        return [], {}, {}

    subjects = sorted([d for d in os.listdir(TRAIN_DIR)
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    label_map = {name: i for i, name in enumerate(subjects)}
    idx_to_class = {i: name for name, i in label_map.items()}
    return subjects, label_map, idx_to_class


subjects, label_map, idx_to_class = get_subjects_and_maps()

subjects = sorted([d for d in os.listdir(TRAIN_DIR)
                  if os.path.isdir(os.path.join(TRAIN_DIR, d))])
label_map = {name: i for i, name in enumerate(subjects)}

# AT&T Statistics
mean = 0.4416
std = 0.1955

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])


class FaceDataset(Dataset):
    def __init__(self, root_dir, label_map=label_map, transform=transform):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        for subject_dir in os.listdir(root_dir):
            if subject_dir not in label_map:
                continue  # Skip unknown folders

            face_dir = os.path.join(root_dir, subject_dir)
            for img_name in os.listdir(face_dir):
                self.data.append(os.path.join(face_dir, img_name))
                self.labels.append(label_map[subject_dir])  # Use the map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float32)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Model Architecture ---


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes=40):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Removed @st.cache_resource completely to force a fresh, accurate load


def load_data_and_model():
    dataset = FaceDataset(TEST_DIR)
    model = FaceRecognitionModel(num_classes=max(1, len(subjects)))
    weight_path = "face_model.pt"

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(
            weight_path, map_location=torch.device('cpu')))
    else:
        st.sidebar.error(f"🚨 Model weights not found at `{weight_path}`!")

    model.eval()
    return dataset, model


if not subjects:
    st.error(
        f"Could not find training directory at `{TRAIN_DIR}`. Please check your paths.")
    st.stop()

dataset, model = load_data_and_model()

# --- Helper Functions for Visualization ---


def plot_activations(tensor, title):
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    num_channels = tensor.shape[0]

    cols = 8
    rows = (num_channels // cols) + (1 if num_channels % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(title, fontsize=16)

    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        if i < num_channels:
            ax.imshow(tensor[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    return fig


# --- State Management ---
if 'img_idx' not in st.session_state:
    st.session_state.img_idx = 0


def next_image():
    st.session_state.img_idx = (st.session_state.img_idx + 1) % len(dataset)


# --- Main UI ---
col1, col2 = st.columns([1, 4])

with col1:
    st.button("⏭️ Next Image", on_click=next_image, use_container_width=True)
    st.write(f"**Image Index:** {st.session_state.img_idx} / {len(dataset)-1}")

    img_tensor, label_idx = dataset[st.session_state.img_idx]
    img_display = img_tensor.numpy().squeeze()

    # Scale beautifully to 0-255 grayscale
    img_display = (img_display - img_display.min()) / \
        (img_display.max() - img_display.min() + 1e-8)
    img_display = (img_display * 255).astype(np.uint8)

    st.image(
        img_display, caption=f"True Class: {idx_to_class.get(label_idx, 'Unknown')}", width=150, clamp=True)

with col2:
    x = img_tensor.unsqueeze(0)

    # 🚨 THIS IS THE FIX: We use hooks to spy on the layers safely
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Attach the spies to conv1 and conv2
    h1 = model.conv1.register_forward_hook(get_activation('conv1'))
    h2 = model.conv2.register_forward_hook(get_activation('conv2'))

    with torch.no_grad():
        # Let the model run its normal, native forward pass so BatchNorm works!
        logits = model(x)
        _, output = torch.max(logits, 1)
        output = output.item()
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    # Clean up the hooks
    h1.remove()
    h2.remove()

    # Recreate the pooling purely for the visualizer
    out_pool1_vis = F.max_pool2d(F.relu(activations['conv1']), 2, 2)
    out_pool2_vis = F.max_pool2d(F.relu(activations['conv2']), 2, 2)

    tab1, tab2, tab3 = st.tabs(["Layer 1 (Low-level features)",
                               "Layer 2 (High-level features)", "Final Output (Probabilities)"])

    with tab1:
        st.write("### Conv1 + ReLU + Pool")
        fig1 = plot_activations(
            out_pool1_vis, "32 Channels - Output of Block 1")
        st.pyplot(fig1)

    with tab2:
        st.write("### Conv2 + ReLU + Pool")
        fig2 = plot_activations(
            out_pool2_vis, "64 Channels - Output of Block 2")
        st.pyplot(fig2)

    with tab3:
        st.write("### Network Predictions")
        st.write(
            f"**Predicted Class:** {idx_to_class.get(output, 'Unknown')} (Confidence: {probs[output]*100:.2f}%)")

        top_indices = np.argsort(probs)[-5:][::-1]
        top_probs = probs[top_indices]
        top_names = [idx_to_class.get(i, f"Class {i}") for i in top_indices]

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        ax_bar.bar(top_names, top_probs, color='skyblue')
        ax_bar.set_ylabel("Probability")
        ax_bar.set_title("Top 5 Class Predictions")
        st.pyplot(fig_bar)
