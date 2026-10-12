import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np


from model import DigitClassifier
from attack import fgsm_attack


st.set_page_config(page_title="Vulnerability Auditor", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div.stButton > button:first-child {
        background-color: #00bf72; color: white; border-radius: 8px; width: 100%; font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e24; padding: 20px; border-radius: 12px; border: 1px solid #31333f; margin-top: 15px;
    }
    .success-text { color: #4cd137; font-weight: bold; font-size: 1.15em; }
    .error-text { color: #e84118; font-weight: bold; font-size: 1.15em; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources(mode):
    if mode == "MNIST (Digits Only)":
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        num_classes = 10
        base_path, rob_path = "baseline_model.pth", "robust_model.pth"
    else: 
        dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transforms.ToTensor())
        num_classes = 47
        base_path, rob_path = "emnist_baseline_model.pth", "emnist_robust_model.pth"
        
    base_model = DigitClassifier(num_classes=num_classes)
    rob_model = DigitClassifier(num_classes=num_classes)
    
    try:
        base_model.load_state_dict(torch.load(base_path, map_location=torch.device('cpu')))
        rob_model.load_state_dict(torch.load(rob_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.sidebar.warning(f"Weights not found for {mode}. Running on fallback state.")
        
    base_model.eval()
    rob_model.eval()
    return dataset, base_model, rob_model, num_classes

EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

def map_label(idx, mode):
    if mode == "MNIST (Digits Only)":
        return str(idx)
    return EMNIST_LABELS[idx] if idx < len(EMNIST_LABELS) else str(idx)


st.sidebar.title("Framework Settings")
project_mode = st.sidebar.selectbox("Select Target Dataset Framework", ["MNIST (Digits Only)", "EMNIST (Alphanumeric Mode)"])

dataset, base_model, rob_model, total_classes = load_resources(project_mode)

if 'img_idx' in st.session_state:
    if st.session_state.img_idx >= len(dataset):
        st.session_state.img_idx = 0

st.sidebar.title("Attack Settings")
epsilon = st.sidebar.slider("Perturbation Intensity (Epsilon, ε)", 0.0, 0.30, 0.15, step=0.05)

st.sidebar.title("Sample Selection")
if 'img_idx' not in st.session_state:
    st.session_state.img_idx = 42

st.sidebar.write("") 
if st.sidebar.button("Fetch Fresh Random Sample"):
    st.session_state.img_idx = int(np.random.randint(0, len(dataset) - 1))
    st.rerun()


image, label = dataset[st.session_state.img_idx]


image_input_base = image.clone().unsqueeze(0)
image_input_base.requires_grad = True
base_out = base_model(image_input_base)
base_loss = nn.CrossEntropyLoss()(base_out, torch.tensor([label]))
base_model.zero_grad()
base_loss.backward()


adv_image = fgsm_attack(image.unsqueeze(0), epsilon, image_input_base.grad.data)


with torch.no_grad():
    base_clean_pred = base_out.argmax(1).item()
    base_clean_conf = torch.softmax(base_out, dim=1)[0][base_clean_pred].item() * 100
    
    base_adv_out = base_model(adv_image)
    base_adv_pred = base_adv_out.argmax(1).item()
    base_adv_conf = torch.softmax(base_adv_out, dim=1)[0][base_adv_pred].item() * 100
    
    rob_adv_out = rob_model(adv_image)
    rob_adv_pred = rob_adv_out.argmax(1).item()
    rob_adv_conf = torch.softmax(rob_adv_out, dim=1)[0][rob_adv_pred].item() * 100


st.markdown(f"""
    <div style='text-align: center; margin-top: -50px; margin-bottom: 30px;'>
        <h1 style='font-size: 2.8em; margin-bottom: 5px;'>Adversarial Vulnerability Auditor</h1>
        <p style='font-size: 1.1em; color: #a3a8b4;'>
            Active Pipeline Mode: <b>{project_mode}</b> | 
            Target Dataset Index: <b>#{st.session_state.img_idx}</b> | 
            True Label: <b>{map_label(label, project_mode)}</b>
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Original Input Image")
    

    st.image(image.squeeze().numpy(), use_container_width=True, clamp=True)
    
    st.markdown(f"""
    <div class='metric-card'>
        <h4>Original Data Analytics</h4>
        <p><b>True Label Target:</b> {map_label(label, project_mode)}</p>
        <p><b>Model Initial Output:</b> {map_label(base_clean_pred, project_mode)}</p>
        <p><b>Initial Confidence:</b> {base_clean_conf:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("2. Baseline Model View")
    
    st.image(adv_image.squeeze().detach().numpy(), use_container_width=True, clamp=True)
    
    base_status_color = "error-text" if base_adv_pred != label else "success-text"
    base_status_msg = "Compromised" if base_adv_pred != label else "Resisted Attack"
    
    st.markdown(f"""
    <div class='metric-card'>
        <h4>Attack Ingestion Output</h4>
        <p><b>Model Prediction:</b> {map_label(base_adv_pred, project_mode)}</p>
        <p><b>Prediction Confidence:</b> {base_adv_conf:.2f}%</p>
        <p><b>Security Status:</b> <span class='{base_status_color}'>{base_status_msg}</span></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.subheader("3. Robust Model View")
    
    st.image(adv_image.squeeze().detach().numpy(), use_container_width=True, clamp=True)
    
    rob_status_color = "success-text" if rob_adv_pred == label else "error-text"
    rob_status_msg = "Integrity Maintained" if rob_adv_pred == label else "Misclassified"
    
    st.markdown(f"""
    <div class='metric-card'>
        <h4>Defensive System Output</h4>
        <p><b>Model Prediction:</b> {map_label(rob_adv_pred, project_mode)}</p>
        <p><b>Prediction Confidence:</b> {rob_adv_conf:.2f}%</p>
        <p><b>Security Status:</b> <span class='{rob_status_color}'>{rob_status_msg}</span></p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")