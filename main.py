import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from model import DigitClassifier
from attack import fgsm_attack


BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"System running on device: {DEVICE}")


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


baseline_model = DigitClassifier().to(DEVICE)
robust_model = DigitClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
robust_optimizer = optim.Adam(robust_model.parameters(), lr=LEARNING_RATE)


def train_baseline():
    print("\n>>> Training Baseline Model (Normal/Defenseless)...")
    baseline_model.train()
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = baseline_model(images)
            loss = criterion(outputs, labels)
            
            baseline_optimizer.zero_grad()
            loss.backward()
            baseline_optimizer.step()
        print(f"    Epoch [{epoch+1}/{EPOCHS}] Complete.")


def train_robust():
    print("\n>>> Training Robust Model (Adversarial Defense)...")
    robust_model.train()
    train_epsilon = 0.15 
    
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            images.requires_grad = True
            
            outputs = robust_model(images)
            loss = criterion(outputs, labels)
            robust_model.zero_grad()
            loss.backward(retain_graph=True)
            

            data_grad = images.grad.data
            adv_images = fgsm_attack(images, train_epsilon, data_grad)
            

            adv_outputs = robust_model(adv_images)
            adv_loss = criterion(adv_outputs, labels)
            
            robust_model.zero_grad()
            adv_loss.backward()
            robust_optimizer.step()
        print(f"    Epoch [{epoch+1}/{EPOCHS}] Complete.")


def test_robustness(model, name, test_epsilon):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images.requires_grad = True
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        adv_images = fgsm_attack(images, test_epsilon, data_grad)
        
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, predicted = adv_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = (correct / total) * 100
    print(f"[{name}] Accuracy under Attack (Epsilon={test_epsilon}): {accuracy:.2f}%")


if __name__ == "__main__":

    train_baseline()
    train_robust()
    

    torch.save(baseline_model.state_dict(), "baseline_model.pth")
    torch.save(robust_model.state_dict(), "robust_model.pth")
    print("Saved trained weights successfully!")
    print("\n================ SHOWDOWN RESULTS ================")
    test_robustness(baseline_model, "Baseline Model", test_epsilon=0.15)
    test_robustness(robust_model, "Robust Model  ", test_epsilon=0.15)