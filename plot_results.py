import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from model import DigitClassifier
from attack import fgsm_attack
from main import train_baseline, train_robust, baseline_model, robust_model, test_loader, criterion


epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def evaluate_over_epsilons(model, epsilon_list):
    model.eval()
    accuracy_results = []
    
    for eps in epsilon_list:
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))
            
         
            if eps > 0:
                images.requires_grad = True
                outputs = model(images)
                loss = criterion(outputs, labels)
                model.zero_grad()
                loss.backward()
                
                data_grad = images.grad.data
                perturbed_images = fgsm_attack(images, eps, data_grad)
            else:
                perturbed_images = images
                
            with torch.no_grad():
                outputs = model(perturbed_images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        acc = (correct / total) * 100
        accuracy_results.append(acc)
        print(f"  Finished evaluation for Epsilon {eps} -> Accuracy: {acc:.2f}%")
        
    return accuracy_results

if __name__ == "__main__":
   
    train_baseline()
    train_robust()
    

    print("\n>>> Testing Baseline Model across multiple attack strengths...")
    baseline_accs = evaluate_over_epsilons(baseline_model, epsilons)
    
    print("\n>>> Testing Robust Model across multiple attack strengths...")
    robust_accs = evaluate_over_epsilons(robust_model, epsilons)
    

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, baseline_accs, "o-", color="red", label="Baseline Model (Defenseless)")
    plt.plot(epsilons, robust_accs, "s-", color="green", label="Robust Model (Adversarial Training)")
    
    plt.title("Model Robustness Comparison: Baseline vs Robust")
    plt.xlabel("Attack Intensity (Epsilon)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    

    plt.savefig("robustness_comparison.png")
    print("\n[SUCCESS] Graph saved as 'robustness_comparison.png' in your project folder!")
    plt.show()