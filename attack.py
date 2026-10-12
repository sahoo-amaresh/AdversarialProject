import torch

def fgsm_attack(image, epsilon, data_grad):
    """
    image: The original clean image
    epsilon: How much noise/distortion to add (higher means more distorted)
    data_grad: The gradient direction that makes the model maximize its error
    """
   
    sign_data_grad = data_grad.sign()
    

    perturbed_image = image + epsilon * sign_data_grad
    
    
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image