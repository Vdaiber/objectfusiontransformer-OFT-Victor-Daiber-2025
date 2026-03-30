# After optimizer step, check gradients for NaN/Inf
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            print(f"[ERROR] NaN/Inf detected in gradients of {name}!")
# Print current learning rate
for param_group in optimizer.param_groups:
    print(f"[DEBUG] Current learning rate: {param_group['lr']}") 