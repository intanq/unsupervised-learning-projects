import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_uns(dataloader, model, loss_fn, optimizer):
    """
    Mini-batched training
    
    Args:
        dataloader ()     : 
        model (nn.Module) : 
        loss_fn ()        : loss/objective function
        optimizer ()      : optimization algorithm
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, X in enumerate(dataloader):
        X = X.to(DEVICE)
        # Compute prediction error
        
        (n, d1, d2) = X.shape
        X = torch.reshape(X, (-1, d1*d2)).float()
        # print("tu train_uns, X shape:", X.shape)
        Xrec = model(X)
        # print("tu train_uns, Xrec shape:", Xrec.shape)
        loss = loss_fn(Xrec, X)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}]/{size:>5d}")
            
def test_uns(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X in dataloader:
            X = X.to(DEVICE)
            
            (n, d1, d2) = X.shape
            X = torch.reshape(X, (-1, d1*d2)).float()
            Xrec = model(X)
            
            test_loss += loss_fn(Xrec, X).item()
    
    test_loss /= num_batches
    
    return test_loss

