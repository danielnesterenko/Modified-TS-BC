import cv2
import torch
import tqdm
from torchvision import transforms

'''
# Define the video path
video_path = 'trainingDataVAE/thirsty-lavender-koala-f153ac423f61-20220414-192402_video_depth.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define a transform to convert frames to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((216, 384))  # Resize to the desired dimensions
])

# Read and preprocess frames
frames = []
for _ in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# Convert BGR to RGB
    frame_tensor = transform(frame)
    frames.append(frame_tensor)

# Stack frames into a single tensor
video_tensor = torch.stack(frames)

# Close the video file
cap.release()

print(f"Video tensor shape: {video_tensor.shape}")  # Should be [6001, 3, 360, 640]
'''
#----

import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(256 * 27 * 48, 1024)
        self.enc_fc2_mu = nn.Linear(1024, 512)
        self.enc_fc2_logvar = nn.Linear(1024, 512)
        
        # Decoder
        self.dec_fc1 = nn.Linear(512, 1024)
        self.dec_fc2 = nn.Linear(1024, 256 * 27 * 48)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
    def encode(self, x):
        #print(f'Start: {x.shape}')
        h = F.relu(self.enc_conv1(x))
        #print('After enc_conv1:', h.shape)
        h = F.relu(self.enc_conv2(h))
        #print('After enc_conv2:', h.shape)
        h = F.relu(self.enc_conv3(h))
        #print('After enc_conv3:', h.shape)
        h = h.view(h.size(0), -1)
        #print('After flatten:', h.shape)
        h = F.relu(self.enc_fc1(h))
        return self.enc_fc2_mu(h), self.enc_fc2_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(h.size(0), 256, 27, 48)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        return torch.sigmoid(self.dec_conv3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


'''
# Instantiate the model
vae = VAE()

#----

import torch.optim as optim

def main():

    # Define the loss function
    def loss_function(recon_x, x, mu, logvar):
        mse_loss = nn.MSELoss(reduction="mean")

        BCE = mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    
    # Define the loss function
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    

    # Define the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=3e-4)

    # Training loop
    num_epochs = 10
    batch_size = 256

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for i in range(0, len(video_tensor), batch_size):
            batch = video_tensor[i:i+batch_size]
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch.float())
            loss = loss_function(recon_batch, batch.float(), mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(video_tensor)}")

    torch.save(vae.state_dict(), 'vae_model_yasd.pth')
    print("Model saved as vae_model_y.pth")


if __name__ == "__main__":
    main()

'''