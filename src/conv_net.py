from torch import nn
import torch.functional as F
import torch


class Conv_NET(nn.Module):
    def __init__(self):
        '''
        Construcción de la Red
        '''
        super(Conv_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.flatten_dim = self._get_flatten_dim()

        self.fc1 = nn.Linear(self.flatten_dim, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def _get_flatten_dim(self):
        x = torch.zeros(1, 3, 32, 32)  # (batch_size, canales, alto, ancho)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.numel()  # número total de elementos

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_net(self,epochs,data_loader,criterion,optimizer,cuda=False, plot=True):
        '''
        Define una función de entrenamiento, ten en cuenta la forma en la que llegan 
        los datos de data_loader, e itera sobre ellos. Realiza también el caso 
        en que se llegue a utilizar cuda. Muestra una gráfica al finalizar el
        entrenamiento usando el lost obtenido.
        '''
        self.train()  

        device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.to(device)

        losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()           
                outputs = self(inputs)          
                loss = criterion(outputs, labels)  
                loss.backward()                 
                optimizer.step()                

                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loader)
            losses.append(epoch_loss)
            if epoch % 10 == 0:
                print(f"Época {epoch+1}/{epochs}, Pérdida: {epoch_loss:.4f}")

        # Graficar la pérdida
        if plot:
            plt.plot(range(1, epochs+1), losses, marker='o')
            plt.xlabel('Épocas')
            plt.ylabel('Pérdida')
            plt.title('Curva de pérdida durante el entrenamiento')
            plt.grid(True)
            plt.show()
        
        return losses
