import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #Embedded layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias= True,
                            batch_first= True,
                            dropout = 0.0)
        
        #full connected layer to map the output of the LSTM layer to vocab size        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device = device),
                torch.zeros(1, batch_size, self.hidden_size, device = device))
    
    def forward(self, features, captions):
        
        # features is of shape (batch_size, embed_size)
        #Defining batch size
        batch_size = features.shape[0]
        
        self.hidden = self.init_hidden(batch_size)
        
        #embedding the caption
        embeds = self.embed(captions[:,:-1])
        
        #taking both features and embedded captions
        embedded = torch.cat((features.unsqueeze(1), embeds), dim=1)
        
        #passing through the lstm
        lstm_out, self.hidden = self.lstm(embedded, self.hidden)
        
        #passing through the fully connected layer to get the desired vocab size
        out = self.fc(lstm_out)
        
        return out
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        self.batch_size = inputs.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        word_count = 0
        
        with torch.no_grad():
            while word_count<max_len:
                outputs, self.hidden = self.lstm(inputs, self.hidden)
                outputs = self.fc(outputs)
                outputs = outputs.squeeze(1)
                outputs = outputs.argmax(dim=1)
                
                output.append(outputs.item())
                
                inputs = self.embed(outputs.unsqueeze(0))
                
                word_count+=1
                if outputs == 1:
                    break

                
            
        return output