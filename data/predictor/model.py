import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len=5, input_dim)
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, decoder_input_dim, hidden_dim, output_dim=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(decoder_input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, future_fd, hidden, cell, output_len):
        """
        future_fd: (batch, output_len) - fixture difficulty per future step
        hidden, cell: from encoder
        output_len: number of steps to predict (e.g., 3)
        """
        batch_size = future_fd.size(0)
        outputs = []
        
        # Start with zero input (can also try last encoder value)
        decoder_input = torch.zeros((batch_size, 1, 1), device=future_fd.device)  # just a scalar input

        for t in range(output_len):
            fd_step = future_fd[:, t].unsqueeze(1).unsqueeze(2)  # shape (batch, 1, 1)
            input_combined = torch.cat([decoder_input, fd_step], dim=2)  # (batch, 1, 2)

            output, (hidden, cell) = self.lstm(input_combined, (hidden, cell))
            pred = self.fc(output.squeeze(1))  # (batch,)
            outputs.append(pred)

            decoder_input = pred.unsqueeze(1)  # autoregressive

        return torch.cat(outputs, dim=1)  # (batch, output_len)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.encoder = LSTMEncoder(encoder_input_dim, hidden_dim)
        self.decoder = LSTMDecoder(decoder_input_dim, hidden_dim, output_dim)

    def forward(self, encoder_inputs, future_fd):
        """
        encoder_inputs: (batch, 5, encoder_input_dim)
        future_fd: (batch, 3) – fixture difficulty
        """
        hidden, cell = self.encoder(encoder_inputs)
        outputs = self.decoder(future_fd, hidden, cell, output_len=future_fd.size(1))
        return outputs  # shape: (batch, 3)

if __name__ == "__main__":
    #do some testing
    input_tensor = torch.randn(32, 5, 25)  # batch_size=32, seq_len=5, feature_dim=25
    future_fd_tensor = torch.randn(32, 3) # batch_size=32, output_len=3
    
    model = Seq2SeqLSTM(encoder_input_dim=25, decoder_input_dim=2, hidden_dim=64, output_dim=1)

    output = model(input_tensor, future_fd_tensor)

    print("Output shape:", output.shape)  # Expected: (32, 3)