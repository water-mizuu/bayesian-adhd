import torch
import torch.nn as nn
import torch.nn.functional as F  # For softmax


class EEG_CNN_LSTM_HPO(nn.Module):
        def __init__(self,
                    cnn_kernels_1=32,
                    cnn_kernel_size_1=3,
                    cnn_kernels_2=64,
                    cnn_kernel_size_2=3,
                    cnn_dropout=0.3,
                    cnn_dense=64,
                    lstm_hidden_size=64,
                    lstm_layers=2,
                    lstm_dense=64,
                    dropout=0.3,
                    num_classes=2):
            super().__init__()

            # --- CNN feature extractor (configurable) ---
            pad1 = cnn_kernel_size_1 // 2
            self.conv1 = nn.Conv2d(1, int(cnn_kernels_1), kernel_size=cnn_kernel_size_1, padding=pad1)
            self.pool1 = nn.AvgPool2d(2)
            pad2 = cnn_kernel_size_2 // 2
            self.conv2 = nn.Conv2d(int(cnn_kernels_1), int(cnn_kernels_2), kernel_size=cnn_kernel_size_2, padding=pad2)
            self.pool2 = nn.AvgPool2d(2)
            self.cnn_dropout = nn.Dropout(cnn_dropout)

            # compute dims after CNN using X_train shape present in the session
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 77, 19)
                out = self._forward_cnn(dummy)
                # We'll treat width (W) as seq_len and flatten channels*height as feature_dim
                self.seq_len = out.size(-1)           # W
                self.feature_dim = out.size(1) * out.size(2)  # C * H

            # projection per time-step from feature_dim -> cnn_dense
            self.cnn_dense = nn.Linear(self.feature_dim, int(cnn_dense))

            # --- LSTM (takes cnn_dense as input_size) ---
            self.lstm = nn.LSTM(
                input_size=int(cnn_dense),
                hidden_size=int(lstm_hidden_size),
                num_layers=int(lstm_layers),
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0
            )

            # optional dense after LSTM
            self.lstm_dense = nn.Linear(int(lstm_hidden_size), int(lstm_dense))

            # final classifier (match your original final style: dropout + linear)
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(int(lstm_dense), num_classes)
            )

        def _forward_cnn(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.cnn_dropout(x)
            return x  # (N, C, H, W)

        def forward(self, x):
            # CNN extraction
            x = self._forward_cnn(x)           # (N, C, H, W)

            # Prepare for LSTM: treat W as sequence length, flatten C*H to feature
            x = x.permute(0, 3, 1, 2)          # (N, W, C, H)
            x = x.reshape(x.size(0), x.size(1), -1)  # (N, seq_len=W, feature_dim=C*H)

            # project per time-step
            x = F.relu(self.cnn_dense(x))      # (N, seq_len, cnn_dense)

            # LSTM
            _, (h_n, _) = self.lstm(x)         # h_n: (num_layers, batch, hidden)
            out = h_n[-1]                      # last layer hidden state -> (batch, hidden)

            # post-LSTM dense + classifier
            out = F.relu(self.lstm_dense(out))
            return self.classifier(out)