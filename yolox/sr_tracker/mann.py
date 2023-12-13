import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, memory_size, memory_feature_size):
        super().__init__()
        self.register_buffer('memory', torch.Tensor(
            memory_size, memory_feature_size))

        # Initialize memory bias
        nn.init.constant_(self.memory, 0.0)

    def reset(self):
        """Initialize memory from bias, for start-of-sequence."""
        nn.init.constant_(self.memory, 0.0)

    def update(self, keys, updates):
        keys = F.softmax(keys, dim=1)
        updates = torch.mm(keys.t(), updates)
        self.memory.data = self.memory.data * 0.999 + updates * 0.001

    def read(self, keys):
        keys = F.softmax(keys, dim=1)
        return torch.mm(keys, self.memory.t())


class MANN(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_feature_size):
        super(MANN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_feature_size = memory_feature_size

        # LSTM layer
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        # Predict head
        self.predict_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size // 2, 4))

    def set_wh(self, wh):
        self.w = int(wh[0])
        self.h = int(wh[1])

    def forward(self, x, contexts, lengths):
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state and cell state
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []
        for i in range(seq_len):
            h, c = self.lstm(x[:, i, :], (h, c))

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        # Predict
        preds = self.predict_head(outputs)
        return preds

    def to_tensor(self, xs, device):
        assert isinstance(xs, list)
        for i in range(len(xs)):
            xs[i] = torch.from_numpy(xs[i].astype(np.float32)).to(device)

        return xs

    @torch.no_grad()
    def predict(self, rx, rhidden, rcell, device, sample_rate, samples=30):
        x, hidden, cell = self.to_tensor([rx, rhidden, rcell], device)

        # LSTM step
        hidden, cell = self.lstm(x, (hidden, cell))

        # Predict
        pred = self.predict_head(hidden)

        pred = pred.cpu().numpy()
        pred[:, 2] *= pred[:, 3]
        pred[:, 0] -= pred[:, 2] / 2
        pred[:, 1] -= pred[:, 3]

        x1, y1, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        xc, y2 = x1 + w / 2, y1 + h
        old_x1, old_y1, old_w, old_h = rx[:, 0], rx[:, 1], rx[:, 2], rx[:, 3]
        old_xc, old_y2 = old_x1 + old_w / 2, old_y1 + old_h

        pos = np.zeros_like(rx)

        pos[:, 0] = x1
        pos[:, 1] = y1
        pos[:, 2] = w
        pos[:, 3] = h
        pos[:, 4] = xc
        pos[:, 5] = y2
        pos[:, 6] = x1 - old_x1
        pos[:, 7] = y1 - old_y1
        pos[:, 8] = w - old_w
        pos[:, 9] = h - old_h
        pos[:, 10] = xc - old_xc
        pos[:, 11] = y2 - old_y2
        pos[:, 12] = (x1 - old_x1) / sample_rate
        pos[:, 13] = (y1 - old_y1) / sample_rate
        pos[:, 14] = (w - old_w) / sample_rate
        pos[:, 15] = (h - old_h) / sample_rate
        pos[:, 16] = (xc - old_xc) / sample_rate
        pos[:, 17] = (y2 - old_y2) / sample_rate
        pos[:, 18] = sample_rate / samples
        pos[:, 19] = sample_rate / samples

        hidden = hidden.cpu().numpy()
        cell = cell.cpu().numpy()

        return pos, hidden, cell

    @torch.no_grad()
    def initiate(self, tlwh, sample_rate, samples=30):
        x1, y1, w, h = tlwh
        x1, y1, w, h = x1 / self.w, y1 / self.h, w / self.w, h / self.h
        xc, y2 = x1 + w / 2, y1 + h
        pos = [x1, y1, w, h,
               xc, y2,
               0, 0, 0, 0,
               0, 0,
               0, 0, 0, 0,
               0, 0,
               sample_rate / samples, sample_rate / samples]
        pos = np.array(pos, dtype=np.float32)
        cov = np.diag([1e-2, 1e-2, 1e-2, 1e-2,
                       1e-2, 1e-2, 1e-2, 1e-2])
        h = np.zeros((self.hidden_size))
        c = np.zeros((self.hidden_size))
        return pos, [h, c], cov

    def update(self, mean, pred_mean, covariance, new_tlwh, sample_rate, samples=30):
        x1, y1, w, h = new_tlwh
        x1, y1, w, h = x1 / self.w, y1 / self.h, w / self.w, h / self.h
        xc, y2 = x1 + w / 2, y1 + h
        old_x1, old_y1, old_w, old_h = mean[0], mean[1], mean[2], mean[3]
        old_xc, old_y2 = mean[4], mean[5]
        pos = [x1, y1, w, h,
               xc, y2,
               x1 - old_x1, y1 - old_y1, w - old_w, h - old_h,
               xc - old_xc, y2 - old_y2,
               (x1 - old_x1) / sample_rate, (y1 - old_y1) /
               sample_rate, (w - old_w) /
               sample_rate, (h - old_h) / sample_rate,
               (xc - old_xc) / sample_rate, (y2 - old_y2) / sample_rate,
               sample_rate / samples, sample_rate / samples]
        pos = np.array(pos, dtype=np.float32)

        pred_mean[0] = x1  # pred_mean[0] * 0.35 + x1 * 0.65
        pred_mean[1] = y1  # pred_mean[1] * 0.35 + y1 * 0.65
        pred_mean[2] = w  # pred_mean[2] * 0.35 + w * 0.65
        pred_mean[3] = h  # pred_mean[3] * 0.35 + h * 0.65
        return pos, pred_mean, covariance
