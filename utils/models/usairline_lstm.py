import torch.nn as nn
import torch.nn.init as init


class LSTMOutOnly(nn.Module):
    """
    一个自定义 LSTM 层，内部调用 nn.LSTM，但只返回 out，不返回 (h, c)。
    并且不会对输入 x 做 in-place 修改。
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
    def forward(self, x):
        # PyTorch 的 LSTM 返回 (out, (h, c))
        # 只接收 out，不接收 (h, c)
        out, _ = self.lstm(x)
        # 此时 out.shape = (batch_size, seq_len, hidden_size) (如果 batch_first=True)
        return out

class USAirline_LSTM(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_classes=3):
        super().__init__()
        # ======== LSTM特征提取器 (单层) ========
        # PyTorch 中, LSTM 默认激活函数就是 tanh, recurrent_activation 就是 sigmoid
        # batch_first=True 表示输入形状为 (batch, seq_len, input_size)

        self.custom_lstm = LSTMOutOnly(input_size=input_size,
                                       hidden_size=hidden_size)

        self.bn_lstm = nn.BatchNorm1d(hidden_size, eps=1e-3, momentum=0.99)

        # ======== 全连接分类器 ========
        self.cls = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        """
        期望 x 的形状: (batch_size, seq_len, input_size)
        对应 create_us_airline_lstm 的 input_shape = (seq_len, input_size).
        """
        # LSTM 输出: (batch_size, seq_len, hidden_size)
        out = self.custom_lstm(x)

        # 仅取最后一个时间步的输出:
        # out[:, -1, :] => (batch_size, hidden_size)
        last_out = out[:, -1, :]

        # 批归一化: 形状 (batch_size, hidden_size)
        last_out = self.bn_lstm(last_out)

        # 全连接 + softmax
        logits = self.cls(last_out)
        out = self.softmax(logits)
        return out

    def _initialize_weights(self):
        """与 Credit_FCN 中相同的初始化策略, 仅针对 Linear 层."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias, 0.01)
