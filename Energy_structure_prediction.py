import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import time
from sklearn.metrics import mean_absolute_error

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(48)
data = pd.read_csv("./Data.csv")
data.plot()
plt.show()

data = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
p = 120

train_data = data[:-p]
test_data = data[-p:]

min_data = np.min(train_data, axis=0)
max_data = np.max(train_data, axis=0)
train_data_scaler = (train_data - min_data) / (max_data - min_data + 1e-8)
def get_x_y(data, step=1):
    x_y = []
    for i in range(len(data) - step):
        x = torch.tensor(data[i:i + step, :], dtype=torch.float32).to(device)
        y = torch.tensor(data[i + step, :], dtype=torch.float32).unsqueeze(0).to(device)
        x_y.append([x, y])
    return x_y
def get_mini_batch(data, batch_size=1):
    for i in range(0, len(data) - batch_size, batch_size):
        samples = data[i:i + batch_size]
        x, y = zip(*samples)
        yield torch.stack(x), torch.stack(y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class RAN(nn.Module):

    def __init__(self, hidden_size, num_layers, output_size, batch_size, num_heads=4):
        super(RAN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.input_size = 8
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True).to(device)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_size, self.output_size * 8).to(device)
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, input, hidden):
        rnn_out, hidden = self.rnn(input, hidden)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        pred = self.fc(attn_out[:, -1, :])
        pred = pred.view(-1, 8, self.output_size)
        return pred, hidden

if __name__ == '__main__':
    time_step = 360
    train_x_y = get_x_y(train_data_scaler, step=time_step)
    random.shuffle(train_x_y)
    batch_size = 32
    hidden_size = 120
    num_layers = 4
    output_size = 1

    model = RAN(hidden_size, num_layers, output_size, batch_size).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8, verbose=True)
    print(model)
    epochs = 100
    loss_list = []

    sum_constraint_weight = 0.3
    max_data_tensor = torch.tensor(max_data, dtype=torch.float32).to(device)
    min_data_tensor = torch.tensor(min_data, dtype=torch.float32).to(device)

    for i in range(epochs):
        start = time.time()
        model.train()
        loss_all = 0
        num = 0
        model.hidden = model.init_hidden(batch_size)

        for seq_batch, label_batch in get_mini_batch(train_x_y, batch_size):
            optimizer.zero_grad()

            y_pred, model.hidden = model(seq_batch, model.hidden)

            model.hidden = model.hidden.detach()
            mse_loss = loss_function(y_pred, label_batch.view_as(y_pred))

            input_last_step = seq_batch[:, -1, :]
            input_last_denorm = input_last_step * (max_data_tensor - min_data_tensor) + min_data_tensor

            y_pred_denorm = y_pred.squeeze(-1) * (max_data_tensor - min_data_tensor) + min_data_tensor

            sum_pred = torch.sum(y_pred_denorm, dim=1)
            sum_loss = torch.mean((sum_pred - 100.0) ** 2)

            total_loss = (
                    mse_loss +
                    sum_constraint_weight * sum_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_all += total_loss.item()
            num += 1

        loss_all /= num
        loss_list.append(loss_all)
        print(f'epoch:{i:3} loss:{loss_all:10.8f} time:{time.time() - start:6}')
        scheduler.step(loss_all)

    plt.plot(list(range(len(loss_list))), loss_list, '_')
    plt.show()

    torch.save(model.state_dict(), 'RNN-1.pth')

    np.save('RNN-1_min.npy', min_data)
    np.save('RNN-1_max.npy', max_data)

    def multi_step_forecast(model, min_data, max_data, initial_input, steps=120):
        current_window = initial_input.copy()
        predictions = []

        for _ in range(steps):
            input_scaled = (current_window - min_data) / (max_data - min_data + 1e-8)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                model.hidden = model.init_hidden(batch_size=1)
                pred_scaled, model.hidden = model(input_tensor, model.hidden)
                pred = pred_scaled.cpu().numpy().squeeze(0).squeeze(-1) * (max_data - min_data) + min_data
                pred = np.where(pred < 0, 0.00001, pred)

            current_window = np.vstack([current_window[1:], pred])
            predictions.append(pred)
        return np.array(predictions)

    forecast_steps = 180
    last_window = data[-time_step:]
    future_pred = multi_step_forecast(
        model=model,
        min_data=min_data,
        max_data=max_data,
        initial_input=last_window,
        steps=forecast_steps)

    pred_df = pd.DataFrame(future_pred,
                           columns=['Hy', 'Wind', 'Solar', 'Oil',
                                    'Coal', 'Renew', 'Gas', 'Nuc'])
    pred_df.to_excel('Forecast.xlsx', index=False)

    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.plot(np.arange(forecast_steps), future_pred[:, i], label=f'Feature {i + 1}')
    plt.title(f'Future {forecast_steps} Steps Forecast')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    model.eval()
    with torch.no_grad():
        model.hidden = model.init_hidden(batch_size=1)

        test_pred = np.zeros(shape=(p, 8))
        for i in range(len(test_data)):
            x = train_data_scaler[-time_step:, :]
            x1 = torch.tensor(np.expand_dims(x, 0), dtype=torch.float32).to(device)
            test_y_pred_scalar, model.hidden = model(x1, model.hidden)
            test_y_pred_scalar = test_y_pred_scalar.cpu().squeeze().numpy()
            train_data_scaler = np.append(train_data_scaler, np.expand_dims(test_y_pred_scalar, 0), axis=0)
            y = test_y_pred_scalar * (max_data - min_data) + min_data
            y = np.where(y < 0, 0.00001, y)
            test_pred[i, :] = y

        mae_per_feature = mean_absolute_error(test_data, test_pred, multioutput='raw_values')
        error_percentage = np.abs((test_pred - test_data) / test_data) * 100
        mean_error_percentage = np.mean(error_percentage, axis=0)

        for i, (mae, ep) in enumerate(zip(mae_per_feature, mean_error_percentage)):
            print(f'Feature {i + 1} Mean Absolute Error: {mae:.4f}, Mean Error Percentage: {ep:.2f}%')

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(np.arange(8) - 0.2, mae_per_feature, width=0.4, label='MAE', color='b')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('MAE')
        ax1.set_title('MAE and Mean Error Percentage for Each Feature')
        ax1.set_xticks(np.arange(8))
        ax1.set_xticklabels([f'Feature {i + 1}' for i in range(8)])
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.bar(np.arange(8) + 0.2, mean_error_percentage, width=0.4, label='Mean Error Percentage', color='r',
                alpha=0.5)
        ax2.set_ylabel('Mean Error Percentage (%)')
        ax2.legend(loc='upper right')
        plt.show()
        x_in = list(range(len(test_pred)))

        plt.figure(figsize=(12, 6))
        for i in range(8):
            plt.plot(x_in, test_data[:, i], f'C{i}o-', label=f'True Feature {i + 1}')
            plt.plot(x_in, test_pred[:, i], f'C{i}+-', label=f'Pred Feature {i + 1}')
        plt.legend(loc='upper right')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.title('Actual vs Predicted Values')
        plt.show()

        columns = [f'{feat}true' for feat in ['Hy', 'Wind', 'Solar', 'Oil', 'Coal', 'Renew', 'Gas', 'Nuc']] + \
                  [f'{feat}pred' for feat in ['Hy', 'Wind', 'Solar', 'Oil', 'Coal', 'Renew', 'Gas', 'Nuc']]
        results = np.hstack((test_data, test_pred))
        df_results = pd.DataFrame(results, columns=columns)

        plt.figure(figsize=(16, 8))

        original_time = np.arange(len(data))
        forecast_time = np.arange(len(data), len(data) + forecast_steps)

        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()

        feature_names = ['Hy', 'Wind', 'Solar', 'Oil', 'Coal', 'Renew', 'Gas', 'Nuc']

        for i in range(8):
            axes[i].plot(original_time, data[:, i],
                         label='Original',
                         color='blue',
                         alpha=0.7)

            axes[i].plot(forecast_time, future_pred[:, i],
                         label='Forecast',
                         color='red',
                         linestyle='--')

            axes[i].axvline(x=len(data) - 1,
                            color='gray',
                            linestyle=':',
                            linewidth=2,
                            label='Prediction Start')

            axes[i].set_title(f'{feature_names[i]} - Actual vs Forecast')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(18, 9))

        combined_data = np.vstack([data, future_pred])
        combined_time = np.arange(len(combined_data))

        for i in range(8):
            plt.plot(combined_time, combined_data[:, i],
                     label=feature_names[i],
                     alpha=0.8,
                     linewidth=1.5)

        plt.axvline(x=len(data) - 1,
                    color='black',
                    linestyle='--',
                    linewidth=2,
                    label='Prediction Start')

        plt.title('Combined Historical Data and Forecast')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()