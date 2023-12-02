train_losses = []
valid_losses = []
with open('./logs/msa-3d-sam-btcv_2023_11_13_19_41_00/Log/2023-11-13-19-41_train.log', 'r') as log_file:
    for line in log_file:
        if 'Train loss' in line:
            # 提取训练损失
            parts = line.split('Train loss: ')
            train_loss = float(parts[1].split(',')[0])
            train_losses.append(train_loss)
        elif 'Valid loss' in line:
            # 提取验证损失
            parts = line.split('Valid loss: ')
            valid_loss = float(parts[1].split(',')[0])
            for i in range(5):
                valid_losses.append(valid_loss)
train_losses = train_losses[1:]
print('Train Losses:', len(train_losses))
print('Valid Losses:', len(valid_losses))
import matplotlib.pyplot as plt


# epoch的数据，根据你的示例中提供的数据进行调整
epochs = [x for x in range(1, 71)]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-')
plt.plot(epochs, valid_losses, 'ro', linestyle='-', label='Valid Loss')  # 只显示最后一个验证损失
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)

# 保存图像（可选）
plt.savefig('loss_plot.png')

# 显示图像
plt.show()