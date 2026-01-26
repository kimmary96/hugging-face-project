import matplotlib.pyplot as plt

# 로그에서 추출한 데이터 (Epoch 기준)
epochs = [0.74, 1.44, 2.15, 3.59, 4.3, 5.0, 5.74, 6.44, 7.15, 7.89, 8.59, 9.3, 10.0]

# 학습 데이터에 대한 오차 (점점 줄어들어야 정상)
train_loss = [
    1.7124, 0.9581, 0.6647, 0.4804, 0.4212, 0.4011, 
    0.3652, 0.3321, 0.3188, 0.3067, 0.2898, 0.2808, 0.2763
]

# 검증 데이터(안 본 문제)에 대한 오차 (줄어들다 멈추거나 올라가면 오버피팅)
eval_loss = [
    1.1527, 0.7854, 0.6481, 0.5511, 0.5209, 0.5011, 
    0.4909, 0.4861, 0.4822, 0.4833, 0.4831, 0.4810, 0.4804
]

plt.figure(figsize=(10, 6))

# Train Loss 그리기 (파란색)
plt.plot(epochs, train_loss, label='Training Loss', marker='o', color='blue', linestyle='-')

# Eval Loss 그리기 (빨간색)
plt.plot(epochs, eval_loss, label='Validation Loss', marker='x', color='red', linestyle='--', linewidth=2)

# 그래프 꾸미기
plt.title('Learning Curve: Exp2_Overfitting (Epoch 10)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 오버피팅/포화 지점 표시 (Epoch 7.15 부근)
plt.axvline(x=7.15, color='green', linestyle=':', label='Optimal Point')
plt.text(7.3, 0.6, 'Saturation Point\n(Loss 0.48)', color='green')

plt.tight_layout()
plt.show()