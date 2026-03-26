import matplotlib.pyplot as plt

train_acc = [0.52,0.77,0.85,0.88,0.90]
val_acc = [0.16,0.71,0.87,0.55,0.69]

epochs = range(1,6)

plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")

plt.legend()
plt.show()