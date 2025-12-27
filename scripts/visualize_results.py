def plot_accuracy(history):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Acc', linestyle='dashed')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Training and Validation Accuracy')
    plt.legend(); plt.grid(True); plt.show()

plot_accuracy(history)

# Plot training & validation precision
def plot_precision(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision', linestyle="dashed")
    plt.xlabel('Epochs'); plt.ylabel('Precision'); plt.title('Training and Validation Precision')
    plt.legend(); plt.grid(True); plt.show()

plot_precision(history)

# Loss plot unchanged
def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss', linestyle="dashed")
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True); plt.show()

plot_loss(history)
