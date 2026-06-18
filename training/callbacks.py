# --- Helpful: show the LR each epoch ---
class LrPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = (self.model.optimizer.learning_rate.numpy()
              if not hasattr(self.model.optimizer.learning_rate, '__call__')
              else tf.keras.backend.get_value(self.model.optimizer.learning_rate(self.model.optimizer.iterations)))
        print(f"\n[Epoch {epoch+1}] learning_rate={lr:.6g}")

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [
    # 1) Reduce LR when val_loss stalls (stabilizes training)
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=4, factor=0.5, min_lr=1e-6, verbose=1
    ),
    # 2) Save the best model by val_loss
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_by_valloss.keras', monitor='val_loss',
        save_best_only=True, save_weights_only=False, verbose=1
    ),
    # 3) TensorBoard logs (scalars, graphs)
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
    # 4) CSV log of metrics per epoch (handy for plots later)
    tf.keras.callbacks.CSVLogger('training_log.csv', append=True),
    # 5) Auto-resume training state if the session restarts
    tf.keras.callbacks.BackupAndRestore(backup_dir='./backup'),
    # 6) Print LR at the end of each epoch
    LrPrinter(),
]
print(f"[TensorBoard] run: {log_dir}")
