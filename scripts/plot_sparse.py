import matplotlib.pyplot as plt
import numpy as np

epochs = np.array([log["epoch"] for log in trainer.state.log_history if "loss" in log])
loss = np.array([log["loss"] for log in trainer.state.log_history if "loss" in log])
stop_epoch = trainer.state.epoch  # The epoch at which training stopped

# Label the axes and title the plot
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

# Plot the training loss and validation loss
plt.plot(epochs, loss,label="Training Loss", color="blue")
# Add a vertical dashed line at the epoch where training stopped
plt.plot(stop_epoch, epochs, label="Stopped Epoch", color="red", linestyle='dotted')
plt.legend()

# Save the plot to the papers figures directory
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
plt.savefig(OUTPUT_DIR / "loss_plot.png")

plt.show()