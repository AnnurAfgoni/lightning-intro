import sys
sys.path.append("src")

from models import MNISTModel
from data import MNISTDataModule


# load data
data = MNISTDataModule(data_dir="data/", batch_size=64, num_workers=11)
data.setup(stage="test")
loader = data.test_dataloader()
x, y = next(iter(loader))

# define model
# my_model = MNISTModel()
my_model = MNISTModel.load_from_checkpoint(
    "checkpoint/lightning_logs/version_4/checkpoints/epoch=4-step=4300.ckpt",
    class_dict=data.class_dict,
    map_location="cpu"
)
my_model.eval()

# predict
y_pred = my_model(x[0])
y_pred = int(y_pred.max(1)[1])

y_true = int(y[0])
print({"pred": y_pred, "true": y_true})

# access hyperparameter
print("input_size : ", my_model.input_size)
print("hidden_dim : ", my_model.hidden_dim)
print("output_size : ", my_model.output_size)
print("learning_rate : ", my_model.learning_rate)
print("class dict : ", my_model.class_dict)