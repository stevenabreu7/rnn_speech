import data
import model
from timeit import default_timer as timer

# initialize the RNN
# TODO correct output size (check the maps in `tools`)
# TODO correct hidden size (trial and error? research?)
rnn = model.RecurrentModel(13, 500, 5, 30)

# initializer data loaders
val_loader = data.val_loader()
train_loader = data.train_loader()

# initialize trainer
# TODO correct number of epochs for training
n_epochs = 2
start = timer()
trainer = model.ModelTrainer(rnn, val_loader, train_loader, n_epochs)

# run the training
trainer.train()
end = timer()
print(end - start)
trainer.save()
