import torch
import math
from torch.utils.data import DataLoader
import pandas as pd
from data import PhaseDiffDataSet
from model import PhaseDiffModel
import ObsGenerator as og
import tqdm


def TrainLocalization(model: PhaseDiffModel, num_epochs: int):

    train_dataset, val_dataset = PhaseDiffDataSet.train_validation_split('GeneratedPhases.csv', 'GeneratedPositions.csv', 0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    for epoch in range(num_epochs):

        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #print('outputs = ')
            #print(outputs)
            #print('labels = ')
            #print(labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #if (i+1) % 100 == 0:
            #    print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_dataloader)}, Loss: {running_loss/(i+1):.4f}")

        # validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for i, data in enumerate(val_dataloader):
                inputs, labels = data
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_dataloader)
            rmse_loss = math.sqrt(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}, Validation RMSE: {rmse_loss:.4f}")

if False:
    og.GenerateObservations(10000, True)

model = PhaseDiffModel()

TrainLocalization(model, num_epochs=100)
torch.save(model.state_dict(), 'model.pt')
