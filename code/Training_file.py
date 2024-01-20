import copy
from os.path import join
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from modules.dataset import MyDataset, find_divide_sets
from modules.testnet import test_net
from modules.utils import convert_to_list
import os
import logging
import traceback
import gc


# PC settings
load_dataset_to_gpu_ram = True
root = # Insert string with path to iBeat

dataPath = root + r"\data"
path_h5 = join(dataPath + r"\data_spect_512.h5") # Change to \data_wav_27520_npy.h5 when training WM

# tensorbord command: tensorboard --logdir runs

# Import models
import spect_model_cmcmcmcl # Change to import wav_model_cmcmcmcl when training WM

models = [
    spect_model_cmcmcmcl.Architecture, # Change to wav_model_cmcmcmcl.Architecture when training WM
]


# Setup basic logging
log_filename = "training_log.txt"
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=log_filename,
)


device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

dataset = ""
for _model in models:
    try:
        temp_model = _model(0)
        convert_to_list(temp_model, "DROPOUT_RATE")
        dropout_rates = copy.deepcopy(temp_model.DROPOUT_RATE)
        split_size = copy.deepcopy(temp_model.SPLIT_SIZE)
        del temp_model
        t.cuda.empty_cache()
        gc.collect()

        ### CREATE DATASET
        # Paths to data files
        path_json = join(dataPath, "data_snippet_bin_430.json")
        list_ids = join(dataPath, "data_ids.json")

        (train_ids, test_ids) = find_divide_sets(path_json, split_size)
        # match model.DATA_SET:
        # case "DataSpectrogram":
        train_dataset = MyDataset(
            path_h5,
            path_json,
            train_ids,
            load_dataset_to_gpu_ram,
        )
        test_dataset = MyDataset(
            path_h5,
            path_json,
            test_ids,
            load_dataset_to_gpu_ram,
        )

        for DROPOUT_RATE in dropout_rates:
            model = _model(DROPOUT_RATE)
            model.to(device)
            convert_to_list(model, "BATCH_SIZE")
            convert_to_list(model, "LEARNING_RATE")
            for BATCH_SIZE in model.BATCH_SIZE:
                ### CREATE DATALOADER
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=model.DATA_SHUFFLE,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                for LEARNING_RATE in model.LEARNING_RATE:
                    ### CREATE CRITERION AND OPTIMIZERS
                    (criterions, optimizers) = model.init_criterion_optimizer(
                        LEARNING_RATE
                    )
                    if type(criterions) is not list:
                        criterions = [criterions]
                    if type(optimizers) is not list:
                        optimizers = [optimizers]
                    for criterion in criterions:
                        for optimizer in optimizers:
                            try:
                                for module in model.modules():
                                    if hasattr(module, 'reset_parameters'):
                                        module.reset_parameters()
                                print(
                                    "\n\n\033[1m" + f"Running: {model.code}" + "\033[0m"
                                )  # Bold
                                print("with hyperparms:")
                                print(
                                    "\033[94m"
                                    + f"  Dropout Rate    : {DROPOUT_RATE:>10}"
                                    + "\033[0m"
                                )  # Blue
                                print(
                                    "\033[92m"
                                    + f"  Batch Size      : {BATCH_SIZE:>10}"
                                    + "\033[0m"
                                )  # Green
                                print(
                                    "\033[93m"
                                    + f"  Learning Rate   : {LEARNING_RATE:>10}"
                                    + "\033[0m"
                                )  # Yellow
                                print(
                                    "\033[91m"
                                    + f"  Criterion       : {type(criterion).__name__:>10}"
                                    + "\033[0m"
                                )  # Red
                                print(
                                    "\033[95m"
                                    + f"  Optimizer       : {type(optimizer).__name__:>10}"
                                    + "\033[0m\n"
                                )  # Magenta

                                ### CREATE SCHEDULER
                                scheduler = model.init_scheduler(optimizer)

                                # Variable to store average losses per epoch
                                losses = []

                                # Variable to store current epoch
                                epoch = 0

                                best_test_loss = -1
                                best_test_loss_at_epoch = -1

                                hyperparams = f"LR_{LEARNING_RATE}_BATCH_{BATCH_SIZE}_DROP_{DROPOUT_RATE}_CRIT_{type(criterion).__name__}_OPTI_{type(optimizer).__name__}_SCHE_{type(scheduler).__name__}"
                                base_dir = root + f"/runs_hyper2/{model.code}/{hyperparams}"
                                if not os.path.exists(base_dir):
                                    os.makedirs(base_dir)

                                i = 1

                                while os.path.exists(f"{base_dir}/{i}"):
                                    i += 1

                                writer = SummaryWriter(
                                    root + f"/runs_hyper2/{model.code}/{hyperparams}/{i}"
                                )
                                (test_loss, _, _) = test_net(
                                    model, test_loader, device, criterion
                                )
                                test_loss_start = test_loss

                                while True:
                                    model.train()
                                    running_loss = 0.0

                                    for batch_idx, (data, target) in enumerate(
                                        train_loader
                                    ):
                                        # Send data to GPU
                                        data, target = data.to(device), target.to(
                                            device
                                        )

                                        # Zero the parameter gradients
                                        optimizer.zero_grad()

                                        # Forward + backward + optimize
                                        output = model(data)
                                        loss = criterion(output.view(-1,430), target)
                                        loss.backward()
                                        optimizer.step()

                                        # Print batch loss
                                        # print(loss)

                                        # Store accumulated loss
                                        running_loss += loss.item()

                                    # Calculate average loss for epoch
                                    epoch_loss = running_loss / len(train_loader)
                                    (test_loss, _, _) = test_net(
                                        model, test_loader, device, criterion
                                    )
                                    print(
                                        f"Epoch {epoch}; Train loss: {epoch_loss:.6f}; Test loss: {test_loss}"
                                    )
                                    losses.append(epoch_loss)
                                    epoch += 1

                                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                                    writer.add_scalar("Loss/test", test_loss, epoch)
                                    writer.flush()

                                    # Inform scheduler of epoch loss
                                    scheduler.step(epoch_loss)

                                    if epoch == 10:
                                        test_loss_10 = test_loss


                                    # Save model
                                    if (
                                        test_loss < best_test_loss
                                        or best_test_loss == -1
                                    ):
                                        best_test_loss = test_loss
                                        best_test_loss_at_epoch = epoch
                                        t.save(
                                            model.state_dict(),
                                            (
                                                root
                                                + r"/code/"
                                                + f"{model.code}/best_params/{model.code}_{hyperparams}_{i}.pth"
                                            ),
                                        )
                                    if (
                                        epoch - best_test_loss_at_epoch
                                        > model.TEST_INCREASE_LIMIT
                                    ) and (epoch > model.MIN_EPOCHS):
                                        break
                                writer.add_hparams(
                                    {
                                        "batch_size": BATCH_SIZE,
                                        "data_shuffle": model.DATA_SHUFFLE,
                                        "learning_rate": LEARNING_RATE,
                                        "schedule_factor": model.SCHEDULE_FACTOR,
                                        "schedule_patience": model.SCHEDULE_PATIENCE,
                                        "desired_loss": model.DESIRED_LOSS,
                                        "test_increase_steps": model.TEST_INCREASE_LIMIT,
                                        "split_size": model.SPLIT_SIZE,
                                        "dropout_rate": DROPOUT_RATE,
                                    },
                                    {
                                        "hparam/loss_start": test_loss_start,
                                        "hparam/loss_10": test_loss_10,
                                        "hparam/loss_min": best_test_loss,
                                        "hparam/best_loss_epoch": best_test_loss_at_epoch,
                                    },
                                )
                            except Exception as e:
                                error_message = f"Failed model training: {str(e)}\n{traceback.format_exc()}"
                                logging.error(error_message)
                                print(error_message)

                            try:
                                writer.close()
                            except:
                                """"""
                            t.cuda.empty_cache()
    except Exception as e:
        error_message = f"Failed model init: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        print(error_message)
        try:
            writer.close()
        except:
            """"""
        t.cuda.empty_cache()
    try:
        del model, train_dataset, test_dataset, train_loader, test_loader
    except:
        """"""

    gc.collect()
    t.cuda.empty_cache()
