import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import Adafactor, AdafactorSchedule

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "147.47.68.91"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
    
    def _run_batch(self, batch):
        self.optimizer.zero_grad()

        # Convert input_ids, attention_mask, and labels to tensors
        input_ids = torch.stack(batch["input_ids"], dim=1).to(self.gpu_id)  # Prompt + SMILES
        attention_mask = torch.stack(batch["attention_mask"], dim=1).to(self.gpu_id)
        labels = torch.stack(batch["labels"], dim=1).to(self.gpu_id)

        # Forward pass
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask, 
                             labels=labels)
        loss = outputs.loss
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        self.model.train()
        b_sz = len(next(iter(self.train_data))["input_ids"])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        total_loss = 0
        for batch in self.train_data:
            loss = self._run_batch(batch)
            total_loss += loss
        avg_loss = total_loss / len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Avg Loss: {avg_loss}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"./base_ft_checkpoints/checkpoint_epoch_{epoch+5}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _validate(self, epoch):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_data:
                # Convert input_ids, attention_mask, and labels to tensors
                input_ids = torch.stack(batch["input_ids"], dim=1).to(self.gpu_id)  # Prompt + SMILES
                attention_mask = torch.stack(batch["attention_mask"], dim=1).to(self.gpu_id)
                labels = torch.stack(batch["labels"], dim=1).to(self.gpu_id)
                # Forward pass
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask, 
                                     labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_data)
        print(f"[GPU{self.gpu_id}] Validation | Epoch {epoch} | Avg Loss: {avg_loss}")
    


    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            # Additionally, save checkpoint at the last epoch
            if self.gpu_id == 0 and epoch == max_epochs - 1:
                self._save_checkpoint(epoch)
            if self.val_data is not None:
                self._validate(epoch)


def load_model_and_data(rank):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from datasets import load_dataset


    tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
    model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
    model.load_state_dict(torch.load("./base_ft_checkpoints/checkpoint_epoch_4.pt", map_location={'cuda:0': f'cuda:{rank}'}))


    # Load and preprocess the dataset
    dataset = load_dataset("language-plus-molecules/LPM-24_train")

    # Define maximum lengths
    max_input_length = 128  # Adjust this value
    max_target_length = 128  # Adjust this value 

    def tokenize_function(examples):
        # Add prompt
        prompt = "Caption the following SMILES: "
        examples['molecule'] = [prompt + molecule for molecule in examples['molecule']]
        
        # Tokenize the input text
        model_inputs = tokenizer(examples["molecule"], 
                                 max_length=max_input_length,
                                 padding="max_length", 
                                 truncation=True)
        
        # Tokenize the target text
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["caption"],
                               max_length=max_target_length, 
                               padding="max_length", 
                               truncation=True)
    
        # Add labels to the model inputs
        model_inputs["labels"] = labels["input_ids"]

        # Replace padding token id's of the labels by -100
        model_inputs["labels"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in label_ids] 
            for label_ids in model_inputs["labels"]
        ]
        
        return model_inputs
    

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    ### Split train dataset into train and validation (85:15)
    dataset = tokenized_datasets["split_train"].train_test_split(test_size=0.15).shuffle(seed=42)

    """ # Only use 15000samples for training and 1000 for validation
    train_dataset = dataset["train"].select(range(15000))
    val_dataset = dataset["test"].select(range(1000)) """

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    return model, train_dataset, val_dataset


def load_train_objs(rank):
    model, train_dataset, val_dataset = load_model_and_data(rank)
    train_set = train_dataset  # load your dataset
    val_set = val_dataset
    model = model  # load your model
    #optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01) # load your optimizer - lr=4e-3 for small, 6e-3 for base
    optimizer = Adafactor(model.parameters(),
                          scale_parameter=False, 
                          relative_step=False, 
                          warmup_init=False, 
                          lr=2e-5, 
                          weight_decay=0.01) # load your optimizer - lr=4e-3 for small, 6e-3 for base
    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_dataset, val_dataset, model, optimizer = load_train_objs(rank)

    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)

    trainer = Trainer(model, train_data, val_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)