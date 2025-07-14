import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm


def main(model: str, checkpoint: int):
    # Load pretrained model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"GT4SD/multitask-text-and-chemistry-t5-{model}-augm"
    ckp = f"../{model}_ft_checkpoints/checkpoint_epoch_{checkpoint}.pt"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device)
    # Load checkpoint
    model.load_state_dict(torch.load(ckp, map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset_name = "language-plus-molecules/LPM-24_train"
    dataset = load_dataset(dataset_name)
    dataset = dataset["split_valid"].shuffle(seed=42)
    #dataset = dataset["split_valid"].shuffle(seed=42).select(range(300))


    # Define function to generate captions for a batch of molecules
    def generate_captions_batch(molecules):
        
        inputs = tokenizer(["Caption the following SMILES: " + text for text in molecules], return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        #print("inputs_ids:", inputs)
        outputs = model.generate(**inputs, 
                                max_length=128, 
                                num_beams=5, 
                                do_sample=True, temperature=0.7, top_p=0.9, 
                                repetition_penalty=1.4,
                                )
        #print("outputs_ids:", outputs)
        captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print("captions:", captions)

        # Process each caption in the batch
        processed_captions = []
        for caption in captions:
            caption = caption.split(tokenizer.eos_token)[0]  # Remove everything after [EOS]
            caption = caption.replace(tokenizer.pad_token, "")  # Remove padding tokens
            caption = caption.strip()  # Remove leading and trailing whitespace
            processed_captions.append(caption)
        
        return processed_captions
        #return captions

    # Generate captions for molecules in batches
    batch_size = 128
    generated_captions = []
    for start_index in tqdm(range(0, len(dataset), batch_size)):
        molecules_batch = dataset[start_index:start_index+batch_size]["molecule"]
        ground_truth_batch = dataset[start_index:start_index+batch_size]["caption"]
        captions_batch = generate_captions_batch(molecules_batch)

        # Append the tab-separated molecule and generated captions to the list
        generated_captions.extend(zip(molecules_batch, ground_truth_batch, captions_batch))

    # Save generated captions to a .txt file
    output_file = f"captions_sample_{checkpoint}.txt"
    with open(output_file, "w") as f:
        for caption in generated_captions:
            f.write("\t".join(caption) + "\n")

    print(f"Generated captions of checkpoint {checkpoint} saved to", output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate captions for molecules')
    parser.add_argument('--model', type=str, help='model to load')
    parser.add_argument('--checkpoint', type=int, help='checkpoint to load')
    args = parser.parse_args()
    main(args.model, args.checkpoint)
