# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import json

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print('ckpt_dir:' + str(ckpt_dir), flush=True)
    print('tokenizer_path:' + str(tokenizer_path), flush=True)

    # load finding section one by one
    test_finding_filepath = './Finding/Test/test.txt'

    # output predictions
    output_file_path = './Finding/Test/predictions_llama_finding.json'

    print('test_finding_filepath:' + str(test_finding_filepath), flush=True)
    print('output_file_path:' + str(output_file_path), flush=True)

    gts = []; texts = []; predictions_llama = []
    with open(test_finding_filepath, 'r') as file:
        for line in file:
            gt, text = line.split('\t')
            gts.append(gt)
            texts.append(text)
            
            dialogs: List[Dialog] = [
                [{"role": "user", "content": "I will provide you with some text, which belongs to the finding section of a radiology report. Can you tell me whether it contains a recommendation for a follow-up? Please respond with exactly one or two words, namely follow-up or no follow-up. The text is the following: " + text}],
            ]
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for dialog, result in zip(dialogs, results):
                for msg in dialog:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n", flush=True)
                predictions_llama.append(result['generation']['content'])

    # Save the list as a JSON file
    with open(output_file_path, 'w') as file:
    	json.dump(predictions_llama, file, indent=4)  # Dump the list into a JSON file with pretty printing

if __name__ == "__main__":
    fire.Fire(main)
