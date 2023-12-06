# Efficient-Fine-tuning-of-LLM-on-a-Single-GPU

Use four fine-tuning methods(low precision, gradient accumulation, gradient checkpointing, LoRA) 
to reduce memory usage when training LLAMA-2-7B.

1. Implement LoRA Linear Module
  Convert the model into PEFT model by replacing Q, V projection layers in
  LLaMA model with LoRA Linear modules.Freeze model parameters except LoRA_A
  and LoRA_B parameters.

2. Implement AMP
  Use Pytorch’s AMP API.Modify the training loop to enable FP16 mixed precision training.

3. Implement gradient accumulation
  Set the batch size to 1 and the gradient accumulation step to 8.

4. Implement gradient checkpointing
   Use Pytorch’s checkpoint API.

5. Run training on Alpaca Dataset
  Extract the first 200 samples from the entire 52000 samples.

After applying all techniques above, you should be able to reduce the memory
usage below 40GB.
