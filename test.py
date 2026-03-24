from trl import SFTConfig, SFTTrainer

try:
    sft_config = SFTConfig(
        output_dir="red_agent_lora_cpu",
        num_train_epochs=1,
        max_steps=15, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        logging_steps=5,
        save_strategy="no",
        dataset_text_field="text",
        max_length=256,
        report_to="none"
    )
except Exception as e:
    with open("err.txt", "w") as f:
        f.write(str(e))
