import os
import torch
from time_moe.runner import TimeMoeRunner

def train_time_moe(
    data_path,
    model_path="Maple728/TimeMoE-50M",
    output_path="logs/time_moe",
    max_length=1024,
    stride=None,
    learning_rate=1e-4,
    min_learning_rate=5e-5,
    train_steps=None,
    num_train_epochs=1.0,
    normalization_method="zero",
    seed=9899,
    attn_implementation="auto",
    lr_scheduler_type="cosine",
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.1,
    global_batch_size=64,
    micro_batch_size=16,
    precision="fp32",
    gradient_checkpointing=False,
    deepspeed=None,
    from_scratch=False,
    save_steps=None,
    save_strategy="no",
    save_total_limit=None,
    save_only_model=False,
    logging_steps=1,
    evaluation_strategy="no",
    eval_steps=None,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    dataloader_num_workers=4,
):
    if normalization_method == "none":
        normalization_method = None

    runner = TimeMoeRunner(
        model_path=model_path,
        output_path=output_path,
        seed=seed,
    )

    runner.train_model(
        from_scratch=from_scratch,
        max_length=max_length,
        stride=stride,
        data_path=data_path,
        normalization_method=normalization_method,
        attn_implementation=attn_implementation,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_steps=train_steps,
        num_train_epochs=num_train_epochs,
        precision=precision,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_checkpointing=gradient_checkpointing,
        deepspeed=deepspeed,
        logging_steps=logging_steps,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=dataloader_num_workers,
        save_only_model=save_only_model,
        save_total_limit=save_total_limit,
    )

def obtain_dist_env_dict():
    num_gpus_per_node = os.getenv('LOCAL_WORLD_SIZE') or torch.cuda.device_count() or 1
    num_nodes = os.getenv('WORLD_SIZE') or 1
    rank = os.getenv('RANK') or 0
    master_addr = os.getenv('MASTER_ADDR') or 'localhost'
    master_port = os.getenv('MASTER_PORT') or 9899

    if master_addr is None:
        return None
    else:
        return {
            'master_addr': master_addr,
            'master_port': master_port,
            'world_size': num_nodes,
            'rank': rank,
            'local_world_size': num_gpus_per_node,
        }

def auto_dist_run(main_file: str, argv: str, port: int = 9899):
    os.environ['MASTER_PORT'] = str(port)
    if torch.cuda.is_available():
        env_dict = obtain_dist_env_dict()
        launch_cmd = ' '.join([
            'torchrun',
            f'--master_addr={env_dict["master_addr"]}',
            f'--master_port={env_dict["master_port"]}',
            f'--node_rank={env_dict["rank"]}',
            f'--nproc_per_node={env_dict["local_world_size"]}',
            f'--nnodes={env_dict["world_size"]}',
        ])
        executed_cmd = launch_cmd + f' {main_file} {argv}'
    else:
        executed_cmd = f'python {main_file} {argv}'

    os.system(f'echo "{executed_cmd}"')

    if os.system(executed_cmd) != 0:
        raise RuntimeError(f'Error occurred when execute: {executed_cmd}')
