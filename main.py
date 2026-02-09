import os
import torch.multiprocessing as mp
import parser_config

if __name__ == '__main__':
    global args
    args = parser_config.get_args()
    world_size = args.world_size

    if args.task == 'llm':
        from train_DiSI_llm_memorization import activate
    else:
        raise ValueError('Task not defined.')
    
    mp.spawn(activate, args=(args,), nprocs=args.gpu_per_node*args.process_per_gpu, join=True)

    print('done.\n\n\n')
