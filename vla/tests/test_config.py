from vla.train_libero import load_vla_config


def test_load_vla_config():
    args = load_vla_config('configs/starflow_vla_libero_128.yaml')
    assert args.action_horizon % 4 == 0
    assert len(args.layers_per_block) % 2 == 0, 'even blocks so top gets PermutationIdentity'
    assert args.sos == 1 and args.seq_order == 'L2R'
    assert args.channel_size == 2 * 48 and args.img_size == 128   # two views x Wan z_dim
    assert args.action_dim == 7
    # SmolVLM condition encoder + proprio token replace the T5 text encoder
    assert args.vlm.startswith('HuggingFaceTB/SmolVLM')
    assert args.proprio_dim == 8 and args.vlm_image_size % 16 == 0
    assert args.txt_size >= 24, 'instruction suffix needs room for ~6 template tokens'
    assert args.vlm_lr <= args.lr
