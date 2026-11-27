"""Load the local LIGER baseline using the original upstream TIGER class."""
import argparse
import sys
from pathlib import Path

import torch
from transformers import T5Config


def load_checkpoint(upstream, checkpoint=None):
    sys.path.insert(0, str(Path(upstream).resolve()))
    from src.tiger import TIGER
    checkpoint = checkpoint or Path(__file__).with_name('best.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    dataset = saved['manifest']['dataset']
    method = saved['manifest']['method']
    t = dataset['TIGER']['T5']
    vocab = saved['model']['shared.weight'].shape[0]
    config = T5Config(num_layers=t['encoder_layers'], num_decoder_layers=t['decoder_layers'],
        d_model=t['d_model'], d_ff=t['d_ff'], num_heads=t['num_heads'], d_kv=t['d_kv'],
        dropout_rate=t['dropout_rate'], vocab_size=vocab, pad_token_id=0,
        decoder_start_token_id=0, eos_token_id=vocab-1,
        n_positions=dataset['TIGER']['n_positions'], feed_forward_proj=t['feed_forward_proj'],
        layer_norm_epsilon=1e-8, initializer_factor=t['initializer_factor'])
    model = TIGER(config, 3, dataset['max_items_per_seq'],
        flag_use_output_embedding=method['flag_use_output_embedding'],
        flag_use_learnable_text_embed=method['flag_add_input_embedding'],
        embedding_head_dict=method['embedding_head_dict'])
    model.load_state_dict(saved['model'], strict=True)
    return model.eval(), saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', required=True, type=Path)
    args = parser.parse_args()
    model, saved = load_checkpoint(args.upstream)
    print({'parameters': sum(p.numel() for p in model.parameters()),
           'step': saved['step'], 'epoch': saved['epoch'], 'strict_load': 'passed'})
