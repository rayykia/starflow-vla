"""SmolVLM condition encoder for STARFlow-VLA.

Replaces the frozen flan-T5 text encoder: the task instruction *and* the current
observation go through SmolVLM (SigLIP vision tower -> pixel-shuffle connector ->
Llama text model, as in SimVLANF/models/modeling_smolvlm_vla.py::forward_vlm_efficient)
and the last hidden state of every token is returned as the conditioning sequence.

The token layout reproduces SmolVLM's own chat template for one un-split image,
so a frozen backbone sees in-distribution inputs:

    <|im_start|>User:<fake_token_around_image><global-img> [n_img x image feats]
    <fake_token_around_image>{instruction}<end_of_utterance>\nAssistant:

Images are given as (B, 3, h, w) tensors in [-1, 1] (the dataset / VAE convention);
they are resized to `image_size` on the GPU and normalized with the processor's
mean/std (0.5/0.5 for SmolVLM, i.e. the identity on [-1, 1]).
"""
import warnings

import torch
import torch.nn.functional as F


class SmolVLMEncoder(torch.nn.Module):
    """(images, instructions) -> (features (B, L, D), mask (B, L)); D = text hidden size."""

    def __init__(self, model_path='HuggingFaceTB/SmolVLM-500M-Instruct', image_size=512,
                 max_text_tokens=40, freeze=False):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.model_path = model_path
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        self.frozen = bool(freeze)

        # AutoModel (not ...ForImageTextToText): no lm_head, so every parameter is used
        # in forward and DDP does not need find_unused_parameters
        self.model = AutoModel.from_pretrained(model_path, dtype=torch.float32)
        for name in ('vision_model', 'connector', 'text_model'):
            assert hasattr(self.model, name), f'{model_path}: expected an Idefics3/SmolVLM model with .{name}'
        self.hidden_size = self.model.config.text_config.hidden_size

        processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = processor.tokenizer
        ip = processor.image_processor
        self.register_buffer('image_mean', torch.tensor(ip.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('image_std', torch.tensor(ip.image_std).view(1, 3, 1, 1), persistent=False)

        # fixed prompt pieces around the image features / the instruction
        fake = getattr(processor, 'fake_image_token', '<fake_token_around_image>')
        glob = getattr(processor, 'global_image_token', None) or '<global-img>'
        eou = getattr(processor, 'end_of_utterance_token', '<end_of_utterance>')
        bos = self.tokenizer.bos_token or '<|im_start|>'
        self.prefix_text = f'{bos}User:{fake}{glob}'
        self.suffix_template = f'{fake}{{instruction}}{eou}\nAssistant:'
        prefix_ids = self.tokenizer(self.prefix_text, add_special_tokens=False, return_tensors='pt')['input_ids']
        self.register_buffer('prefix_ids', prefix_ids, persistent=False)  # (1, n_prefix)
        self._warned_truncation = False

        if self.frozen:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()  # a frozen backbone stays in eval mode
        return self

    # ------------------------------------------------------------------ pieces
    def preprocess_images(self, images):
        """(B, 3, h, w) in [-1, 1] -> (B, 3, S, S) normalized for the vision tower."""
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(images.float(), size=(self.image_size, self.image_size),
                                   mode='bicubic', align_corners=False).clamp(-1, 1)
        x01 = (images.float() + 1) / 2
        return (x01 - self.image_mean) / self.image_std

    def encode_images(self, images):
        """-> image features in the text-model space: (B, n_img, D)."""
        pixel_values = self.preprocess_images(images)
        vision_out = self.model.vision_model(pixel_values=pixel_values).last_hidden_state
        return self.model.connector(vision_out)

    def tokenize_instructions(self, instructions, device):
        """-> (ids (B, txt), mask (B, txt)) for the instruction suffix, right-padded."""
        texts = [self.suffix_template.format(instruction=s) for s in instructions]
        tok = self.tokenizer(texts, padding='max_length', max_length=self.max_text_tokens,
                             truncation=True, add_special_tokens=False, return_tensors='pt')
        mask = tok['attention_mask']
        if not self._warned_truncation and bool(mask.all(dim=1).any()):
            warnings.warn(f'SmolVLMEncoder: an instruction filled all {self.max_text_tokens} text '
                          f'slots and may have been truncated; raise txt_size', stacklevel=2)
            self._warned_truncation = True
        return tok['input_ids'].to(device), mask.to(device)

    # ----------------------------------------------------------------- forward
    def forward(self, images, instructions):
        """images (B, 3, h, w) in [-1, 1]; instructions: B strings.

        Returns (features (B, L, D), mask (B, L)) with padded positions zeroed;
        L = n_prefix + n_img + max_text_tokens.
        """
        assert images.size(0) == len(instructions), 'one instruction per image'
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            B, device = images.size(0), images.device
            embed = self.model.text_model.get_input_embeddings()
            img_feats = self.encode_images(images)                                  # (B, n_img, D)
            suffix_ids, suffix_mask = self.tokenize_instructions(instructions, device)
            prefix = embed(self.prefix_ids.to(device)).expand(B, -1, -1)              # (B, n_prefix, D)
            suffix = embed(suffix_ids)                                               # (B, txt, D)
            inputs_embeds = torch.cat([prefix, img_feats.to(prefix.dtype), suffix], dim=1)
            mask = torch.cat([torch.ones(B, prefix.size(1) + img_feats.size(1),
                                         dtype=suffix_mask.dtype, device=device), suffix_mask], dim=1)
            hidden = self.model.text_model(inputs_embeds=inputs_embeds, attention_mask=mask,
                                           use_cache=False).last_hidden_state
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)                     # zero the padding
        return hidden, mask.bool()

    def extra_repr(self):
        return (f'model_path={self.model_path}, image_size={self.image_size}, '
                f'max_text_tokens={self.max_text_tokens}, frozen={self.frozen}, hidden_size={self.hidden_size}')
