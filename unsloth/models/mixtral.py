# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .mistral import *
from ._utils import __version__
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralSparseMoeBlock,
    MixtralModel,
    MixtralForCausalLM,
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
    load_balancing_loss_func,
)

# For Pytorch 2.1.1
try:
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralSdpaAttention,
        MixtralFlashAttention2,
    )
except:
    MixtralSdpaAttention = MixtralAttention
    MixtralFlashAttention2 = MixtralAttention
pass


def MixtralDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    *args,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = fast_rms_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = fast_rms_layernorm(hidden_states)
    hidden_states, router_logits = self.block_sparse_moe(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits,)

    return outputs


pass


def MixtralModel_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, MoeModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    past_key_values_length = 0

    # Fix out of bounds tokenization
    if hasattr(self, "max_seq_length"):
        if seq_length > self.max_seq_length:
            logger.warning_once(
                f"Unsloth: Input IDs of length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:, : self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:, : self.max_seq_length, :]
        pass
    pass

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if use_cache:
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if (
        attention_mask is not None
        and self._attn_implementation == "flash_attention_2"
        and use_cache
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask_for_sdpa,
        )

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
        )

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attns,
                all_router_logits,
            ]
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )


pass


def MixtralForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, MoeCausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )

    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = fast_cross_entropy_loss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits if return_dict else outputs[-1],
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(
                loss.device
            )  # make sure to reside in the same device

    if not return_dict:
        output = (logits,) + outputs[1:]
        if output_router_logits:
            output = (aux_loss,) + output
        return (loss,) + output if loss is not None else output

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


pass


class FastMixtralModel:
    @staticmethod
    def pre_patch():
        MixtralAttention.forward = MistralAttention_fast_forward
        MixtralSdpaAttention.forward = MistralAttention_fast_forward
        MixtralFlashAttention2.forward = MistralAttention_fast_forward
        MixtralDecoderLayer.forward = MixtralDecoderLayer_fast_forward
        MixtralModel.forward = MixtralModel_fast_forward
        MixtralForCausalLM.forward = MixtralForCausalLM_fast_forward
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        return

    pass

    @staticmethod
    def from_pretrained(
        model_name="ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        token=None,
        device_map="sequential",
        rope_scaling=None,  # Mixtral does not support RoPE scaling
        fix_tokenizer=True,
        **kwargs,
    ):
        # Mixtral does NOT support RoPE Scaling!
        if rope_scaling is not None:
            logger.warning_once("Unsloth: Mixtral models do not support RoPE scaling.")
        pass

        SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = (
            f"==((====))==  Unsloth: Fast Mixtral patching release {__version__}\n"
            f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform = {platform_system}.\n"
            f"O^O/ \_/ \\    Pytorch: {torch.__version__}. CUDA = {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit = {torch.version.cuda}.\n"
            f"\        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. Xformers = {xformers_version}. FA = {HAS_FLASH_ATTENTION}.\n"
            f' "-____-"     Apache 2 free license: http://github.com/unslothai/unsloth'
        )
        print(statistics)
        FastMixtralModel.pre_patch()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once(
                "Device does not support bfloat16. Will change to float16."
            )
            dtype = torch.float16

        assert (
            dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32
        )

        # Check max sequence length
        model_config = AutoConfig.from_pretrained(model_name, token=token)
        model_max_seq_length = model_config.max_position_embeddings

        # Mixtral does NOT support RoPE Scaling sadly so we have to error out.
        if max_seq_length > model_max_seq_length:
            raise RuntimeError(
                "Unsloth: Unfortunately Mixtral type models do not support RoPE scaling!\n"
                f"The maximum sequence length supported is {model_max_seq_length}.",
            )
        pass

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )

        max_position_embeddings = max(max_seq_length, model_max_seq_length)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=bnb_config,
            token=token,
            # rope_scaling      = rope_scaling,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=max_position_embeddings,
            padding_side="right",
            token=token,
        )

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = FastMixtralModel.post_patch(model)

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o = original_apply_o
        pass

        # Save max_seq_length
        max_position_embeddings = max(
            max_seq_length, model.config.max_position_embeddings
        )
        model.max_seq_length = max_position_embeddings
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_position_embeddings
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_position_embeddings

        # We check the tokenizer first for errors
        if fix_tokenizer:
            tokenizer = check_tokenizer(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                model_max_length=max_position_embeddings,
                padding_side="right",
                token=token,
            )
        pass
        patch_saving_functions(tokenizer)

        # Fix up config for transformers uploading PEFT
        # Not necessary anymore since we require transformers>=4.37
        if False:
            name = model.config._name_or_path
            if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                name = name[: len(name) - len("-bnb-4bit")]
                model.config.update({"_name_or_path": name})
            pass

        # Log Unsloth version for future fastpaths for inference
        model.config.update({"unsloth_version": __version__})

        # Add save modules
        patch_saving_functions(model)

        return model, tokenizer

    pass

    @staticmethod
    def post_patch(model):
        # Patch model
        layers = model.model.layers

        # Torch.compile fails on embedding matrix??
        # Workaround randomnly fixes it for torch versions < 2.2
        model.model.embed_tokens = torch.nn.Embedding.from_pretrained(
            model.model.embed_tokens.weight
        )
        model.config.update({"unsloth_version": __version__})

        # We also do this for the lm_head
        lm_head = torch.nn.Linear(1, 1, bias=None)
        del lm_head.weight
        lm_head.weight = model.lm_head.weight
        lm_head.in_features = lm_head.weight.shape[1]
        lm_head.out_features = lm_head.weight.shape[0]
        model.lm_head = lm_head

        # Also patch all dtypes - BnB seems to not allocate the correct type?
        # BnB default dtype seems to be float16!
        correct_dtype = lm_head.weight.dtype

        for name, module in model.named_modules():
            if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
                weight = module.weight
                quant_state = weight.quant_state

                if type(quant_state) is list:
                    # BnB seems to have float16 as default!
                    module.weight.quant_state[2] = (
                        correct_dtype  # Cast to correct dtype
                    )
                else:
                    # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                    quant_state.dtype = correct_dtype
                pass
            pass
        pass

        # Clear deleted GPU items
        import gc

        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model

    pass

    @staticmethod
    def get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate",
            "w1",
            "w2",
            "w3",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        layers_to_transform=None,
        layers_pattern=None,
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=2048,  # not used anymore
        use_rslora=False,
        init_lora_weights=True,
        loftq_config={},
        **kwargs,
    ):
        transformers_set_seed(random_state)

        if isinstance(model, PeftModelForCausalLM):
            raise TypeError(
                "Unsloth: Your model already has LoRA adapters. No need to run this again!"
            )
        pass

        import inspect

        signature = str(inspect.signature(LoraConfig))
        SUPPORTS_LOFTQ = "loftq_config" in signature
        SUPPORTS_RSLORA = "use_rslora" in signature

        assert max_seq_length <= model.max_seq_length

        if lora_dropout != 0:
            logger.warning_once(
                f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if bias != "none":
            logger.warning_once(
                f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if not (
            type(init_lora_weights) is bool
            or init_lora_weights == "gaussian"
            or init_lora_weights == "loftq"
        ):
            raise ValueError(
                'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq"].'
            )
        pass

        if init_lora_weights == "loftq":

            if not SUPPORTS_LOFTQ:
                import peft

                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"
                    "Please install PEFT 0.7.2 or higher.\n"
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass

            if loftq_config == {}:
                from peft import LoftQConfig

                logger.warning_once(
                    f"Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"
                    f"We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
                )
                loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=1)
            pass

            if hasattr(model.config, "quantization_config"):
                raise ValueError(
                    "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"
                    "Reload your model without any quantization by setting `load_in_4bit = False`."
                )
            pass
        pass

        assert type(use_rslora) is bool
        if use_rslora:
            if not SUPPORTS_RSLORA:
                # We manually check for PEFT
                import peft

                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support `use_rslora`.\n"
                    "Please install PEFT 0.7.2 or higher.\n"
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass
        pass

        accepted_modules = frozenset(
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "w1", "w2", "w3"),
        )
        model.config.update({"unsloth_version": __version__})
        for module in target_modules:
            assert module in accepted_modules
        pass

        # Get LoRA
        arguments = dict(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
            layers_to_transform=layers_to_transform,
            init_lora_weights=init_lora_weights,
            loftq_config=loftq_config,
            use_rslora=use_rslora,
            **kwargs,
        )
        if not SUPPORTS_LOFTQ:
            del arguments["loftq_config"]
        if not SUPPORTS_RSLORA:
            del arguments["use_rslora"]

        lora_config = LoraConfig(**arguments)
        model = _get_peft_model(model, lora_config)

        model = FastMixtralModel.patch_peft_model(model, use_gradient_checkpointing)
        return model

    pass

    @staticmethod
    def patch_peft_model(
        model,
        use_gradient_checkpointing=True,
    ):
        if not isinstance(model, PeftModelForCausalLM):
            raise TypeError(
                "Unsloth: Your model needs to call `.get_peft_model` first!"
            )
        pass

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_reentrant=True,
        )

        # Fix up config for transformers uploading PEFT
        for active_adapter in model.peft_config.keys():
            # Not necessary since we requires transformers >= 4.37
            if False:
                name = model.peft_config[active_adapter].base_model_name_or_path
                if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                    name = name[: len(name) - len("-bnb-4bit")]
                    model.peft_config[active_adapter].base_model_name_or_path = name
                pass
            # Add revision to enable future fast inference paths
            model.peft_config[active_adapter].revision = f"unsloth"
        pass

        # Do patching
        n_mlp = 0
        n_qkv = 0
        n_o = 0
        import types

        active_adapter = (
            model.active_adapters[0]
            if hasattr(model, "active_adapters")
            else model.active_adapter
        )

        # Get dropout and bias
        lora_dropout = model.peft_config[active_adapter].lora_dropout
        bias = model.peft_config[active_adapter].bias

        if lora_dropout == 0 and bias == "none":
            for idx, layer in enumerate(model.model.model.layers):

                # # MLP patching
                # gate_proj = layer.mlp.gate_proj
                # up_proj   = layer.mlp.  up_proj
                # down_proj = layer.mlp.down_proj

                # if  hasattr(gate_proj, "lora_A") and \
                #     hasattr(  up_proj, "lora_A") and \
                #     hasattr(down_proj, "lora_A") and \
                #     (gate_proj.base_layer if hasattr(gate_proj, "base_layer") else gate_proj).bias is None and \
                #     (  up_proj.base_layer if hasattr(  up_proj, "base_layer") else   up_proj).bias is None and \
                #     (down_proj.base_layer if hasattr(down_proj, "base_layer") else down_proj).bias is None:

                #     # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
                #     layer.mlp.forward = types.MethodType(apply_lora_mlp, layer.mlp)
                #     n_mlp += 1
                # else:
                #     logger.warning_once(
                #         "Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters\n"\
                #         "are not enabled or a bias term (like in Qwen) is used."
                #     )
                # pass

                # QKV attention patching
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                if (
                    hasattr(q_proj, "lora_A")
                    and hasattr(k_proj, "lora_A")
                    and hasattr(v_proj, "lora_A")
                    and (
                        q_proj.base_layer if hasattr(q_proj, "base_layer") else q_proj
                    ).bias
                    is None
                    and (
                        k_proj.base_layer if hasattr(k_proj, "base_layer") else k_proj
                    ).bias
                    is None
                    and (
                        v_proj.base_layer if hasattr(v_proj, "base_layer") else v_proj
                    ).bias
                    is None
                ):

                    layer.self_attn.apply_qkv = apply_lora_qkv
                    n_qkv += 1
                else:
                    logger.warning_once(
                        "Unsloth cannot patch Attention layers with our manual autograd engine since either LoRA adapters\n"
                        "are not enabled or a bias term (like in Qwen) is used."
                    )
                pass

                # O attention patching
                o_proj = layer.self_attn.o_proj
                if (
                    hasattr(o_proj, "lora_A")
                    and (
                        o_proj.base_layer if hasattr(o_proj, "base_layer") else o_proj
                    ).bias
                    is None
                ):

                    layer.self_attn.apply_o = apply_lora_o
                    n_o += 1
                else:
                    logger.warning_once(
                        "Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters\n"
                        "are not enabled or a bias term (like in Qwen) is used."
                    )
                pass
            pass
        pass

        logger.warning_once(
            f"Unsloth {__version__} patched {len(model.model.model.layers)} layers with "
            f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
        )
        patch_saving_functions(model)

        # Patch cross entropy loss labels
        # Fixes https://github.com/unslothai/unsloth/issues/10
        max_seq_length = model.max_seq_length
        extra_ignored_labels = torch.full((max_seq_length, 1), -100, device="cuda")
        model.model.extra_ignored_labels = extra_ignored_labels
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_seq_length
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_seq_length
        return model

    pass

    @staticmethod
    def for_inference(model):
        if not hasattr(model, "_original_forward"):
            model._original_forward = model.forward
        pass
        model.forward = torch.inference_mode(model._original_forward)

        internal_model = model
        internal_model.gradient_checkpointing = False
        internal_model.training = False

        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
            internal_model.gradient_checkpointing = False
            internal_model.training = False
        pass

    pass

    @staticmethod
    def for_training(model, use_gradient_checkpointing=True):
        if hasattr(model, "_original_forward"):
            model.forward = model._original_forward
        pass

        internal_model = model
        internal_model.gradient_checkpointing = use_gradient_checkpointing
        internal_model.training = True

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora
        pass

        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
            internal_model.gradient_checkpointing = use_gradient_checkpointing
            internal_model.training = True
        pass

    pass


pass
