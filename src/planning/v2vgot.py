"""Invoke original V2V-GoT projector and driving model with isolated ego inputs."""
import copy
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from common import v2v4real_meta as M

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get('TOOLV2X_LLAVA_ROOT', str(ROOT / 'vendor/v2vgot_llava')))
CHECKPOINT = Path(os.environ.get('TOOLV2X_V2VGOT_CHECKPOINT', str(Path(M.V2VGOT_ROOT) /
    'LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2/checkpoint-4330')))
BASE = Path(os.environ.get('TOOLV2X_LLAVA_BASE', str(ROOT / 'models/llava-v1.5-7b')))
CLIP = Path(os.environ.get('TOOLV2X_CLIP_ROOT', str(ROOT / 'models/clip-vit-large-patch14-336')))
# User handles all Hugging Face downloads. Original loaders must stay offline.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.multimodal_projector.builder import build_vision_projector, build_scene_vision_projector
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from planning.inputs import make_prompt, parse_q8, parse_q9, validate_evidence

MODEL_CONFIG = dict(mm_scene_projector_input_size=3072, scene_level_only=False,
    object_level_only=False, scene_feature_mode='shallow', object_feature_mode='shallow',
    num_input_frames=2, ego_only=True, feature_source='no_fusion_keep_all',
    dataset_source='v2v4real', num_latency_frames=0, positional_error_std=0.)
Q8_MAX_NEW_TOKENS = 128
Q9_MAX_NEW_TOKENS = 256


class ContextBudgetError(ValueError):
    def __init__(self, input_tokens, max_new_tokens, context_limit):
        self.budget = dict(input_tokens=input_tokens, max_new_tokens=max_new_tokens, context_limit=context_limit)
        super().__init__('explicit token budget exceeded: %d + %d > %d; refuse upstream silent truncation' %
                         (input_tokens, max_new_tokens, context_limit))


class PointProjector(torch.nn.Module, LlavaMetaForCausalLM):
    """Original projection factories and inherited original token generation."""
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mm_projector = build_vision_projector(config)
        self.mm_scene_projector = build_scene_vision_projector(config)

    def get_model(self):
        return self


def load_projector(checkpoint=CHECKPOINT, device='cpu'):
    checkpoint = Path(checkpoint)
    config = SimpleNamespace(**json.loads((checkpoint / 'config.json').read_text()))
    raw = torch.load(str(checkpoint / 'non_lora_trainables.bin'), map_location='cpu', mmap=True)
    prefix = 'base_model.model.model.'
    if not all(key.startswith(prefix) for key in raw):
        raise ValueError('unexpected original projector state prefix')
    dtype = torch.float32 if str(device) == 'cpu' else torch.float16
    state = {key[len(prefix):]: value.to(device=device, dtype=dtype) for key, value in raw.items()}
    with torch.device('meta'):
        model = PointProjector(config)
    loaded = model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model, dict(source=str(UPSTREAM / 'llava/model/llava_arch.py'),
        checkpoint=str(checkpoint / 'non_lora_trainables.bin'), state_items_loaded=len(state),
        missing_keys=list(loaded.missing_keys), unexpected_keys=list(loaded.unexpected_keys),
        parameter_count=sum(p.numel() for p in model.parameters()), device=str(device),
        dtype=str(dtype), used_projector='mm_projector for both shallow scene and object tokens',
        language_model_executed=False, actual_rgb_input=False)


def model_features(features, device, dtype):
    mask = np.asarray(features['active_agent_mask'])
    if mask.shape != (1, 2, 1) or mask.dtype != bool or not mask[0, 0, 0]:
        raise ValueError('expected one ego source, current frame present')
    valid = mask[0, :, 0]
    expected = {'regression_map': (1, 2, 1, 14, 50, 88),
                'classification_map': (1, 2, 1, 2, 50, 88),
                'detection_box_score': (1, 2, 1, 50, 8)}
    tensors = {}
    for key, shape in expected.items():
        value = np.asarray(features[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid single-ego feature dimensions: ' + key)
        tensors[key] = torch.as_tensor(value[:, valid], device=device, dtype=dtype)
    tensors['active_agent_mask'] = torch.ones((1, int(valid.sum()), 1), dtype=torch.bool, device=device)
    # Shallow upstream code only uses this tensor's shape; no deep feature is read.
    tensors['object_features'] = torch.zeros((1, int(valid.sum()), 1, 50, 256), device=device, dtype=dtype)
    tensors['scene_point_feature_map'] = None
    return tensors


@torch.inference_mode()
def project_ego_features(projector, features):
    param = next(projector.parameters())
    tensors = model_features(features, param.device, param.dtype)
    return projector.generate_point_features(MODEL_CONFIG, **tensors).detach()


def prompt_tokens(tokenizer, prompt):
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + prompt)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')


def rounded(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, list):
        return [rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def fit_evidence(tokenizer, ego_state, evidence, feature_tokens, context_limit=4096, reserve=Q9_MAX_NEW_TOKENS, evidence_format='json'):
    """Explicit same-rule context selection, with the full ledger kept by caller.

    ponytail: current-distance priority is a fixed initial receiver rule; evaluate
    planning-aware priority after the real task chain and adaptation training.
    """
    validate_evidence(evidence)
    view = rounded(copy.deepcopy(evidence))
    objects = sorted(view['objects'], key=lambda obj: (np.linalg.norm(obj['box'][:2]), obj['source'], obj['track_id']))
    view['objects'] = []
    # Association candidates are logged in full; only retained object pairs enter the prompt.
    relations = view.pop('relations', [])
    prototype = 'The suggested speed setting is: very slow. The suggested steering setting is: slightly right.'

    def size(candidate):
        q8 = make_prompt('Q8', ego_state, candidate, evidence_format=evidence_format)
        q9 = make_prompt('Q9', ego_state, candidate, prototype, evidence_format=evidence_format)
        # Reserve a full Q8 generation in addition to the prototype. Decoding and
        # re-tokenization can change length, so _generate still checks the actual prompt.
        return max(len(prompt_tokens(tokenizer, q8)),
                   len(prompt_tokens(tokenizer, q9)) + Q8_MAX_NEW_TOKENS) - 1 + feature_tokens

    if size(view) + reserve > context_limit:
        raise ValueError('task/header alone exceeds context capacity')
    dropped = []
    for obj in objects:
        trial = dict(view, objects=view['objects'] + [obj])
        if size(trial) + reserve <= context_limit:
            view = trial
        else:
            dropped.append({'source': obj['source'], 'track_id': obj['track_id']})
    ego_ids = {obj['track_id'] for obj in view['objects'] if obj['source'] == 'ego'}
    peer_ids = {obj['track_id'] for obj in view['objects'] if obj['source'] != 'ego'}
    for rel in relations:
        if rel['ego_id'] in ego_ids and rel['peer_id'] in peer_ids:
            trial = dict(view, relations=view.get('relations', []) + [rel])
            if size(trial) + reserve <= context_limit:
                view = trial
    return view, dict(context_limit=context_limit, reserved_generation_tokens=reserve,
        reserved_q8_parent_tokens=Q8_MAX_NEW_TOKENS, actual_prompt_budget_check_required=True,
        input_token_bound=size(view), objects_total=len(objects), objects_retained=len(view['objects']),
        dropped_objects=dropped, numeric_decimal_places=2,
        evidence_format=evidence_format,
        selection='ascending current anchor distance; same rule for all actions')


def local_adapter(checkpoint, clip, directory=None):
    """Use local CLIP files without changing the original release directory."""
    checkpoint, clip = Path(checkpoint).expanduser().resolve(), Path(clip).expanduser().resolve()
    missing = [name for name in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin') if not (clip / name).is_file()]
    if missing:
        raise FileNotFoundError('missing user-supplied CLIP files in ' + str(clip) + ': ' + ', '.join(missing))
    adapter = Path(directory).expanduser().resolve() if directory is not None else ROOT / 'models/llava-toolv2x-lora-ego'
    adapter.mkdir(exist_ok=True, parents=True)
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'non_lora_trainables.bin',
                 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json'):
        source, destination = checkpoint / name, adapter / name
        if not source.is_file():
            raise FileNotFoundError(str(source))
        if destination.is_symlink() or destination.exists():
            if destination.resolve() != source.resolve():
                raise FileExistsError('existing adapter points to different weights: ' + str(destination))
        else:
            destination.symlink_to(source)
        if not destination.is_file():
            raise FileNotFoundError('adapter link does not resolve to a file: ' + str(destination))
    config = json.loads((checkpoint / 'config.json').read_text())
    config.update(mm_vision_tower=str(clip), ego_only=True)
    (adapter / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return adapter


class V2VGoTPlanner:
    def __init__(self, checkpoint=CHECKPOINT, base=BASE, clip=CLIP, context_limit=4096,
                 evidence_format='json', adapter_directory=None, trainable_lora=False):
        from llava.model.builder import load_pretrained_model
        checkpoint, base = Path(checkpoint), Path(base)
        index_path = base / 'pytorch_model.bin.index.json'
        if not index_path.is_file():
            raise FileNotFoundError('missing original LLaVA base index: ' + str(index_path))
        index = json.loads(index_path.read_text())
        required = set(index['weight_map'].values()) | {'config.json', 'tokenizer.model'}
        missing = [name for name in sorted(required) if not (base / name).is_file()]
        if missing:
            raise FileNotFoundError('missing base model files: ' + ', '.join(missing))
        if not torch.cuda.is_available():
            raise RuntimeError('original full-model GPU inference unavailable in this execution environment')
        adapted_checkpoint = local_adapter(checkpoint, Path(clip), adapter_directory)
        begin = perf_counter()
        attention = 'flash_attention_2' if trainable_lora else 'sdpa'
        self.tokenizer, self.model, _, _ = load_pretrained_model(str(adapted_checkpoint), str(base),
            'llava-toolv2x-lora', device_map='cuda:0', device='cuda',
            my_model_config=dict(MODEL_CONFIG), attn_implementation=attention, trainable_lora=trainable_lora)
        self.model.eval()
        native_limit = self.model.config.max_position_embeddings
        if not 512 <= context_limit <= native_limit:
            raise ValueError('context limit exceeds original positional capacity')
        self.released_tokenizer_limit = self.model.config.tokenizer_model_max_length
        self.model.config.tokenizer_model_max_length = context_limit
        self.context_limit = context_limit
        if evidence_format not in ('json', 'compact'):
            raise ValueError('unknown evidence serialization')
        self.evidence_format = evidence_format
        self.provenance = dict(checkpoint=str(checkpoint), adapted_checkpoint=str(adapted_checkpoint), base=str(base), clip=str(clip),
            model_class=type(self.model).__name__, source=str(UPSTREAM / 'llava/model/language_model/llava_llama.py'),
            my_model_config=MODEL_CONFIG, loading_seconds=perf_counter() - begin,
            released_tokenizer_limit=self.released_tokenizer_limit, context_limit=context_limit,
            evidence_format=evidence_format,
            adapted_to_tool_evidence=False, training_provenance_verified=False,
            trainable_lora_loaded=trainable_lora,
            attention_implementation=attention,
            actual_rgb_input=False, point_cloud_feature_input=True)
        if (checkpoint / 'training_state.json').is_file():
            state = json.loads((checkpoint / 'training_state.json').read_text())
            self.provenance.update(adapted_to_tool_evidence=state.get('examples_seen', 0) > 0,
                                   local_training_state=state)

    @torch.inference_mode()
    def _generate(self, features, prompt, max_new_tokens):
        device = self.model.device
        tensors = model_features(features, device, self.model.dtype)
        ids = prompt_tokens(self.tokenizer, prompt).unsqueeze(0).to(device)
        count = ids.shape[1] - 1 + int(tensors['active_agent_mask'].sum()) * 270
        if count + max_new_tokens > self.context_limit:
            raise ContextBudgetError(count, max_new_tokens, self.context_limit)
        torch.cuda.synchronize()
        begin = perf_counter()
        output = self.model.generate(ids, images=torch.zeros((1, 3, 336, 336), device=device, dtype=self.model.dtype),
            image_sizes=[(336, 336)], do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            use_cache=True, **tensors)
        torch.cuda.synchronize()
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip(), dict(
            seconds=perf_counter() - begin, input_tokens=count, output_tokens=int(output.shape[1]),
            feature_tokens=int(tensors['active_agent_mask'].sum()) * 270)

    def plan_prepared(self, features, prepared):
        if (prepared.get('input_layout') != 'source_blocks_v1' or prepared.get('decoding') != 'direct' or
                prepared.get('q8_executed') is not False or prepared.get('q8_raw')):
            raise ValueError('expected a source-separated direct input')
        expected = make_prompt('Trajectory', prepared['ego_motion'], prepared['evidence_used'],
                               evidence_format='compact', remote_evidence=prepared.get('remote_evidence_used'))
        selection = prepared['evidence_selection']
        feature_count = int(np.asarray(features['active_agent_mask']).sum()) * 270
        if (expected != prepared['q9_prompt'] or selection['context_limit'] != self.context_limit or
                selection['feature_tokens'] != feature_count):
            raise ValueError('prepared input differs from the model input contract')
        result = dict(prepared, q9_executed=False, language_model_executed=False)
        try:
            raw, cost = self._generate(features, expected, Q9_MAX_NEW_TOKENS)
        except ContextBudgetError as exc:
            result.update(status='q9_context_overflow', error=str(exc), q9_budget=exc.budget)
            return result
        result.update(q9_raw=raw, q9_cost=cost, q9_executed=True, language_model_executed=True)
        try:
            result['waypoints'] = parse_q9(raw).tolist()
            result['status'] = 'parsed'
        except ValueError as exc:
            result.update(status='invalid_q9', error=str(exc))
        return result

    def plan(self, features, ego_state, evidence, evidence_format=None, decoding='q8_q9', input_layout='legacy'):
        if input_layout == 'source_blocks_v1':
            if decoding != 'direct':
                raise ValueError('source-separated execution currently uses the direct trajectory task')
            from planning.context import build_plan_input
            count = int(np.asarray(features['active_agent_mask']).sum()) * 270
            return self.plan_prepared(features, build_plan_input(self.tokenizer, ego_state, evidence,
                                                                count, self.context_limit))
        if input_layout != 'legacy':
            raise ValueError('unknown input layout')
        if decoding not in ('q8_q9', 'direct'):
            raise ValueError('unknown driving decoding mode')
        evidence_format = self.evidence_format if evidence_format is None else evidence_format
        feature_count = int(np.asarray(features['active_agent_mask']).sum()) * 270
        view, selection = fit_evidence(self.tokenizer, ego_state, evidence, feature_count, self.context_limit,
                                      evidence_format=evidence_format)
        result = dict(decoding=decoding, q8_executed=False, q8_prompt=None, q8_raw=None, q8_cost=None,
                      evidence_selection=selection, evidence_used=view, q9_executed=False,
                      language_model_executed=False)
        if decoding == 'q8_q9':
            q8 = make_prompt('Q8', ego_state, view, evidence_format=evidence_format)
            raw8, cost8 = self._generate(features, q8, Q8_MAX_NEW_TOKENS)
            result.update(q8_prompt=q8, q8_raw=raw8, q8_cost=cost8, q8_executed=True,
                          language_model_executed=True)
            try:
                result['action'] = parse_q8(raw8)
            except ValueError as exc:
                result.update(status='invalid_q8', error=str(exc))
                return result
            q9 = make_prompt('Q9', ego_state, view, raw8, evidence_format=evidence_format)
        else:
            q9 = make_prompt('Trajectory', ego_state, view, evidence_format=evidence_format)
        result['q9_prompt'] = q9
        try:
            raw9, cost9 = self._generate(features, q9, Q9_MAX_NEW_TOKENS)
        except ContextBudgetError as exc:
            result.update(status='q9_context_overflow', error=str(exc), q9_budget=exc.budget)
            return result
        result.update(q9_raw=raw9, q9_cost=cost9, q9_executed=True, language_model_executed=True)
        try:
            result['waypoints'] = parse_q9(raw9).tolist()
            result['status'] = 'parsed'
        except ValueError as exc:
            result.update(status='invalid_q9', error=str(exc))
        return result
