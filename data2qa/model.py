import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, \
    HubertModel, UniSpeechSatModel, Wav2Vec2Model, WavLMModel
from transformers import ViTFeatureExtractor, ViTModel, Wav2Vec2FeatureExtractor


def handle_decoder_input_none(decoder_config, batch=1, device='cpu'):
    return torch.tensor([[decoder_config.decoder_start_token_id]] * batch).to(device)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class Data2QA(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, cv_model_config='google/vit-base-patch16-224-in21k',
                 share_layer_ratio=0.3,
                 down_scale_speech=8,
                 down_scale_nlp=4,
                 down_scale_cv=8,
                 weighted_sum=False,
                 fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], **kwargs):
        super(Data2QA, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'hubert' in speech_model_config:
            self.speech_model = HubertModel
        elif 'unispeech' in speech_model_config:
            self.speech_model = UniSpeechSatModel
        elif 'wavlm' in speech_model_config:
            self.speech_model = WavLMModel
        else:
            self.speech_model = Wav2Vec2Model
        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config).to(self.device)
        self.nlp_encoder_model = self.seq2seq_model.base_model.encoder.to(self.device)
        self.decoder_model = self.seq2seq_model.base_model.decoder.to(self.device)
        self.speech_encoder_model = self.speech_model.from_pretrained(speech_model_config).to(self.device)
        self.cv_encoder_model = ViTModel.from_pretrained(cv_model_config).to(self.device)
        self.speech_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(speech_model_config)
        self.cv_feature_extractor = ViTFeatureExtractor.from_pretrained(cv_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        self.weighted_sum = weighted_sum
        num_nlp_encoder_layers = 0
        if hasattr(self.seq2seq_model.base_model.encoder, 'layers'):
            num_nlp_encoder_layers = len(self.seq2seq_model.base_model.encoder.layers)
        elif hasattr(self.seq2seq_model.base_model.encoder, 'block'):
            num_nlp_encoder_layers = len(self.seq2seq_model.base_model.encoder.block)
        print("Before layer sharing num_speech_encoder_layers", len(self.speech_encoder_model.encoder.layers))
        print("Before layer sharing num_cv_encoder_layers", len(self.cv_encoder_model.encoder.layer))
        remove_layers_speech = int(
            len(self.speech_encoder_model.encoder.layers) * share_layer_ratio) if share_layer_ratio != 0 else 0
        self.speech_encoder_model.encoder.layers = self.speech_encoder_model.encoder.layers[
                                                   :len(
                                                       self.speech_encoder_model.encoder.layers) - remove_layers_speech]
        remove_layers_cv = int(
            len(self.cv_encoder_model.encoder.layer) * share_layer_ratio) if share_layer_ratio != 0 else 0
        self.cv_encoder_model.encoder.layers = self.cv_encoder_model.encoder.layer[
                                               :len(self.cv_encoder_model.encoder.layer) - remove_layers_cv]
        self.num_speech_encoder_layers = len(self.speech_encoder_model.encoder.layers)
        self.num_cv_encoder_layers = len(self.cv_encoder_model.encoder.layer)
        print("After layer sharing ",
              "num_speech_encoder_layers", self.num_speech_encoder_layers,
              "num_cv_encoder_layers", self.num_cv_encoder_layers,
              "num_nlp_encoder_layers", num_nlp_encoder_layers,
              "share_layer_ratio", share_layer_ratio,
              "remove_layers_cv", remove_layers_cv,
              "remove_layers_speech", remove_layers_speech)

        # length downscale - nlp
        self.downsize_nlp = down_scale_nlp
        self.downloop_nlp = int(math.log(self.downsize_nlp, 2))
        if self.downsize_nlp > 1:
            self.length_adapters_nlp = nn.Sequential(
                *[nn.Conv1d(in_channels=self.nlp_encoder_model.config.hidden_size,
                            out_channels=self.nlp_encoder_model.config.hidden_size,
                            kernel_size=2,
                            stride=2) for _ in range(self.downloop_nlp)]).to(self.device)
        else:
            self.length_adapters_nlp = nn.Sequential(nn.Identity()).to(self.device)

        # length downscale - speech
        self.downsize_speech = down_scale_speech
        self.downloop_speech = int(math.log(self.downsize_speech, 2))
        if self.downsize_speech > 1:
            self.length_adapters_speech = nn.Sequential(
                *[nn.Conv1d(in_channels=self.speech_encoder_model.config.hidden_size,
                            out_channels=self.speech_encoder_model.config.hidden_size,
                            kernel_size=2,
                            stride=2) for _ in range(self.downloop_speech)]).to(self.device)
        else:
            self.length_adapters_speech = nn.Sequential(nn.Identity()).to(self.device)

        # length downscale - cv
        self.downsize_cv = down_scale_cv
        self.downloop_cv = int(math.log(self.downsize_cv, 2))
        if self.downsize_cv > 1:
            self.length_adapters_cv = nn.Sequential(
                *[nn.Conv1d(in_channels=self.cv_encoder_model.config.hidden_size,
                            out_channels=self.cv_encoder_model.config.hidden_size,
                            kernel_size=2,
                            stride=2) for _ in range(self.downloop_cv)]).to(self.device)
        else:
            self.length_adapters_cv = nn.Sequential(nn.Identity()).to(self.device)

        self.weights_sum = nn.Parameter(torch.zeros(self.num_speech_encoder_layers + 1)).to(self.device)
        self.enc_to_dec_proj_speech = nn.Linear(self.speech_encoder_model.config.hidden_size,
                                                self.decoder_model.config.hidden_size).to(self.device)
        self.enc_to_dec_proj_nlp = nn.Linear(self.nlp_encoder_model.config.hidden_size,
                                             self.decoder_model.config.hidden_size).to(self.device)
        self.enc_to_dec_proj_cv = nn.Linear(self.cv_encoder_model.config.hidden_size,
                                            self.decoder_model.config.hidden_size).to(self.device)
        self.custom_modules(**kwargs)
        self.nlp_encoder_model.eval()
        self.cv_encoder_model.eval()
        self.speech_encoder_model.eval()

        for xcoder in [self.speech_encoder_model.named_parameters,
                       self.nlp_encoder_model.named_parameters,
                       self.cv_encoder_model.named_parameters]:
            for name, param in xcoder():
                if param.requires_grad:
                    param.requires_grad = False

        # if fixed_parameters:
        #     for xcoder in [self.speech_encoder_model.named_parameters,
        #                    self.nlp_encoder_model.named_parameters,
        #                    self.cv_encoder_model.named_parameters]:
        #         for name, param in xcoder():
        #             if param.requires_grad:
        #                 if any([k in name for k in fixed_except]):
        #                     param.requires_grad = True
        #                 else:
        #                     param.requires_grad = False

        list_no_grad = []
        list_grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                list_grad.append(name)
            else:
                list_no_grad.append(name)

        self.speech_encoder_layer = len(self.speech_encoder_model.encoder.layers)
        self.cv_encoder_layer = len(self.cv_encoder_model.encoder.layer)
        self.nlp_encoder_layer = num_nlp_encoder_layers
        self.list_grad = list_grad
        self.list_no_grad = list_no_grad

    def custom_modules(self, **kwargs):
        return None

    def cal_loss(self, inputs_embeds, attention_mask=None, labels=None):
        output = self.seq2seq_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return output

    def forward(self, input_batch, labels=None, return_model_detail=False):
        tensor_array = []
        for batch in input_batch:
            input_embeds = []
            for i in batch:
                tensor = torch.tensor(i['array']).to(self.device)
                if i['type'] == 'audio':
                    input_embeds.append(torch.squeeze(
                        self.enc_to_dec_proj_speech(
                            self.length_adapters_speech(
                                tensor.transpose(1, 2)).transpose(1, 2))))
                if i['type'] == 'image':
                    input_embeds.append(torch.squeeze(
                        self.enc_to_dec_proj_cv(
                            self.length_adapters_cv(
                                tensor.transpose(1, 2)).transpose(1, 2))))
                if i['type'] == 'text':
                    input_embeds.append(torch.squeeze(
                        self.enc_to_dec_proj_nlp(
                            self.length_adapters_nlp(
                                tensor.transpose(1, 2)).transpose(1, 2))))
            tensor_array.append(torch.cat(input_embeds).to(self.device).squeeze(0))
        input_embeds = pad_sequence(tensor_array, batch_first=True, padding_value=-100).to(self.device)
        return_dict = {}
        outputs = self.cal_loss(inputs_embeds=input_embeds, labels=labels)
        return_dict['logits'] = torch.argmax(outputs['logits'], -1)
        if 'loss' in outputs:
            return_dict['loss'] = outputs['loss']
        return return_dict
