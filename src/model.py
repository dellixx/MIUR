from typing import Dict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.matrix_attention import DotProductMatrixAttention, CosineMatrixAttention, BilinearMatrixAttention, \
    LinearMatrixAttention
from allennlp.nn import util
from torch.nn import ModuleDict
from torch.nn.utils.rnn import pad_sequence

from data_utils import Scorer, BatchAverage, FScoreMetric, get_class_mapping, transmit_seq, CorpusBLEUMetric
from similar_functions import ElementWiseMatrixAttention


from mlp import MIUR


def count_parameters(model):
    parameter_count = ["Name: {}\t\tCount: {}".format(name, p.numel()) for name, p in model.named_parameters()
                       if p.requires_grad]
    return "\n".join(parameter_count)


@Model.register('rewrite')
class UnifiedFollowUp(Model):
    def __init__(self, vocab: Vocabulary,
                 text_encoder: Seq2SeqEncoder,
                 word_embedder: TextFieldEmbedder,
                 enable_training_log: bool = False,
                 inp_drop_rate: float = 0.2,
                 out_drop_rate: float = 0.2,
                 loss_weights: List = (0.2, 0.4, 0.4),
                 super_mode: str = 'before',
                 max_len: int = 160,
                 tc_emb: int = 320):
        super(UnifiedFollowUp, self).__init__(vocab)
        
        self.word_embedder = word_embedder

        """
        Define model arch choices
        """

        # input dropout
        if inp_drop_rate > 0:
            self.var_inp_dropout = InputVariationalDropout(p=inp_drop_rate)
        else:
            self.var_inp_dropout = lambda x: x
        # output dropout
        if out_drop_rate > 0:
            self.var_out_dropout = InputVariationalDropout(p=out_drop_rate)
        else:
            self.var_out_dropout = lambda x: x

        self.output_size = 768

        # dot -> dot product
        # cos -> cosine similarity
        # linear -> linear similarity
        self.segment_choices = ['dot', 'cos', 'linear'] 
        
        
        self.similar_function = ModuleDict({
            'ele': ElementWiseMatrixAttention(),
            'dot': DotProductMatrixAttention(),
            'cos': CosineMatrixAttention(),
            'emb_dot': DotProductMatrixAttention(),
            'emb_cos': CosineMatrixAttention(),
            'bilinear': BilinearMatrixAttention(matrix_1_dim=self.output_size, matrix_2_dim=self.output_size),
            'linear': LinearMatrixAttention(tensor_1_dim=self.output_size, tensor_2_dim=self.output_size)
        })


        self.class_mapping: Dict[str, int] = get_class_mapping(super_mode=super_mode)

        # MLP backbone (only one-layer)
        self.mlp_m = MIUR(dim= self.output_size, depth=1, token_dim=tc_emb, channel_dim=tc_emb, num_patch=max_len)


        class_zero_weight = loss_weights[0]
        class_one_weight = loss_weights[1]

        self.register_buffer('weight_tensor', torch.tensor([class_zero_weight, class_one_weight,
                                                            1 - class_zero_weight - class_one_weight]))
        self.loss = nn.CrossEntropyLoss(ignore_index=-1,
                                        weight=self.weight_tensor)

        # initialize metrics measurement
        self.metrics = {'ROUGE': BatchAverage(),
                        '_ROUGE1': BatchAverage(),
                        '_ROUGE2': BatchAverage(),
                        # TODO: You can speed up the code by disable BLEU since
                        #  the corpus-based BLEU metric is much time-consuming.
                        'BLEU': CorpusBLEUMetric(),
                        'EM': BatchAverage(),
                        'F1': FScoreMetric(prefix="1"),
                        'F2': FScoreMetric(prefix="2"),
                        'F3': FScoreMetric(prefix="3")}

        parameter_num = count_parameters(self)
        print(parameter_num)

        self.min_width = 8
        self.min_height = 8
        self.enable_training_log = enable_training_log
        self.bn = nn.BatchNorm2d(3)

        

    def forward(self, matrix_map: torch.Tensor,
                context_str: List[str],
                cur_str: List[str],
                restate_str: List[str],
                joint_tokens: Dict[str, torch.Tensor] = None,
                joint_border: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        attn_features = []

        # joint encoding
        if 'bert-type-ids' in joint_tokens:
            joint_tokens['bert-type-ids'] = torch.fmod(joint_tokens['bert-type-ids'], 2)

        joint_embedding = self.word_embedder(joint_tokens)
        joint_embedding = self.var_inp_dropout(joint_embedding)

        # MLP backbone
        joint_repr = self.mlp_m(joint_embedding)

        batch_size, _ = joint_border.shape
        joint_border = joint_border.view(batch_size)

        context_reprs = []
        context_embeddings = []
        cur_reprs = []
        cur_embeddings = []
        for i in range(batch_size):
            context_embeddings.append(joint_embedding[i, :joint_border[i]])
            context_reprs.append(joint_repr[i, :joint_border[i]])
            cur_embeddings.append(joint_embedding[i, joint_border[i]:])
            cur_reprs.append(joint_repr[i, joint_border[i]:])

        context_repr = pad_sequence(context_reprs, batch_first=True)
        cur_repr = pad_sequence(cur_reprs, batch_first=True)


        # padding feature map matrix 
        if context_repr.shape[1] < self.min_height:
            _, cur_height, hidden_size = context_repr.shape
            out_tensor = context_repr.data.new(batch_size, self.min_height, hidden_size).fill_(0)
            out_tensor[:, :cur_height, :] = context_repr
            context_repr = out_tensor

        if cur_repr.shape[1] < self.min_width:
            _, cur_width, hidden_size = cur_repr.shape
            out_tensor = cur_repr.data.new(batch_size, self.min_width, hidden_size).fill_(0)
            out_tensor[:, :cur_width, :] = cur_repr
            cur_repr = out_tensor



        for choice in self.segment_choices:
            attn_features.append(self.similar_function[choice](context_repr,cur_repr).unsqueeze(dim=1))
        
        # joint feature map
        attn_input = torch.cat(attn_features, dim=1)
        attn_input = self.bn(attn_input)
        attn_map = attn_input.permute(0, 2, 3, 1).contiguous()
        
        
        batch_size, width, height, class_size = attn_map.size()

        # if the current length and height is not equal to matrix-map
        if width != matrix_map.shape[1] or height != matrix_map.shape[2]:
            out_tensor = matrix_map.data.new(batch_size, width, height).fill_(-1)
            out_tensor[:, :matrix_map.shape[1], :matrix_map.shape[2]] = matrix_map
            matrix_map = out_tensor

        attn_mask = (matrix_map != -1).long()
        attn_map_flatten = attn_map.view(batch_size * width * height, class_size)
        matrix_map_flatten = matrix_map.view(batch_size * width * height).long()
        
        # cross entropy loss
        loss_val = self.loss(attn_map_flatten, matrix_map_flatten)
        outputs = {'loss': loss_val}
        

        if (self.training and self.enable_training_log) or (not self.training):
            attn_map_numpy = attn_map.data.cpu().numpy()
            attn_mask_numpy = attn_mask.data.cpu().numpy()
            predict_str = []

            for i in range(batch_size):
                sample_predict_str = self._predict_base_on_attn_map(attn_map_numpy[i],
                                                                    attn_mask_numpy[i],
                                                                    cur_str[i],
                                                                    context_str[i])
                if sample_predict_str.strip() == '':
                    # To avoid error when evaluating on ROUGE
                    sample_predict_str = 'hello'
                predict_str.append(sample_predict_str)

            self.evaluate_metrics(restate_str=restate_str,
                                  predict_str=predict_str,
                                  cur_str=cur_str)
            outputs['predicted_tokens'] = predict_str
        return outputs

    def evaluate_metrics(self, restate_str: List[str], predict_str: List[str], cur_str: List[str]):
        """
        BLEU Score
        """
        self.metrics['BLEU'](restate_str, predict_str)
        """
        Exact Match Score
        """
        em_score = Scorer.em_score(restate_str, predict_str)
        self.metrics['EM'](em_score)

        """
        ROUGE Score
        """
        rouge1, rouge2, rouge = Scorer.rouge_score(restate_str, predict_str)
        self.metrics['ROUGE'](rouge)
        self.metrics['_ROUGE1'](rouge1)
        self.metrics['_ROUGE2'](rouge2)

        """
        F-Score (note this one is the rewriting F-score)
        See definition in paper: https://ai.tencent.com/ailab/nlp/dialogue/papers/EMNLP_zhufengpan.pdf
        """
        i1c, p1c, r1c, i2c, p2c, r2c, i3c, p3c, r3c = Scorer.restored_count(
            restate_str, predict_str, cur_str)
        self.metrics['F1'](i1c, p1c, r1c)
        self.metrics['F2'](i2c, p2c, r2c)
        self.metrics['F3'](i3c, p3c, r3c)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        other_metrics = {k: v.get_metric(reset) for k, v in self.metrics.items() if k not in ['F1', 'F2', 'F3', 'BLEU']}
        f_metrics_dict = {k: v.get_metric(reset) for k, v in self.metrics.items() if k in ['F1', 'F2', 'F3']}
        f_metrics_dict = {**f_metrics_dict['F1'], **f_metrics_dict['F2'], **f_metrics_dict['F3']}
        bleu_metrics = self.metrics['BLEU'].get_metric(reset)
        return {**other_metrics, **f_metrics_dict, **bleu_metrics}

    def _predict_base_on_attn_map(self, attn_map, attn_mask, cur_str, context_str) -> str:
        """
        Detection the operation op, keeping the same format as the result of export_conflict_map
        :param attn_map: attention_map, with shape `height x width x class_size`
        :return: ordered operation sequence
        """
        discrete_attn_map = np.argmax(attn_map, axis=2)
        discrete_attn_map = attn_mask * discrete_attn_map
        op_seq: List = []

        for label, label_value in self.class_mapping.items():
            if label_value == 0:
                # do nothing
                continue
            connect_matrix = discrete_attn_map.copy()
            # make the non label value as zero
            connect_matrix = np.where(connect_matrix != label_value, 0,
                                      connect_matrix)
            ops = UnifiedFollowUp._scan_twice(connect_matrix)

            for op in ops:
                op_seq.append([label, *op])

        op_seq = sorted(op_seq, key=lambda x: x[2][1], reverse=True)
        predict_str = transmit_seq(cur_str, context_str, op_seq)
        return predict_str

    @staticmethod
    def _scan_twice(connect_matrix):
        label_num = 1
        label_equations = {}
        height, width = connect_matrix.shape
        for i in range(height):
            for j in range(width):
                if connect_matrix[i, j] == 0:
                    continue
                if j != 0:
                    left_val = connect_matrix[i, j - 1]
                else:
                    left_val = 0
                if i != 0:
                    top_val = connect_matrix[i - 1, j]
                else:
                    top_val = 0
                if i != 0 and j != 0:
                    left_top_val = connect_matrix[i - 1, j - 1]
                else:
                    left_top_val = 0
                if any([left_val > 0, top_val > 0, left_top_val > 0]):
                    neighbour_labels = [v for v in [left_val, top_val,
                                                    left_top_val] if v > 0]
                    min_label = min(neighbour_labels)
                    connect_matrix[i, j] = min_label
                    set_min_label = min([label_equations[label] for label in
                                         neighbour_labels])
                    for label in neighbour_labels:
                        label_equations[label] = min(set_min_label, min_label)
                    if set_min_label > min_label:
                        for key, value in label_equations:
                            if value == set_min_label:
                                label_equations[key] = min_label
                else:
                    new_label = label_num
                    connect_matrix[i, j] = new_label
                    label_equations[new_label] = new_label
                    label_num += 1
        for i in range(height):
            for j in range(width):
                if connect_matrix[i, j] == 0:
                    continue
                label = connect_matrix[i, j]
                normalized_label = label_equations[label]
                connect_matrix[i, j] = normalized_label
        groups = list(set(label_equations.values()))
        ret_boxes = []
        for group_label in groups:
            points = np.argwhere(connect_matrix == group_label)
            points_y = points[:, (0)]
            points_x = points[:, (1)]
            min_width = np.amin(points_x)
            max_width = np.amax(points_x) + 1
            min_height = np.amin(points_y)
            max_height = np.amax(points_y) + 1
            ret_boxes.append([[min_width, max_width], [min_height, max_height]])
        return ret_boxes
