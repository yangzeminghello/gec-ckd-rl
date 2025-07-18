# coding:utf-8
"""
    @文件：ckd_rl_entropy.py
    @时间：2024/11/23 15:12
    @作者：杨泽明
    @邮箱：2046492745@qq.com
"""
import math
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import logging
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
from fairseq.data.encoders.gpt2_bpe import get_encoder

logger = logging.getLogger('transformers')
logger.setLevel(logging.ERROR)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def get_student_token(student_out, target):
    r = target.shape[0]
    l = target.shape[1]
    target = target.view(-1)
    student_token = torch.tensor([])
    if target.dim() != 1:
        target = target.view(-1)
    for index, token in enumerate(target):
        if index == 0:
            student_token = student_out[index][token].unsqueeze(0)
        else:
            student_token = torch.cat((student_token, student_out[index][token].unsqueeze(0)), dim=0)
    return student_token.reshape(r, l)


def compute_rl_loss(student_token, q_value, reward):
    q_value = F.softmax(q_value, dim=-1)
    # reward = F.cosine_similarity(F.softmax(student_token, dim=-1), q_value, dim=-1).unsqueeze(1)
    target_mask = student_token != 0
    rl_loss = -1 * torch.sum(torch.abs(reward) * (q_value * student_token))
    return rl_loss / target_mask.sum()


def idx_to_str(idx):
    if idx.dim() != 2:
        idx = idx.unsqueeze(0)
    idx = idx.cpu().numpy()
    text_list = []
    for i in idx:
        text_list.append(' '.join([str(item) for item in i]))
    return text_list


def t_decode(idx_list, encoder):
    text_list = []
    for item in idx_list:
        _, text = encoder.decode_lines([item])
        text_list.append(text)
    return text_list


def compute_positive_negative_loss(teacher_well, teacher_bed, student_well, student_bed, KL):
    # teacher_well, teacher_bed, student_well, student_bed = compute_positive_negative_set(student_out, teacher_out, device)
    if teacher_well.numel() != 0:
        positive_kd = KL(student_bed, F.log_softmax(teacher_well, dim=0))
    else:
        positive_kd = torch.tensor(0)
    if teacher_bed.numel() != 0:
        negative_kd = KL(student_well, F.log_softmax(teacher_bed))
        negative_kd = torch.clamp((0.1 - negative_kd), max=0)
    else:
        negative_kd = torch.tensor(0)

    return positive_kd, negative_kd


def compute_positive_negative_set(student_out, teacher_out, device):
    t_pos = torch.tensor([]).to(device)
    t_neg = torch.tensor([]).to(device)
    s_pos = torch.tensor([]).to(device)
    s_neg = torch.tensor([]).to(device)
    for s_item, t_item in zip(student_out, teacher_out):
        s_item = s_item.to(device)
        t_item = t_item.to(device)
        s_exp_probs = F.softmax(s_item)
        q_student = -1 * torch.sum((s_item * s_exp_probs), dim=0)
        q_teacher = -1 * torch.sum((F.softmax(t_item, dim=0) * F.log_softmax(t_item, dim=0)), dim=0)
        if q_teacher >= q_student:
            t_pos = torch.cat((t_pos, t_item.unsqueeze(0)), dim=0)
            s_neg = torch.cat((s_neg, s_item.unsqueeze(0)), dim=0)
        else:
            t_neg = torch.cat((t_neg, t_item.unsqueeze(0)), dim=0)
            s_pos = torch.cat((s_pos, s_item.unsqueeze(0)), dim=0)
    return t_pos, t_neg, s_pos, s_neg


def compute_ki_loss(previous_out, current_out, KL, device):
    previous_out = previous_out.to(device)
    current_out = current_out.to(device)
    pre_r = previous_out.shape[0]
    cur_r = current_out.shape[0]
    cur_l = current_out.shape[1]
    if pre_r > cur_r:
        previous_out = dimensional_transformation(previous_out, pre_r, cur_r, cur_l, device)
    elif pre_r < cur_r:
        current_out = dimensional_transformation(current_out, cur_r, pre_r, cur_l, device)
    ki = KL(previous_out, current_out)
    return torch.abs(ki)


def dimensional_transformation(out1, out1_r, out2_r, l, device):
    out1 = out1.reshape(l, out1_r).to(device)
    linear1 = torch.nn.Linear(in_features=out1_r, out_features=out2_r).to(device)
    out1 = linear1(out1)
    out1 = out1.reshape(out2_r, l)
    return out1


def Semantic_embedding(sentence, device):
    bert_model = BertModel.from_pretrained(
        '/home/jiangzuo/yangzeming/generic-pretrained-GEC-master/BART-GEC/english/bert-base-uncased/',
        local_files_only=True).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(
        '/home/jiangzuo/yangzeming/generic-pretrained-GEC-master/BART-GEC/english/bert-base-uncased/',
        local_files_only=True)

    teacher_out = torch.tensor([]).to(device)
    for text in sentence:
        encoded_input = bert_tokenizer.encode(text, add_special_tokens=False, max_length=512,
                                              truncation=True)
        input_ids = torch.tensor([encoded_input]).to(device)
        with torch.no_grad():
            outputs = bert_model(input_ids)
        pooled_output = outputs.last_hidden_state.squeeze(0).to(device)
        teacher_out = torch.cat((teacher_out, pooled_output), dim=0)
    return teacher_out  # F.softmax(teacher_out, dim=0)


def transformation_row(x, max_r, min_r, device):
    x = x.reshape(1, -1).to(device)
    row_line = torch.nn.Linear(max_r, min_r).to(device)
    x = row_line(x)
    return x.reshape(-1, 1)


def compute_all_loss(sample, sample_lm, lprobs, t, KL, device, encoder):
    denominator = 1 - 0.999 ** t
    # 分子
    molecule = 1 - 0.999 ** (t - 1)
    if denominator == 0:
        alpha = 1
    else:
        alpha = 0.999 * molecule / denominator

    student_token = get_student_token(lprobs, sample['target']).view(-1, 1)  # [n * m]
    teacher_token = sample_lm['target'].view(-1, 1)

    teacher_list = idx_to_str(teacher_token)
    teacher_text = t_decode(teacher_list, encoder)

    teacher_out = Semantic_embedding(teacher_text, device)

    teacher_line = torch.nn.Linear(768, 1).to(device)
    teacher_out = teacher_line(teacher_out)
    t_r = teacher_out.shape[0]
    s_r = student_token.shape[0]
    if s_r > t_r:
        student_token = transformation_row(student_token, s_r, t_r, device)
    elif t_r > s_r:
        teacher_out = transformation_row(teacher_out, t_r, s_r, device)

    # 完全蒸馏
    kd_loss = kd_loss_plain(student_token, teacher_out, KL)
    # 置信度过滤
    conf_filter_loss = kd_loss_conf_filter(student_token, F.softmax(teacher_out, dim=0))
    # 拒绝采样
    target_mask = student_token != 0
    rejection_loss = kd_loss_rejection(student_token, F.softmax(teacher_out, dim=0), target_mask)


    teacher_well, teacher_bed, student_well, student_bed = compute_positive_negative_set(student_token,
                                                                                         teacher_out, device)
    positive_loss, negative_loss = compute_positive_negative_loss(teacher_well, teacher_bed, student_well, student_bed, KL)
    kf_loss = positive_loss

    rl_loss = compute_rl_loss(student_well, teacher_bed, negative_loss)

    return kd_loss
    # return conf_filter_loss
    # return rejection_loss
    # return kf_loss, alpha, rl_loss

# Confidence-Based Filtering
def kd_loss_conf_filter(student_logits, teacher_probs, conf_threshold=0.8):
    """
    只对 teacher 置信度高于阈值的 token 计算 cross-entropy loss
    """
    # teacher 的 soft label 置信度
    mask = teacher_probs >= conf_threshold
    # cross entropy loss
    loss_per_token = F.kl_div(student_logits, teacher_probs, reduction='none').sum(dim=-1)  # (B, T)
    loss_masked = loss_per_token.masked_fill(~mask, 0.0)

    return loss_masked.sum() / mask.sum().clamp(min=1)

# Rejection Sampling
def kd_loss_rejection(student_logits, teacher_probs, token_mask):
    """
    对 teacher 一致性不高的 token，直接拒绝蒸馏（通过 mask 控制）
    - token_mask: (B, T) = True 表示保留该 token 蒸馏
    """
    # log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_probs, dim=-1)  # 确保 teacher_probs 是概率分布
    loss_per_token = F.kl_div(student_logits, teacher_probs, reduction='none').sum(dim=-1)
    loss_masked = loss_per_token.masked_fill(~token_mask, 0.0)
    return loss_masked.sum() / token_mask.sum().clamp(min=1)

# Plain KD
def kd_loss_plain(student_token, teacher_out, KL):
    return KL(student_token, F.log_softmax(teacher_out, dim=0))


@register_criterion('ckd_entropy')
class CkdEntropy(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.pre_out = torch.tensor([0], dtype=torch.float)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.KL = torch.nn.KLDivLoss(log_target=True, reduction="sum")
        self.encoder_json = './bart_model/encoder.json'
        self.vocab_bpe = './bart_model/vocab.bpe'
        self.encoder = MultiprocessingEncoder(self.encoder_json, self.vocab_bpe)
        self.encoder.initializer()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, sample_lm=None, is_first=0, t=1, reduce=True):

        net_output = model(**sample['net_input'])
        loss_ce, nll_loss, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)

        if sample_lm is not None:
            # kf_loss, alpha, rl_loss = compute_all_loss(sample, sample_lm, lprobs, t, self.KL, self.device, self.encoder)
            kd_loss = compute_all_loss(sample, sample_lm, lprobs, t, self.KL, self.device, self.encoder)

            if is_first != 0:
                ki_loss = compute_ki_loss(self.pre_out, lprobs, self.KL, self.device)
            else:
                ki_loss = torch.tensor(0)
            self.pre_out = lprobs

            loss = loss_ce + alpha * (kf_loss + rl_loss) + (1 - alpha) * ki_loss
        #     完全蒸馏
            loss = loss_ce + kd_loss + ki_loss
        else:
            loss = loss_ce

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss, lprobs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2 ** meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


class MultiprocessingEncoder(object):

    def __init__(self, encoder_json, vocab_bpe, keep_empty=False):
        self.encoder_json = encoder_json
        self.vocab_bpe = vocab_bpe
        self.keep_empty = keep_empty

    def initializer(self):
        global bpe
        bpe = get_encoder(self.encoder_json, self.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]
