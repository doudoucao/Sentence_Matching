import torch
import torch.nn as nn


def sort_by_seq_lens(batch, sequence_lengths, descending=True):
    """
    sort a batch of padded variable length sequences by length
    :param batch: A batch of padded Variable length sequence. The batch should have the dimensions (batch_size, max_seq, *)
    :param sequence_lengths: A tensor containing the lengths of the sequences in the input batch
    :param descending:
    :return: sorted_batch :  A tensor containing the input batch reordered by sequences lengths
            sorted_seq_lens: A tensor containing the sorted lengths of the sequences in the batch
            sorting_idx: A tensor containing the indices used to permute the input batch in order to get 'sorted batch'
            restoration_idx: A tensor containing the indices that can be used to restore the order of sequences in 'sorted batch'
    """
    sorted_seq_lens, sorting_index = sequence_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)

    idx_range = sequence_lengths.new_tensor(torch.arange(0, len(sequence_lengths)))

    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def get_mask(sequence_batch, sequence_lengths):
    """
    Get the mask for a batch of padded variable length sequence.
    :param sequence_batch: A batch of padded  variable length sequences containing word indices
    :param sequence_lengths:
    :return: A mask of size (batch max_sequence_len)
    """
    batch_size = sequence_batch.size()[0]
    max_length = torch.max(sequence_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequence_batch[:, :max_length] == 0] = 0.0
    return mask


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor
    The input tensor and mask should be of size (batch, * , sequence_len)
    :param tensor: the tensor on which the softmax function must be applied along the last dimension.
    :param mask:  A mask of the same size as the tensor.
    :return: A tensor of the same size as the inputs containing the result of the softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)

    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of tensor, and mask the vectors in the result with 'mask'
    :param tensor:
    :param weights:
    :param mask:
    :return:
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)

    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


def replace_masked(tensor, mask, value):
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def max_pooling(tensor, mask, dim=0):
    masks = mask.view(mask.size(0), mask.size(1), -1)
    masks = masks.expand(-1, -1, tensor.size(2)).float()
    return torch.max(tensor + (-999999*masks.le(0.5).float()), dim=dim)


def mean_pooling(tensor, mask, dim=0):
    masks = mask.view(mask.size(0), mask.size(1), -1).float()
    return torch.sum(tensor*masks, dim=dim) / torch.sum(masks, dim=1)

