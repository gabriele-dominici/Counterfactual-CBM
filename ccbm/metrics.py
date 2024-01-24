import torch


def variability(a: torch.Tensor, b: torch.Tensor):
    bool_a = a > 0.5
    bool_b = b > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    return len(unique_a) / len(unique_b) if len(unique_b) else -1


def intersection_over_union(a: torch.Tensor, b: torch.Tensor):
    bool_a = a > 0.5
    bool_b = b > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    intersection = unique_a.intersection(unique_b)
    union = unique_a.union(unique_b)
    return len(intersection) / len(union) if len(union) else -1

def cf_in_distribution(a: torch.Tensor, b: torch.Tensor):
    bool_a = a > 0.5
    bool_b = b > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    intersection = unique_a.intersection(unique_b)
    result = torch.zeros(1)
    for i in intersection:
        count = (bool_a == torch.tensor(i)).all(dim=1).sum()
        result += count
    return result.item() / bool_a.shape[0]

def distance_train(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, y_set: torch.Tensor):
    bool_a = (a > 0.5).float()
    bool_b = (b > 0.5).float()

    bool_a_ext = bool_a.repeat(bool_b.shape[0], 1, 1).transpose(1, 0)
    bool_b_ext = bool_b.repeat(bool_a.shape[0], 1, 1)
    # dist = torch.cdist(bool_a, bool_b)
    dist = (bool_a_ext != bool_b_ext).sum(dim=-1, dtype=torch.float)
    y_ext = y.repeat(y_set.shape[0], 1, 1).transpose(1, 0)
    y_set_ext = y_set.repeat(y.shape[0], 1, 1)
    filter = y_ext.argmax(dim=-1) != y_set_ext.argmax(dim=-1)
    dist[filter] = bool_a.shape[-1]
    min_distances = torch.min(dist, dim=-1)[0]
    return min_distances.mean()

def difference_over_union(a: torch.Tensor, b: torch.Tensor):
    bool_a = a > 0.5
    bool_b = b > 0.5
    unique_a = set([tuple(i) for i in bool_a.cpu().detach().numpy()])
    unique_b = set([tuple(i) for i in bool_b.cpu().detach().numpy()])
    difference = unique_a.difference(unique_b)
    union = unique_a.union(unique_b)
    return len(difference) / len(union) if len(union) else -1


if __name__ == '__main__':
    a = torch.FloatTensor([[0.9, 0.1], [0.1, 0.1], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])
    b = torch.FloatTensor([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.6, 0.4]])
    print(intersection_over_union(a, b))
    print(difference_over_union(a, b))


