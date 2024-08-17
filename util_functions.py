def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def scale_to_range(tensor, min_val, max_val):
    return tensor * (max_val - min_val) + min_val