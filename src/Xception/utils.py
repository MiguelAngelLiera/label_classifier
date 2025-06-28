from numpy import ceil, floor
from typing import Union

def out_size(dimension: int, kernel_size: int, padding: int = 0, stride: int = 1) -> int:
    try:
        return int(floor((dimension - kernel_size + 2*padding)/stride)) + 1
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        return 0


def same_pad(dimension: int, kernel_size: int, stride: int) -> Union[int, str]:
    pad = (dimension * (stride - 1) + kernel_size - stride) / 2
    return int(ceil(pad))
