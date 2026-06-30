import torch
class Accumulator:
    """
    Used for Accumulating results for metrics.
    Args:
        n (int): number of accumulators
    """
    def __init__(self, n: int):
        self.data =[0.0] * n
        self.size = n

    def add(self, *args):
        """
        Args:
            *args: The number to be added to the accumulator
                The length of args must be equal to the size of the accumulator
        """
        assert len(args) == len(self.data), "The number of arguments does not match the size of the accumulator"
        self.data = torch.Tensor([a + float(b) for a, b in zip(self.data, args)])

    def clear(self):
        """
        Clears the accumulator.
        """
        self.data = [0.0] * self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size

    def resize(self, n: int):
        """
        Resize the accumulator.
        Args:
            n (int): The new size of the accumulator

        Returns:

        """
        self.data = [0.0] * self.size
        self.size = n
