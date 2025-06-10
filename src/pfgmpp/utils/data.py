from collections.abc import Iterable


class InfiniteDataLoader:
    def __init__(self, dataloader: Iterable):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
