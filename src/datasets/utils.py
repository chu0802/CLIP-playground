from torch.utils.data import DataLoader


class DataIterativeLoader:
    def __init__(self, dataloader, device="cuda"):
        self.len = len(dataloader)
        self.iterator = iter(dataloader)
        self.device = device

    def __iter__(self):
        num = 0
        while num < self.len:
            x, y = next(self.iterator)
            x = x.to(self.device)
            y = y.to(self.device)

            yield x, y
            num += 1

    def __len__(self):
        return self.len


def build_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def build_iter_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    device="cuda",
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return DataIterativeLoader(dataloader, device=device)
