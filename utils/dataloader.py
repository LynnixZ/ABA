# mini_batcher.py
import torch
import numpy as np

class DataLoader:
    def __init__(
        self,
        *,
        data_encoder,
        train_data_list,
        val_data_list,
        train_data_str,
        val_data_str,
        arithmetic_batch: bool,
        batch_size: int,
        block_size: int,
        operator: str,
        no_pad_in_target: bool,
        device,
        device_type: str,
        pad_char: str = ".",
        ignore_index: int = -1,
    ):
        self.data_encoder = data_encoder
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.train_data_str = train_data_str
        self.val_data_str = val_data_str
        self.arithmetic_batch = arithmetic_batch
        self.batch_size = batch_size
        self.block_size = block_size
        self.operator = operator
        self.no_pad_in_target = no_pad_in_target
        self.device = device
        self.device_type = device_type
        self.ignore_index = ignore_index

        def _first_id(tok_out):
            if isinstance(tok_out, (list, tuple, np.ndarray)):
                return int(tok_out[0])
            return int(tok_out)

        self.equal_token_id   = _first_id(self.data_encoder("=")) 
        self.dollar_token_id  = _first_id(self.data_encoder("$")) 
        self.newline_token_id = _first_id(self.data_encoder("\n"))
        self.pad_token_id     = _first_id(self.data_encoder(pad_char))

        if self.operator == "+":
            self.operator_token_id = _first_id(self.data_encoder("+")) 
        elif self.operator == "*":
            self.operator_token_id = _first_id(self.data_encoder("*"))
        else:
            self.operator_token_id = None

        if self.arithmetic_batch:
            self.train_tok_lines = [torch.tensor(self.data_encoder(line), dtype=torch.long)
                                    for line in self.train_data_list]
            self.val_tok_lines   = [torch.tensor(self.data_encoder(line), dtype=torch.long)
                                    for line in self.val_data_list]
        else:
            self.train_stream = np.asarray(self.data_encoder(self.train_data_str), dtype=np.int64)
            self.val_stream   = np.asarray(self.data_encoder(self.val_data_str),   dtype=np.int64)

    def _to_device(self, x, y):
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def get_batch(self, split: str, _compat_arithmetic_batch=None):
        if self.arithmetic_batch:
            data_tokenized = self.train_tok_lines if split == "train" else self.val_tok_lines
            assert len(data_tokenized) > 0, "no tokenized lines"
            line_ix = torch.randint(len(data_tokenized), (self.batch_size,))
            line_list = []
            for idx in line_ix:
                t = data_tokenized[int(idx)]
                if t.numel() < 2:
                    t = torch.tensor([0, 0], dtype=torch.long)
                line_list.append(t)

            max_len = max(len(t) for t in line_list)
            samples = torch.full((self.batch_size, max_len), fill_value=self.pad_token_id, dtype=torch.long)
            for i, t in enumerate(line_list):
                samples[i, :len(t)] = t

            x = samples[:, :-1].clone()
            y = samples[:, 1:].clone()

        else:
            data = self.train_stream if split == "train" else self.val_stream
            assert len(data) > self.block_size, "data shorter than block_size"
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
            x = torch.stack([torch.from_numpy(data[i:i + self.block_size]) for i in ix])
            y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size]) for i in ix])

        if not self.no_pad_in_target:
            mask = torch.ones_like(y, dtype=torch.bool)

            for b in range(y.size(0)):
                seq = y[b]
                eq_pos = (seq == self.equal_token_id).nonzero(as_tuple=False).flatten().tolist()
                dol_pos = (seq == self.dollar_token_id).nonzero(as_tuple=False).flatten().tolist()
                if not dol_pos:
                    dol_pos = (seq == self.newline_token_id).nonzero(as_tuple=False).flatten().tolist()
                dol_pos.sort()

                if self.operator_token_id is not None:
                    op_pos = (seq == self.operator_token_id).nonzero(as_tuple=False).flatten().tolist()
                else:
                    op_pos = []

                for eq in eq_pos:
                    nxt = next((p for p in dol_pos if p >= eq), None)
                    if nxt is None:
                        continue
                    if self.operator_token_id is not None and self.operator in ["+", "*"]:
                        if not any(op < eq for op in op_pos):
                            continue
                    left = eq + 1
                    right = nxt + 1
                    if left < y.size(1):
                        mask[b, left:right] = False

            y[mask] = self.ignore_index

        return self._to_device(x, y)


    def preview(self, split: str, data_decoder, max_show: int = 10, print_raw: bool = False):
        x, y = self.get_batch(split)
        if print_raw:
            print(y)

        y = y.clone()
        ignore_val = getattr(self, "ignore_index", -1)
        y[y == ignore_val] =  self.pad_token_id

        decoded_x = [data_decoder(seq.cpu().tolist()) for seq in x]
        decoded_y = [data_decoder(seq.cpu().tolist()) for seq in y]

        for i in range(min(len(decoded_x), max_show)):
            print(f"{split} Sample {i}:")
            print(f"x = {decoded_x[i]}")
            print(f"y = {decoded_y[i]}")
            print("-" * 50)

        return decoded_x, decoded_y

    def preview_train_val(self, data_decoder,  max_show: int = 10, print_raw: bool = False):
        self.preview("train", data_decoder, max_show=max_show, print_raw=print_raw)
        self.preview("val",   data_decoder, max_show=max_show, print_raw=print_raw)
