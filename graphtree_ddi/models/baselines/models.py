"""基线网络定义。

MLP / DDIMDL-style 吃 4120 维药对特征；DeepDDI_SSP 吃 100 维 SSP-PCA；
DistMult / ComplEx：`--kge_entity lookup` 为 nn.Embedding 查表，`proj` 为 W·x_drug。
"""

from __future__ import annotations

from pathlib import Path
import sys as _sys

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "graphtree_ddi" / "paths.py").is_file():
            return p
    raise RuntimeError("cannot locate repository root (graphtree_ddi/paths.py)")

_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from common import MODALITY_SLICES, N_CLASSES


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


class PairMLP(nn.Module):
    """4120 → 1024-512-256 → 5，类别加权 CE 在训练循环里施加。"""

    def __init__(self, in_dim: int = 4120, dropout: float = 0.3, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            _mlp_block(in_dim, 1024, dropout),
            _mlp_block(1024, 512, dropout),
            _mlp_block(512, 256, dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SubDNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DDIMDLNet(nn.Module):
    """DDIMDL 式多模态 DNN：各模态子网络 + 联合融合网络。

    模态分块来自 preprocess 药对特征，不是原文的相似度矩阵+PCA。
    """

    def __init__(self, dropout: float = 0.3, n_classes: int = N_CLASSES):
        super().__init__()
        self.slices = {
            name: (sl.start, sl.stop) for name, sl in MODALITY_SLICES.items()
        }
        self.fp_net = _SubDNN(4096, 512, 128, dropout)
        self.cyp_net = _SubDNN(8, 32, 16, dropout)
        self.tp_net = _SubDNN(8, 32, 16, dropout)
        self.tgt_net = _SubDNN(8, 32, 16, dropout)
        fused = 128 + 16 + 16 + 16
        self.fusion = nn.Sequential(
            _mlp_block(fused, 256, dropout),
            _mlp_block(256, 128, dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fp = x[:, self.slices["fingerprint"][0]: self.slices["fingerprint"][1]]
        cyp = x[:, self.slices["cyp"][0]: self.slices["cyp"][1]]
        tp = x[:, self.slices["transporter"][0]: self.slices["transporter"][1]]
        tgt = x[:, self.slices["target"][0]: self.slices["target"][1]]
        h = torch.cat(
            [self.fp_net(fp), self.cyp_net(cyp), self.tp_net(tp), self.tgt_net(tgt)],
            dim=1,
        )
        return self.fusion(h)


class DeepDDINet(nn.Module):
    """SSP-PCA 拼接（默认 100 维）→ DNN → 5 类。"""

    def __init__(self, in_dim: int = 100, dropout: float = 0.3, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            _mlp_block(in_dim, 512, dropout),
            _mlp_block(512, 256, dropout),
            _mlp_block(256, 128, dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistMultClassifier(nn.Module):
    """5 个关系对应类别 0–4。lookup=nn.Embedding；proj=W·x_drug。"""

    def __init__(
        self,
        in_dim: int = 2048,
        emb_dim: int = 200,
        n_rel: int = N_CLASSES,
        entity_mode: str = "proj",
        n_drugs: int | None = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.entity_mode = entity_mode
        if entity_mode == "lookup":
            if not n_drugs:
                raise ValueError("lookup 需要 n_drugs")
            self.ent = nn.Embedding(int(n_drugs), emb_dim)
            nn.init.xavier_uniform_(self.ent.weight)
            self.proj = None
        else:
            self.proj = nn.Linear(in_dim, emb_dim, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
            self.ent = None
        self.rel = nn.Parameter(torch.empty(n_rel, emb_dim))
        nn.init.xavier_uniform_(self.rel)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.ent is not None:
            return self.ent(x.long())
        return self.proj(x)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        ha = self._embed(xa)
        hb = self._embed(xb)
        return torch.einsum("bd,rd,bd->br", ha, self.rel, hb)


class ComplExClassifier(nn.Module):
    """复数实体/关系。lookup 时 Embedding 输出 2*emb_dim=Re‖Im。"""

    def __init__(
        self,
        in_dim: int = 2048,
        emb_dim: int = 200,
        n_rel: int = N_CLASSES,
        entity_mode: str = "proj",
        n_drugs: int | None = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.entity_mode = entity_mode
        out = emb_dim * 2
        if entity_mode == "lookup":
            if not n_drugs:
                raise ValueError("lookup 需要 n_drugs")
            self.ent = nn.Embedding(int(n_drugs), out)
            nn.init.xavier_uniform_(self.ent.weight)
            self.proj = None
        else:
            self.proj = nn.Linear(in_dim, out, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
            self.ent = None
        self.rel = nn.Parameter(torch.empty(n_rel, out))
        nn.init.xavier_uniform_(self.rel)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.ent is not None:
            return self.ent(x.long())
        return self.proj(x)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        d = self.emb_dim
        ha = self._embed(xa)
        hb = self._embed(xb)
        ha_re, ha_im = ha[:, :d], ha[:, d:]
        hb_re, hb_im = hb[:, :d], hb[:, d:]
        rel_re, rel_im = self.rel[:, :d], self.rel[:, d:]
        scores = (
            torch.einsum("bd,rd,bd->br", ha_re, rel_re, hb_re)
            + torch.einsum("bd,rd,bd->br", ha_re, rel_im, hb_im)
            + torch.einsum("bd,rd,bd->br", ha_im, rel_re, hb_im)
            - torch.einsum("bd,rd,bd->br", ha_im, rel_im, hb_re)
        )
        return scores


def build_model(method: str, **kwargs) -> nn.Module:
    if method == "MLP":
        return PairMLP(
            in_dim=int(kwargs.get("in_dim", 4120)),
            dropout=float(kwargs.get("dropout", 0.3)),
        )
    if method == "DDIMDL":
        return DDIMDLNet(dropout=float(kwargs.get("dropout", 0.3)))
    if method == "DeepDDI_SSP":
        return DeepDDINet(
            in_dim=int(kwargs.get("in_dim", 100)),
            dropout=float(kwargs.get("dropout", 0.3)),
        )
    if method == "DistMult":
        return DistMultClassifier(
            in_dim=int(kwargs.get("fp_dim", 2048)),
            emb_dim=int(kwargs.get("emb_dim", 200)),
            entity_mode=str(kwargs.get("entity_mode", "proj")),
            n_drugs=kwargs.get("n_drugs"),
        )
    if method == "ComplEx":
        return ComplExClassifier(
            in_dim=int(kwargs.get("fp_dim", 2048)),
            emb_dim=int(kwargs.get("emb_dim", 200)),
            entity_mode=str(kwargs.get("entity_mode", "proj")),
            n_drugs=kwargs.get("n_drugs"),
        )
    raise ValueError(f"未知方法: {method}")


def n_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
