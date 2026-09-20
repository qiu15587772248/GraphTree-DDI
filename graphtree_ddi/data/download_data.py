"""
DrugBank 完整数据解析脚本 (v2)
需要文件（放在 data/raw/ 目录下）：
  - full database.xml   （从 COMPLETE DATABASE 标签下载，解压后）
  - structures.sdf      （从 STRUCTURES 标签下载 All，解压后）

解析内容：
  - 药物基本信息（ID、名称、ATC分类）
  - 分子结构 SMILES（从SDF解析，比XML快10倍）
  - 代谢酶（enzyme）：名称 + action（substrate/inhibitor/inducer）
  - 转运体（transporter）：名称 + action
  - 靶点（target）：名称 + action（agonist/antagonist/inhibitor 等）
  - 药物-药物相互作用（含描述文本，用于风险标注）

v2 改动（2026-04）：
  - enzyme/transporter/target 均提取 action 类型，存为 "名称:action" 格式
  - 新增 transporters 列和 target_actions 列
  - 修复 v1 中同一酶名重复出现的问题（同一药-酶有多条 action 时合并为一条）
"""

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

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from graphtree_ddi.paths import DATA_RAW  # noqa: E402
RAW_DIR = DATA_RAW

DRUGBANK_XML = RAW_DIR / "full database.xml"
STRUCTURES_SDF = RAW_DIR / "structures.sdf"



# ─────────────────────────────────────────────────
# Step 1：从 SDF 快速提取 DrugBank ID → SMILES 映射
# ─────────────────────────────────────────────────
def parse_smiles_from_sdf(sdf_path: Path = STRUCTURES_SDF) -> dict:
    """
    DrugBank SDF 文件每个分子条目包含 DRUGBANK_ID 属性。
    返回 {drugbank_id: smiles} 字典。
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("请先安装 rdkit: conda install -c conda-forge rdkit")

    if not sdf_path.exists():
        print(f"[跳过] 未找到 {sdf_path.name}，将从 XML 中提取 SMILES（较慢）")
        return {}

    print(f"从 {sdf_path.name} 提取 SMILES...")
    smiles_map = {}

    # 判断是否为 gzip 压缩（文件头魔数 1f 8b）
    with open(sdf_path, "rb") as f:
        magic = f.read(2)
    is_gzip = (magic == b"\x1f\x8b")

    try:
        if is_gzip:
            print("  检测到 gzip 压缩格式，使用解压读取...")
            import gzip
            file_obj = gzip.open(str(sdf_path), "rb")
        else:
            # 用 Python 打开文件再传给 RDKit，避免中文路径问题
            file_obj = open(sdf_path, "rb")

        supplier = Chem.ForwardSDMolSupplier(
            file_obj, sanitize=False, removeHs=True
        )

        for mol in tqdm(supplier, desc="解析SDF"):
            if mol is None:
                continue
            props = mol.GetPropsAsDict()

            # DrugBank SDF 的 ID 字段（尝试多种属性名）
            db_id = (props.get("DRUGBANK_ID")
                     or props.get("DATABASE_ID")
                     or props.get("drugbank_id", ""))
            # 备选：从分子名（第一行）读取
            if not db_id and mol.HasProp("_Name"):
                candidate = mol.GetProp("_Name").strip().split()[0]
                if candidate.startswith("DB"):
                    db_id = candidate

            if not db_id or not db_id.startswith("DB"):
                continue

            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi:
                    smiles_map[db_id] = smi
            except Exception:
                pass

    except Exception as e:
        print(f"  ⚠️  SDF 解析失败（{e}），将从 XML 中提取 SMILES（速度较慢）")
        return {}

    print(f"  [完成] 获取 SMILES: {len(smiles_map)} 个药物")
    return smiles_map


# ─────────────────────────────────────────────────
# Step 2：解析 DrugBank XML
# ─────────────────────────────────────────────────
def parse_drugbank_xml(xml_path: Path = DRUGBANK_XML,
                       smiles_map: dict = None) -> tuple:
    """
    解析 full database.xml，提取：
      - 药物信息表 (drugs_df)
      - 药物相互作用表 (ddi_df)

    参数 smiles_map: 若已从SDF读取，直接使用；否则从XML提取
    """
    try:
        from lxml import etree
    except ImportError:
        raise ImportError("请先安装 lxml: pip install lxml")

    if not xml_path.exists():
        raise FileNotFoundError(
            f"未找到 {xml_path}\n"
            "请将 DrugBank 'full database.xml' 放入 data/raw/ 目录"
        )

    if smiles_map is None:
        smiles_map = {}

    ns = {"db": "http://www.drugbank.ca"}
    drugs, interactions = [], []

    print(f"\n解析 {xml_path.name}（文件较大，约需 5-15 分钟）...")
    context = etree.iterparse(
        str(xml_path), events=("end",),
        tag=f"{{{ns['db']}}}drug"
    )

    for _, drug_elem in tqdm(context, desc="解析药物条目"):
        # 只处理小分子药物
        if drug_elem.get("type") != "small molecule":
            drug_elem.clear()
            continue

        db_id_elem = drug_elem.find("db:drugbank-id[@primary='true']", ns)
        if db_id_elem is None:
            drug_elem.clear()
            continue
        db_id = db_id_elem.text

        # 药物名称
        name_elem = drug_elem.find("db:name", ns)
        name = name_elem.text if name_elem is not None else ""

        # SMILES：优先用SDF，否则从XML提取
        smiles = smiles_map.get(db_id, "")
        if not smiles:
            smiles_elem = drug_elem.find(
                ".//db:calculated-properties/db:property"
                "[db:kind='SMILES']/db:value", ns
            )
            if smiles_elem is not None:
                smiles = smiles_elem.text or ""

        # ATC 分类
        atc_codes = [
            e.get("code", "")
            for e in drug_elem.findall(".//db:atc-code", ns)
        ]

        # ── 代谢酶：提取 name + action（substrate/inhibitor/inducer）──
        # 格式："名称:action1,action2"，同一药-酶的多条 action 合并
        # 例如 "Cytochrome P450 3A4:substrate,inhibitor"
        enzyme_entries: dict[str, set[str]] = {}  # {name: {action1, action2}}
        for enz in drug_elem.findall(".//db:enzymes/db:enzyme", ns):
            enz_name_e = enz.find("db:name", ns)
            if enz_name_e is None or not enz_name_e.text:
                continue
            ename = enz_name_e.text.strip()
            actions = set()
            for a in enz.findall(".//db:actions/db:action", ns):
                if a.text and a.text.strip():
                    actions.add(a.text.strip().lower())
            if ename in enzyme_entries:
                enzyme_entries[ename] |= actions
            else:
                enzyme_entries[ename] = actions
        cyp_enzymes_v2 = []
        for ename, acts in enzyme_entries.items():
            if acts:
                cyp_enzymes_v2.append(f"{ename}:{','.join(sorted(acts))}")
            else:
                cyp_enzymes_v2.append(ename)

        # ── 转运体：提取 name + action ──
        tp_entries: dict[str, set[str]] = {}
        for tp in drug_elem.findall(".//db:transporters/db:transporter", ns):
            tp_name_e = tp.find("db:name", ns)
            if tp_name_e is None or not tp_name_e.text:
                continue
            tname = tp_name_e.text.strip()
            actions = set()
            for a in tp.findall(".//db:actions/db:action", ns):
                if a.text and a.text.strip():
                    actions.add(a.text.strip().lower())
            if tname in tp_entries:
                tp_entries[tname] |= actions
            else:
                tp_entries[tname] = actions
        transporters_v2 = []
        for tname, acts in tp_entries.items():
            if acts:
                transporters_v2.append(f"{tname}:{','.join(sorted(acts))}")
            else:
                transporters_v2.append(tname)

        # ── 靶点：提取 name + action ──
        tgt_entries: dict[str, set[str]] = {}
        for tgt in drug_elem.findall(".//db:targets/db:target", ns):
            tgt_name_e = tgt.find("db:name", ns)
            if tgt_name_e is None or not tgt_name_e.text:
                continue
            tgtname = tgt_name_e.text.strip()
            actions = set()
            for a in tgt.findall(".//db:actions/db:action", ns):
                if a.text and a.text.strip():
                    actions.add(a.text.strip().lower())
            if tgtname in tgt_entries:
                tgt_entries[tgtname] |= actions
            else:
                tgt_entries[tgtname] = actions
        targets_v2 = []
        for tgtname, acts in tgt_entries.items():
            if acts:
                targets_v2.append(f"{tgtname}:{','.join(sorted(acts))}")
            else:
                targets_v2.append(tgtname)

        drugs.append({
            "drugbank_id": db_id,
            "name":        name,
            "smiles":      smiles,
            "atc_codes":   "|".join(atc_codes),
            "cyp_enzymes": "|".join(cyp_enzymes_v2),
            "transporters": "|".join(transporters_v2),
            "targets":     "|".join(targets_v2),
        })

        # DDI 记录
        for ddi in drug_elem.findall(".//db:drug-interaction", ns):
            target_id = ddi.find("db:drugbank-id", ns)
            desc      = ddi.find("db:description", ns)
            if target_id is not None and target_id.text:
                interactions.append({
                    "drug1_id":    db_id,
                    "drug2_id":    target_id.text,
                    "description": desc.text if desc is not None else "",
                })

        drug_elem.clear()

    drugs_df = pd.DataFrame(drugs)
    ddi_df   = pd.DataFrame(interactions)

    # 保存
    drugs_df.to_csv(RAW_DIR / "drugbank_drugs.csv",  index=False, encoding="utf-8")
    ddi_df.to_csv(  RAW_DIR / "drugbank_ddi.csv",    index=False, encoding="utf-8")

    smiles_count  = (drugs_df["smiles"] != "").sum()
    enzyme_count  = (drugs_df["cyp_enzymes"] != "").sum()
    tp_count      = (drugs_df["transporters"] != "").sum()
    target_count  = (drugs_df["targets"] != "").sum()

    print(f"\n{'='*55}")
    print(f"[完成] 药物总数:            {len(drugs_df):>8,}")
    print(f"       有SMILES结构:         {smiles_count:>8,}")
    print(f"       有代谢酶信息:         {enzyme_count:>8,}")
    print(f"       有转运体信息:         {tp_count:>8,}")
    print(f"       有靶点信息:           {target_count:>8,}")
    print(f"       DDI记录总数:          {len(ddi_df):>8,}")
    print(f"{'='*55}")

    return drugs_df, ddi_df


# ─────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────
def run():
    print("=" * 55)
    print("  DrugBank 完整数据解析")
    print("=" * 55)

    # Step 1: 从 SDF 快速获取 SMILES
    smiles_map = parse_smiles_from_sdf()

    # Step 2: 解析 XML（传入 smiles_map 避免重复提取）
    drugs_df, ddi_df = parse_drugbank_xml(smiles_map=smiles_map)

    print(f"\n✅ 解析完成，可以运行 preprocess.py")


if __name__ == "__main__":
    run()
