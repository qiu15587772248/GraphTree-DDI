"""
数据预处理 (v2)：
  - 全库 DrugBank 数据
  - 5级分类（0-4），clinician-reviewed in March 2026
  - 4120维特征 = 2048×2（Morgan指纹积/差）+ 24维药理特征
      CYP酶 8维（底物/抑制剂/诱导剂 action 交互）
      转运体 8维（P-gp/OATP1B1 + 底物-抑制剂交互）
      靶点 8维（激动/拮抗/抑制 action 交互）
  - 去除 A-B/B-A 镜像重复（DrugBank 双向存储导致）
  - 全量模式：特征写磁盘（use_memmap=True），训练时按需读取
  - 负样本构建（结构远离 + 靶点不重叠）

v2 改动（2026-04）：
  - 药理特征从4维扩展到24维，利用enzyme/transporter/target的action类型
  - 数据来源：download_data.py v2 生成的 "名称:action" 格式字段
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

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _split_multi_field(text) -> set[str]:
    """
    兼容两种分隔符：'|' 和 ';'
    例如：'CYP3A4|CYP2D6' 或 'CYP3A4;CYP2D6'
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return set()
    s = str(text).strip()
    if not s:
        return set()
    parts = re.split(r"[|;]", s)
    return {p.strip() for p in parts if p and p.strip() and p.strip().lower() != "nan"}


def _parse_v2_field(text) -> dict[str, set[str]]:
    """
    解析 v2 格式的 enzyme/transporter/target 字段。
    输入格式：'Name1:action1,action2|Name2:action3|Name3'（无action则归入 'unknown'）
    返回：{action_type: {name1, name2, ...}}
    例如：{'substrate': {'CYP3A4','CYP2D6'}, 'inhibitor': {'CYP3A4'}}
    """
    result: dict[str, set[str]] = {}
    entries = _split_multi_field(text)
    for entry in entries:
        if ":" in entry:
            name, acts_str = entry.split(":", 1)
            name = name.strip()
            acts = [a.strip().lower() for a in acts_str.split(",") if a.strip()]
            if not acts:
                acts = ["unknown"]
        else:
            name = entry.strip()
            acts = ["unknown"]
        if not name:
            continue
        for act in acts:
            result.setdefault(act, set()).add(name)
    return result


def _names_by_action(parsed: dict[str, set[str]], action: str) -> set[str]:
    """从解析结果中获取某个 action 类型对应的全部名称集合"""
    return parsed.get(action, set())


def _all_names(parsed: dict[str, set[str]]) -> set[str]:
    """从解析结果中获取全部名称（不分 action）"""
    all_n: set[str] = set()
    for names in parsed.values():
        all_n |= names
    return all_n


_CYP_PREFIX = "Cytochrome P450"
_PGP_NAMES = {"P-glycoprotein 1", "Multidrug resistance protein 1",
              "P-glycoprotein", "MDR1"}
_OATP1B1_NAMES = {"Solute carrier organic anion transporter family member 1B1",
                  "OATP1B1"}


def _is_cyp(name: str) -> bool:
    return name.startswith(_CYP_PREFIX)


def _filter_cyp(names: set[str]) -> set[str]:
    return {n for n in names if _is_cyp(n)}


def _has_pgp(names: set[str]) -> bool:
    return bool(names & _PGP_NAMES)


def _has_oatp1b1(names: set[str]) -> bool:
    return bool(names & _OATP1B1_NAMES)

# 新版 RDKit Morgan 生成器（替代 GetMorganFingerprintAsBitVect）
try:
    from rdkit.Chem import rdFingerprintGenerator as _rdFPGen
    _MORGAN_GEN = _rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
    _MORGAN_GEN_INITED = True
except Exception:
    _MORGAN_GEN = None
    _MORGAN_GEN_INITED = False

from graphtree_ddi.paths import DATA_PROCESSED, DATA_RAW  # noqa: E402
RAW_DIR = DATA_RAW
PROCESSED_DIR = DATA_PROCESSED

# ── 5级分类：DrugBank 描述模板 + 医生审核的症状短语映射 ──────────────
#
#   模板B/C/D/E/F/G/H（→ 级别1）：PK/PD机制变化，未命名具体症状
#   模板A（→ 级别2/3/4）：含 "the risk or severity of [具体症状]"
#       症状短语 → 级别由医生于2026年3月按临床严重程度审核确认
#
# 分级定义（WHO/FDA ICH-E2A参考）：
#   4 = 危重：生命威胁/可致死/不可逆器官损伤（需立即干预）
#   3 = 严重：需住院/延长住院/重要医学事件（需密切监护）
#   2 = 中度：有具体症状但不危及生命（需关注，可门诊处理）
#   1 = 一般：PK/PD机制变化，无命名具体症状（剂量调整参考）
#   0 = 安全：负样本，无已知DDI

_RISK_SEVERITY_RE  = re.compile(r"the risk or severity of", re.IGNORECASE)
_GENERIC_ADVERSE_RE = re.compile(
    r"the risk or severity of adverse (effects|events)", re.IGNORECASE
)
_EXTRACT_PHRASE_RE  = re.compile(
    r"the risk or severity of (.+?) can be", re.IGNORECASE
)

# 医生标注的症状短语 → 风险等级（2=中度, 3=严重, 4=危重）
# clinician-reviewed on 184 DrugBank Level-2 phrases in March 2026
PHRASE_LABEL_MAP: dict[str, int] = {
    'bleeding': 4,
    'hypoglycemia': 3,
    'bleeding and hemorrhage': 4,
    'nephrotoxicity': 3,
    'serotonin syndrome': 4,
    'gastrointestinal bleeding': 4,
    'renal failure, hyperkalemia, and hypertension': 4,
    'myopathy, rhabdomyolysis, and myoglobinuria': 4,
    'hemorrhage': 4,
    'bleeding and bruising': 4,
    'renal failure, hypotension, and hyperkalemia': 4,
    'hypotension, sedation, death, somnolence, and respiratory depression': 4,
    'renal failure': 4,
    'seizure': 4,
    'hypotension and syncope': 4,
    'elevated intracranial pressure': 4,
    'respiratory depression': 4,
    'liver damage': 3,
    'renal failure and hypertension': 4,
    'myopathy and rhabdomyolysis': 4,
    'orthostatic hypotension and syncope': 4,
    'gastrointestinal bleeding and peptic ulcer': 4,
    'thromboembolism': 4,
    "reye's syndrome": 4,
    'bleeding, nephrotoxicity, and gastrointestinal bleeding': 4,
    'gastrointestinal ulceration and gastrointestinal irritation': 3,
    'bleeding and thrombocytopenia': 4,
    'renal failure and hypotension': 4,
    'serotonin syndrome and seizure': 4,
    'lactic acidosis': 4,
    'congestive heart failure and hypotension': 4,
    'bleeding and gastrointestinal bleeding': 4,
    'serotonin syndrome and opioid toxicity': 4,
    'hypertension and cardiovascular complications': 4,
    'serotonin syndrome and hypomania': 4,
    'metabolic acidosis': 4,
    'gastrointestinal bleeding and thrombocytopenia': 4,
    'myocardial depression': 4,
    'congestive heart failure': 3,
    'cardiovascular complications': 4,
    'rhabdomyolysis': 4,
    'gastrointestinal bleeding and gastrointestinal ulceration': 4,
    'rhabdomyolysis, myoglobinuria, and elevated creatine kinase (cpk)': 4,
    'qtc prolongation, torsade de pointes, hypokalemia, hypomagnesemia, and cardiac arrest': 4,
    'confusion, irritability, and sleep disorders': 2,
    'qtc prolongation, ventricular arrhythmias, torsade de pointes, and convulsion': 4,
    'serotonin syndrome and neuroleptic malignant syndrome': 4,
    'generalized seizure': 4,
    'qtc prolongation and serotonin syndrome': 4,
    'generalized seizure and bradycardia': 4,
    'cardiac arrest': 4,
    'death': 4,
    'hyperkalemia and metabolic acidosis': 4,
    'rash, hypersensitivity reaction, stevens-johnson syndrome, and cutaneous drug reaction': 4,
    'stevens-johnson syndrome': 4,
    'seizure and encephalopathy': 4,
    'qtc prolongation, torsade de pointes, and cardiac arrhythmia': 4,
    'hemorrhage, gastrointestinal bleeding, and gastrointestinal ulceration': 4,
    'tumor lysis syndrome': 4,
    'cns depression': 3,
    'qtc prolongation': 3,
    'hyperkalemia': 3,
    'hypotension': 3,
    'dehydration': 3,
    'hypokalemia': 3,
    'infection': 3,
    'tendinopathy': 2,
    'thrombosis': 3,
    'neuromuscular blockade': 4,
    'cardiac arrhythmia': 3,
    'bradycardia': 3,
    'hypotension and cns depression': 3,
    'sedation, somnolence, and cns depression': 3,
    'tachycardia and drowsiness': 2,
    'neutropenia and thrombocytopenia': 3,
    'electrolyte imbalance': 2,
    'immunosuppression': 3,
    'nephrotoxicity and hypocalcemia': 3,
    'neuropsychiatric effects': 3,
    'myelosuppression': 3,
    'sedation and cns depression': 3,
    'qtc prolongation and torsade de pointes': 4,
    'hypotension and orthostatic hypotension': 3,
    'qtc prolongation and ventricular arrhythmias': 3,
    'cardiotoxicity': 3,
    'ventricular arrhythmias and cardiac arrhythmia': 3,
    'jaw osteonecrosis and anti-angiogenesis': 3,
    'pseudotumor cerebri': 3,
    'orthostatic hypotension and dizziness': 3,
    'ventricular arrhythmias, bradycardia, and heart block': 4,
    'gastrointestinal ulceration': 3,
    'hyperthermia and oligohydrosis': 3,
    'orthostatic hypotension': 3,
    'hypertension, hyponatremia, and water intoxication': 3,
    'ototoxicity and nephrotoxicity': 3,
    'hypersensitivity reaction': 3,
    'qtc prolongation, torsade de pointes, and cardiotoxicity': 4,
    'extrapyramidal symptoms and cns depression': 3,
    'hypotension, hyperkalemia, and nephrotoxicity': 3,
    'myelosuppression, anemia, and severe leukopenia': 4,
    'hypotension, nitritoid reactions, facial flushing, nausea, and vomiting': 3,
    'myopathy': 3,
    'hyponatremia and water intoxication': 3,
    'cns depression and hypotonia': 3,
    'hypertension and tardive dyskinesia': 3,
    'pulmonary toxicity': 3,
    'cardiac arrhythmia and cns stimulation': 3,
    'cytopenia': 3,
    'electrolyte abnormality': 2,
    'cardiovascular impairment': 3,
    'sedation and orthostatic hypotension': 3,
    'qtc prolongation and hypotension': 3,
    'bradycardia and heart block': 3,
    'hypotension, hyperkalemia, and reduced intravascular volume': 3,
    'granulocytopenia': 3,
    'hypotension and sinus node depression': 3,
    'hypotension and priapism': 3,
    'congestive heart failure, bleeding, hypotension, and tachycardia': 4,
    'hypertension and tachycardia': 3,
    'ototoxicity': 3,
    'heart failure': 3,
    'bronchospasm, shortness of breath, and dyspnea': 3,
    'hypercoagulability': 3,
    'hyperkinetic symptoms': 2,
    'torsade de pointes': 4,
    'hypotension and hyperkalemia': 3,
    'qtc prolongation and cardiac arrhythmia': 3,
    'anticonvulsant toxicity': 3,
    'torsade de pointes and cardiac arrhythmia': 4,
    'cytotoxicity': 3,
    'ventricular arrhythmias and torsade de pointes': 4,
    'drowsiness and cns depression': 3,
    'sinus node depression': 3,
    'jaw osteonecrosis': 3,
    'visual accommodation disturbances': 2,
    'sedation and extrapyramidal symptoms': 3,
    'water intoxication': 3,
    'ventricular arrhythmias': 3,
    'psychotic reaction': 3,
    'hemorrhagic cystitis': 3,
    'hypotension, bradycardia, and cardiac arrhythmia': 3,
    'hypertension': 2,
    'methemoglobinemia': 3,
    'tachycardia': 2,
    'hyperglycemia': 2,
    'gastrointestinal irritation': 2,
    'edema formation': 2,
    'angioedema': 3,
    'sedation': 2,
    'myopathy and weakness': 3,
    'extrapyramidal symptoms': 2,
    'neutropenia': 2,
    'constipation': 2,
    'hyponatremia': 2,
    'fluid retention': 2,
    'hypercalcemia': 2,
    'reduced gastrointestinal motility': 2,
    'sedation and somnolence': 2,
    'thrombocytopenia': 2,
    'urinary retention': 2,
    'peripheral neuropathy': 2,
    'liver enzyme elevations': 2,
    'hypocalcemia': 2,
    'urinary retention and constipation': 2,
    'urinary retention, reduced gastrointestinal motility, and constipation': 2,
    'ulceration': 2,
    'vasospastic reactions': 3,
    'weight gain and edema formation': 2,
    'osteomalacia': 2,
    'mucosal ulceration and ischemic colitis': 3,
    'anemia': 2,
    'hypotension, hyperglycemia, and hyperuricemia': 3,
    'increased transaminases': 2,
    'somnolence and peripheral neuropathy': 2,
    'hypothyroidism': 2,
    'hypertrichosis': 2,
    'gouty arthritis': 2,
    'increased glucose': 2,
    'hyperbilirubinemia': 2,
    'lymphopenia': 2,
    'leukopenia': 2,
    'rash': 2,
    'intraocular pressure': 2,
    'increased serum creatinine': 2,
}


def map_severity(description: str) -> int:
    """
    5级分类（0不由此函数返回，仅用于负样本）：
      1 = 一般：PK/PD机制变化（疗效/代谢/血药浓度/排泄/活性/吸收），无命名具体症状
      2 = 中度：DrugBank命名了具体症状，医生判定为中度
      3 = 严重：DrugBank命名了具体症状，医生判定为严重
      4 = 危重：DrugBank命名了具体症状，医生判定为危重
    """
    if not isinstance(description, str) or not description.strip():
        return 1

    # 泛泛 "adverse effects/events" → 级别1
    if _GENERIC_ADVERSE_RE.search(description):
        return 1

    # 无 "risk or severity of" → 级别1（PK/PD模板）
    if not _RISK_SEVERITY_RE.search(description):
        return 1

    # 提取症状短语并查表
    m = _EXTRACT_PHRASE_RE.search(description)
    if m:
        phrase = m.group(1).strip().lower()
        label = PHRASE_LABEL_MAP.get(phrase)
        if label is not None:
            return label

    # 未收录短语 → 保守归为严重(3)
    return 3


# ── 分子指纹 ─────────────────────────────────────────────────────────
def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if _MORGAN_GEN_INITED and radius == 2 and n_bits == 2048:
            return _MORGAN_GEN.GetFingerprintAsNumPy(mol)
        # 非默认参数时临时创建 generator
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return gen.GetFingerprintAsNumPy(mol)
    except Exception:
        return None


def build_fingerprint_cache(drugs_df: pd.DataFrame) -> dict:
    """为所有有 SMILES 的药物预计算 Morgan 指纹，返回 {drugbank_id: np.array}"""
    cache = {}
    print("计算分子指纹缓存...")
    for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df), desc="指纹计算"):
        if pd.notna(row.get("smiles")) and row["smiles"]:
            fp = smiles_to_fingerprint(row["smiles"])
            if fp is not None:
                cache[row["drugbank_id"]] = fp
    print(f"  成功: {len(cache)}/{len(drugs_df)} 个药物")
    return cache


# ── 药物对特征构建（v2：24维药理特征）──────────────────────────────────

# 24维药理特征名称，与向量下标一一对应，供 SHAP 解释使用
BIO_FEATURE_NAMES = [
    # CYP 酶 (8维)
    "CYP底物重叠率",           # 0  两药共享CYP底物酶占比
    "CYP共享底物(有/无)",       # 1  是否存在共享CYP底物
    "CYP底物-抑制剂交互",      # 2  A是某CYP底物 & B是该CYP抑制剂（或反之）
    "CYP底物-诱导剂交互",      # 3  A是某CYP底物 & B是该CYP诱导剂（或反之）
    "CYP抑制剂重叠率",         # 4  两药共同抑制同一CYP的比例
    "CYP诱导剂重叠率",         # 5  两药共同诱导同一CYP的比例
    "共享CYP酶总数",           # 6  不区分action的共享CYP酶个数
    "CYP任意交互(有/无)",      # 7  上述任一CYP交互是否存在
    # 转运体 (8维)
    "转运体底物重叠率",         # 8
    "转运体共享底物(有/无)",     # 9
    "转运体底物-抑制剂交互",    # 10
    "转运体底物-诱导剂交互",    # 11
    "P-gp共享(有/无)",         # 12 两药是否都涉及P-gp
    "P-gp底物-抑制剂交互",     # 13
    "OATP1B1共享(有/无)",      # 14
    "共享转运体总数",           # 15
    # 靶点 (8维)
    "靶点重叠Jaccard",         # 16
    "靶点共享(有/无)",          # 17
    "靶点激动剂重叠率",         # 18
    "靶点拮抗剂重叠率",         # 19
    "靶点抑制剂重叠率",         # 20
    "靶点激动-拮抗冲突",        # 21 A激动某靶点 & B拮抗该靶点（或反之）
    "靶点激动-抑制冲突",        # 22
    "共享靶点总数",             # 23
]

N_BIO = len(BIO_FEATURE_NAMES)  # 24
FEAT_DIM = 2048 * 2 + N_BIO     # 4120


def build_pair_features(id_a: str, id_b: str,
                         fp_cache: dict,
                         drugs_dict: dict) -> np.ndarray | None:
    """
    给一对药物计算 4120 维特征向量（v2）。

    输出向量布局：
      [0:2048]     fp_product（共有子结构）
      [2048:4096]  fp_diff（差异子结构）
      [4096:4120]  24维药理特征（CYP 8维 + 转运体 8维 + 靶点 8维）
    """
    fp_a = fp_cache.get(id_a)
    fp_b = fp_cache.get(id_b)
    if fp_a is None or fp_b is None:
        return None

    fp_product = fp_a * fp_b
    fp_diff    = np.abs(fp_a.astype(float) - fp_b.astype(float))

    drug_a = drugs_dict.get(id_a, {})
    drug_b = drugs_dict.get(id_b, {})

    bio = np.zeros(N_BIO, dtype=np.float32)

    # ── CYP 酶 (8维) ──
    enz_a = _parse_v2_field(drug_a.get("cyp_enzymes", ""))
    enz_b = _parse_v2_field(drug_b.get("cyp_enzymes", ""))

    cyp_sub_a = _filter_cyp(_names_by_action(enz_a, "substrate"))
    cyp_sub_b = _filter_cyp(_names_by_action(enz_b, "substrate"))
    cyp_inh_a = _filter_cyp(_names_by_action(enz_a, "inhibitor"))
    cyp_inh_b = _filter_cyp(_names_by_action(enz_b, "inhibitor"))
    cyp_ind_a = _filter_cyp(_names_by_action(enz_a, "inducer"))
    cyp_ind_b = _filter_cyp(_names_by_action(enz_b, "inducer"))
    cyp_all_a = _filter_cyp(_all_names(enz_a))
    cyp_all_b = _filter_cyp(_all_names(enz_b))

    shared_sub = cyp_sub_a & cyp_sub_b
    bio[0] = len(shared_sub) / max(len(cyp_sub_a | cyp_sub_b), 1)
    bio[1] = float(bool(shared_sub))
    # 底物-抑制剂交互：A的底物酶被B抑制，或B的底物酶被A抑制
    sub_inh = (cyp_sub_a & cyp_inh_b) | (cyp_sub_b & cyp_inh_a)
    bio[2] = float(bool(sub_inh))
    sub_ind = (cyp_sub_a & cyp_ind_b) | (cyp_sub_b & cyp_ind_a)
    bio[3] = float(bool(sub_ind))
    shared_inh = cyp_inh_a & cyp_inh_b
    bio[4] = len(shared_inh) / max(len(cyp_inh_a | cyp_inh_b), 1)
    shared_ind = cyp_ind_a & cyp_ind_b
    bio[5] = len(shared_ind) / max(len(cyp_ind_a | cyp_ind_b), 1)
    shared_cyp_all = cyp_all_a & cyp_all_b
    bio[6] = float(len(shared_cyp_all))
    bio[7] = float(bool(sub_inh or sub_ind or shared_sub))

    # ── 转运体 (8维) ──
    tp_a = _parse_v2_field(drug_a.get("transporters", ""))
    tp_b = _parse_v2_field(drug_b.get("transporters", ""))

    tp_sub_a = _names_by_action(tp_a, "substrate")
    tp_sub_b = _names_by_action(tp_b, "substrate")
    tp_inh_a = _names_by_action(tp_a, "inhibitor")
    tp_inh_b = _names_by_action(tp_b, "inhibitor")
    tp_ind_a = _names_by_action(tp_a, "inducer")
    tp_ind_b = _names_by_action(tp_b, "inducer")
    tp_all_a = _all_names(tp_a)
    tp_all_b = _all_names(tp_b)

    shared_tp_sub = tp_sub_a & tp_sub_b
    bio[8]  = len(shared_tp_sub) / max(len(tp_sub_a | tp_sub_b), 1)
    bio[9]  = float(bool(shared_tp_sub))
    tp_sub_inh = (tp_sub_a & tp_inh_b) | (tp_sub_b & tp_inh_a)
    bio[10] = float(bool(tp_sub_inh))
    tp_sub_ind = (tp_sub_a & tp_ind_b) | (tp_sub_b & tp_ind_a)
    bio[11] = float(bool(tp_sub_ind))
    # P-gp
    pgp_a = {n for n in tp_all_a if _has_pgp({n})}
    pgp_b = {n for n in tp_all_b if _has_pgp({n})}
    bio[12] = float(bool(pgp_a and pgp_b))
    pgp_sub_a = {n for n in tp_sub_a if _has_pgp({n})}
    pgp_sub_b = {n for n in tp_sub_b if _has_pgp({n})}
    pgp_inh_a = {n for n in tp_inh_a if _has_pgp({n})}
    pgp_inh_b = {n for n in tp_inh_b if _has_pgp({n})}
    bio[13] = float(bool((pgp_sub_a and pgp_inh_b) or (pgp_sub_b and pgp_inh_a)))
    # OATP1B1
    oatp_a = {n for n in tp_all_a if _has_oatp1b1({n})}
    oatp_b = {n for n in tp_all_b if _has_oatp1b1({n})}
    bio[14] = float(bool(oatp_a and oatp_b))
    shared_tp_all = tp_all_a & tp_all_b
    bio[15] = float(len(shared_tp_all))

    # ── 靶点 (8维) ──
    tgt_a = _parse_v2_field(drug_a.get("targets", ""))
    tgt_b = _parse_v2_field(drug_b.get("targets", ""))

    tgt_all_a = _all_names(tgt_a)
    tgt_all_b = _all_names(tgt_b)
    shared_tgt = tgt_all_a & tgt_all_b
    bio[16] = len(shared_tgt) / max(len(tgt_all_a | tgt_all_b), 1)
    bio[17] = float(bool(shared_tgt))

    tgt_ago_a = _names_by_action(tgt_a, "agonist")
    tgt_ago_b = _names_by_action(tgt_b, "agonist")
    tgt_ant_a = _names_by_action(tgt_a, "antagonist")
    tgt_ant_b = _names_by_action(tgt_b, "antagonist")
    tgt_inh_a = _names_by_action(tgt_a, "inhibitor")
    tgt_inh_b = _names_by_action(tgt_b, "inhibitor")

    shared_ago = tgt_ago_a & tgt_ago_b
    bio[18] = len(shared_ago) / max(len(tgt_ago_a | tgt_ago_b), 1)
    shared_ant = tgt_ant_a & tgt_ant_b
    bio[19] = len(shared_ant) / max(len(tgt_ant_a | tgt_ant_b), 1)
    shared_tinh = tgt_inh_a & tgt_inh_b
    bio[20] = len(shared_tinh) / max(len(tgt_inh_a | tgt_inh_b), 1)
    ago_ant = (tgt_ago_a & tgt_ant_b) | (tgt_ago_b & tgt_ant_a)
    bio[21] = float(bool(ago_ant))
    ago_inh = (tgt_ago_a & tgt_inh_b) | (tgt_ago_b & tgt_inh_a)
    bio[22] = float(bool(ago_inh))
    bio[23] = float(len(shared_tgt))

    return np.concatenate([fp_product, fp_diff, bio])


# ── 负样本构建 ───────────────────────────────────────────────────────
def build_negative_samples(drugs_df: pd.DataFrame,
                             positive_pairs: set,
                             fp_cache: dict,
                             n_samples: int,
                             tanimoto_threshold: float = 0.3) -> list:
    """
    策略：从全库药物中随机抽对，过滤条件：
      1. 不在已知 DDI 正样本集中
      2. Tanimoto 相似度 < 阈值（结构远离）
      3. 不共享 CYP 酶（若有数据）
      4. 不共享靶点（若有数据）
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    # 使用全库有指纹的药物作为候选（不再限呼吸系统）
    all_ids = [did for did in drugs_df["drugbank_id"] if did in fp_cache]
    drugs_dict = drugs_df.set_index("drugbank_id").to_dict("index")

    has_bio = (
        drugs_df["cyp_enzymes"].notna().any() and
        (drugs_df["cyp_enzymes"].astype(str) != "").any()
    ) or (
        "transporters" in drugs_df.columns and
        drugs_df["transporters"].notna().any()
    )

    rdkit_fps = {}
    print(f"预计算 RDKit 指纹对象（{len(all_ids)} 个药物）...")
    for did in tqdm(all_ids, desc="RDKit指纹"):
        smiles = drugs_dict[did].get("smiles", "") or ""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # GetFingerprint 返回 ExplicitBitVect，可直接用于 TanimotoSimilarity
                rdkit_fps[did] = _MORGAN_GEN.GetFingerprint(mol)
        except Exception:
            pass
    valid_ids = [i for i in all_ids if i in rdkit_fps]
    print(f"  可用候选: {len(valid_ids)} 个")

    negative_pairs = []
    attempts       = 0
    max_attempts   = n_samples * 15
    rng            = np.random.default_rng(42)

    pbar = tqdm(total=n_samples, desc="构建负样本")
    while len(negative_pairs) < n_samples and attempts < max_attempts:
        attempts += 1
        i, j   = rng.choice(len(valid_ids), size=2, replace=False)
        id_a, id_b = valid_ids[i], valid_ids[j]
        pair_key   = tuple(sorted([id_a, id_b]))
        if pair_key in positive_pairs:
            continue
        # 结构相似对排除：Tanimoto ≥ 阈值的药物对结构高度相似，
        # 存在未收录相互作用的可能性较高，不适合作为"安全"负样本。
        sim = DataStructs.TanimotoSimilarity(rdkit_fps[id_a], rdkit_fps[id_b])
        if sim >= tanimoto_threshold:
            continue

        # ★ 已知方法论缺陷（见 DATA_AUDIT_README.md 第五节）★
        # 以下两个过滤条件会导致所有负样本的 CYP 重叠率和靶点 Jaccard 值恒为 0，
        # 而这两者也是模型的输入特征，造成"类别0可通过特征值全零直接识别"的构造偏差。
        # 这属于负类构建策略缺陷，会虚高类别0指标，不属于训练/测试之间的数据泄漏。
        # 下一版本计划去掉此处过滤，改用纯随机负样本以获得更保守的评估结果。
        if has_bio:
            cyp_a = _all_names(_parse_v2_field(drugs_dict[id_a].get("cyp_enzymes", "")))
            cyp_b = _all_names(_parse_v2_field(drugs_dict[id_b].get("cyp_enzymes", "")))
            if cyp_a & cyp_b:
                continue
            tgt_a = _all_names(_parse_v2_field(drugs_dict[id_a].get("targets", "")))
            tgt_b = _all_names(_parse_v2_field(drugs_dict[id_b].get("targets", "")))
            if tgt_a & tgt_b:
                continue
        negative_pairs.append((id_a, id_b))
        pbar.update(1)
    pbar.close()
    print(f"  实际构建: {len(negative_pairs)} 对（尝试 {attempts} 次）")
    return negative_pairs


# ── 主预处理流程 ─────────────────────────────────────────────────────
def run_preprocessing(
    max_level1: int | None = None,  # None = 全量不截断；填整数则作为采样上限
    max_level2: int | None = None,
    max_level3: int | None = None,
    max_level4: int | None = None,
    neg_count:  int = 150_000,      # 负样本数量
    random_state: int = 42,
    use_memmap: bool = True,        # True=全量模式：特征写磁盘，训练时按需读取
                                    # 需约22GB磁盘空间（全量~128万样本×4120×4B）
):
    """
    Full-library DrugBank preprocessing, 5-tier clinician-reviewed labels.
    流程：
      1. 加载全部 DDI 记录并打标签（文本处理，内存可控）
      2. 去除 A-B/B-A 镜像重复
      3. 分层采样（默认全量，use_memmap=True避免OOM）
      4. 计算特征矩阵（写磁盘）
      5. 构建负样本并合并
    """
    rng = np.random.default_rng(random_state)

    # ── 1. 加载数据 ──────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("加载 DrugBank 数据...")
    drugs_df = pd.read_csv(RAW_DIR / "drugbank_drugs.csv")
    ddi_df   = pd.read_csv(RAW_DIR / "drugbank_ddi.csv")
    print(f"  药物总数: {len(drugs_df):,}，DDI 记录总数: {len(ddi_df):,}")

    # ── 2. 标签分配（全部 2.6M 条记录，仅根据文字描述） ────────────
    # 标签由 map_severity() 函数读取每条 DDI 的 description 字段确定，
    # Rules are defined by PHRASE_LABEL_MAP (clinician-reviewed, March 2026).
    # 这一步在 train/test 划分之前完成，且不引用任何特征数值或分布信息，
    # 因此标签与特征之间不存在信息泄漏。
    print("\n打标签（策略一：症状词检测）...")
    ddi_df["risk_level"] = ddi_df["description"].apply(map_severity)

    level_counts = ddi_df["risk_level"].value_counts().sort_index()
    print("  原始分布（含镜像重复）：")
    label_desc = {
        1: "级别1·一般(PK/PD机制变化)",
        2: "级别2·中度(具体轻症)",
        3: "级别3·严重(需干预)",
        4: "级别4·危重(生命威胁)",
    }
    for lvl, cnt in level_counts.items():
        print(f"    {label_desc.get(lvl, str(lvl))}: {cnt:,} ({cnt/len(ddi_df)*100:.1f}%)")

    # ── 2.5 去除镜像重复（A-B 与 B-A 视为同一对）────────────────────
    # DrugBank 在两种药物各自的记录页面都写入了一次 DDI，
    # 解析成 CSV 后每对相互作用会出现两行（drug1/drug2 位置互换）。
    # 两行的特征向量完全相同（因为指纹乘积和差值都是对称运算），
    # 保留其中一行即可，否则同一对样本在训练集和测试集中重复计数。
    print("\n去除镜像重复对（A-B == B-A）...")
    before_dedup = len(ddi_df)
    ddi_df["pair_key"] = ddi_df.apply(
        lambda r: tuple(sorted([r["drug1_id"], r["drug2_id"]])), axis=1
    )
    ddi_df = ddi_df.drop_duplicates(subset="pair_key").drop(columns="pair_key")
    ddi_df = ddi_df.reset_index(drop=True)
    after_dedup = len(ddi_df)
    print(f"  去重前: {before_dedup:,} 条 → 去重后: {after_dedup:,} 条（移除 {before_dedup-after_dedup:,} 条镜像重复）")

    level_counts = ddi_df["risk_level"].value_counts().sort_index()
    print("  去重后分布：")
    for lvl, cnt in level_counts.items():
        print(f"    {label_desc.get(lvl, str(lvl))}: {cnt:,} ({cnt/len(ddi_df)*100:.1f}%)")

    # ── 3. 分层采样（cap=None 则全量使用，填整数则截断到该上限）────
    parts = []
    for lvl, cap, name in [
        (1, max_level1, "级别1·一般"),
        (2, max_level2, "级别2·中度"),
        (3, max_level3, "级别3·严重"),
        (4, max_level4, "级别4·危重"),
    ]:
        subset = ddi_df[ddi_df["risk_level"] == lvl]
        orig = len(subset)
        if cap is not None and orig > cap:
            subset = subset.sample(n=cap, random_state=random_state)
            print(f"  {name} 采样至 {cap:,} 条（原 {orig:,} 条）")
        else:
            print(f"  {name}: 全量 {orig:,} 条")
        parts.append(subset)

    ddi_sampled = pd.concat(parts, ignore_index=True)
    print(f"\n  采样后正样本: {len(ddi_sampled):,} 条")

    # ── 4. 计算分子指纹缓存（全库一次性计算，快速） ────────────────
    fp_cache = build_fingerprint_cache(drugs_df)
    if not fp_cache:
        print("无有效 SMILES，请检查 drugbank_drugs.csv")
        return None, None

    drugs_dict = drugs_df.set_index("drugbank_id").to_dict("index")

    # ── 5. 建立已知正样本集合（用于负样本排重）─────────────────────
    # 负样本不能包含 DrugBank 中已有 DDI 记录的药物对，
    # 否则会把"有相互作用但未被采样进训练集"的样本标为"安全"，引入错误标签。
    # 这里用去重后的全量 ddi_df 来建集合，确保负样本排重使用的是完整正样本名单。
    print("\n构建正样本 ID 对集合（用于负样本排重）...")
    positive_pairs: set = set()
    for drug1, drug2 in tqdm(
        zip(ddi_df["drug1_id"], ddi_df["drug2_id"]),
        total=len(ddi_df), desc="正样本ID对"
    ):
        positive_pairs.add(tuple(sorted([drug1, drug2])))
    print(f"  已知正样本对: {len(positive_pairs):,}")

    # 释放 ddi_df（不再需要，节省 2-3 GB 内存）
    del ddi_df
    import gc; gc.collect()

    # ── 6. 计算正样本特征 ────────────────────────────────────────
    # 内存优化：预分配 numpy 数组原地写入，避免 Python list 转 array 的峰值双倍内存
    # 对比：list 方式峰值 ~28GB；预分配方式峰值 ~18GB（32GB 机器可安全运行）
    # FEAT_DIM 使用模块级定义（4120 = 2048×2 + 24维药理特征）
    n_total_est = len(ddi_sampled) + neg_count  # 上界（部分缺 SMILES 会略少）

    if use_memmap:
        mm_path = PROCESSED_DIR / "X_full.dat"
        print(f"\n[全量模式] 创建 memmap 文件：{mm_path}")
        print(f"  预估大小：{n_total_est * FEAT_DIM * 4 / 1e9:.1f} GB（磁盘）")
        X_buf = np.memmap(mm_path, dtype="float32", mode="w+",
                          shape=(n_total_est, FEAT_DIM))
    else:
        print(f"\n预分配特征矩阵（上界 {n_total_est:,} × {FEAT_DIM}，"
              f"约 {n_total_est * FEAT_DIM * 4 / 1e9:.1f} GB）...")
        X_buf = np.empty((n_total_est, FEAT_DIM), dtype=np.float32)

    y_buf    = np.empty(n_total_est, dtype=np.int32)
    id1_buf: list[str] = []   # drug1_id for each valid row
    id2_buf: list[str] = []   # drug2_id for each valid row
    ptr = 0

    print(f"构建正样本特征矩阵（{len(ddi_sampled):,} 条）...")
    skipped = 0
    for _, row in tqdm(ddi_sampled.iterrows(), total=len(ddi_sampled),
                       desc="正样本特征"):
        feat = build_pair_features(row["drug1_id"], row["drug2_id"],
                                   fp_cache, drugs_dict)
        if feat is not None:
            X_buf[ptr] = feat
            y_buf[ptr] = int(row["risk_level"])
            id1_buf.append(row["drug1_id"])
            id2_buf.append(row["drug2_id"])
            ptr += 1
        else:
            skipped += 1
    print(f"  有效正样本: {ptr:,}，跳过（缺SMILES）: {skipped:,}")
    n_pos = ptr

    if n_pos == 0:
        print("❌ 正样本特征为空")
        return None, None

    # 释放采样 DataFrame（不再需要）
    del ddi_sampled
    gc.collect()

    # ── 7. 负样本构建 ────────────────────────────────────────────
    neg_pairs = build_negative_samples(
        drugs_df, positive_pairs, fp_cache,
        n_samples=neg_count, tanimoto_threshold=0.3,
    )
    # positive_pairs 使用完毕，释放内存（约 500MB）
    del positive_pairs
    gc.collect()

    print("构建负样本特征...")
    n_neg = 0
    for id_a, id_b in tqdm(neg_pairs, desc="负样本特征"):
        feat = build_pair_features(id_a, id_b, fp_cache, drugs_dict)
        if feat is not None:
            X_buf[ptr] = feat
            y_buf[ptr] = 0
            id1_buf.append(id_a)
            id2_buf.append(id_b)
            ptr += 1
            n_neg += 1
    print(f"  有效负样本: {n_neg:,}")

    # ── 8. 截断到实际大小并保存 ──────────────────────────────────
    total_actual = ptr
    label_names = {
        0: "安全·无已知相互作用",
        1: "一般·PK/PD机制变化",
        2: "中度·具体症状(轻)",
        3: "严重·具体症状(需干预)",
        4: "危重·具体症状(生命威胁)",
    }

    if use_memmap:
        X = X_buf  # memmap 无需截断（截断由 y 的长度控制）
        y = y_buf[:total_actual].copy()
        np.save(PROCESSED_DIR / "y.npy", y)
        # 删除旧的 X.npy，防止 baseline_xgboost.py 的 load_data() 加载过期文件
        old_x = PROCESSED_DIR / "X.npy"
        if old_x.exists():
            old_x.unlink()
            print("  [清理] 已删除旧 X.npy，将使用 X_full.dat")
        size_gb = total_actual * FEAT_DIM * 4 / 1e9
        print(f"\n[全量模式] X 写入磁盘: {mm_path}，形状: ({total_actual}, {FEAT_DIM})")
        print(f"  实际占用磁盘: {size_gb:.1f} GB")
    else:
        # 截断（只复制实际用到的行，释放多余预分配空间）
        X = X_buf[:total_actual].copy()
        y = y_buf[:total_actual].copy()
        del X_buf, y_buf
        gc.collect()
        np.save(PROCESSED_DIR / "X.npy", X)
        np.save(PROCESSED_DIR / "y.npy", y)

    counts = np.bincount(y, minlength=5)
    meta = {
        "n_classes": 5,
        "label_remap": {},
        "label_names": {str(k): v for k, v in label_names.items()},
        "feature_dim": int(FEAT_DIM),
        "n_samples": int(total_actual),   # 用实际行数，而非预分配缓冲区大小
        "strategy": "5-tier: DrugBank symptom phrases + clinician review (2026-03)",
        "sampling": {
            "max_level1": max_level1,
            "max_level2": max_level2,
            "max_level3": max_level3,
            "max_level4": max_level4,
            "neg_count": neg_count,
        },
    }
    with open(PROCESSED_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存药物对 ID（与 X 每行一一对应，供 SHAP 药对可视化使用）
    pairs_path = PROCESSED_DIR / "pairs.csv"
    pd.DataFrame({"drug1_id": id1_buf, "drug2_id": id2_buf}).to_csv(
        pairs_path, index=False
    )
    print(f"  [保存] 药物对映射 → {pairs_path}  ({total_actual:,} 行)")

    print(f"\n{'='*55}")
    print(f"[完成] 有效样本: {total_actual:,}，特征维度: {FEAT_DIM}")
    print("各类别样本数:")
    for i, cnt in enumerate(counts):
        print(f"  类别{i} {label_names[i]}: {cnt:,} ({cnt/len(y)*100:.1f}%)")
    print(f"指纹维度: {FEAT_DIM - N_BIO} 维 + {N_BIO} 维药理特征")
    print(f"{'='*55}")
    return X, y



if __name__ == "__main__":
    run_preprocessing(
        use_memmap=True,      # 特征写磁盘，避免OOM（全量~128万样本需~21GB）
        neg_count=150_000,    # 负样本略增，与正样本比例更均衡
    )
