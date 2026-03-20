"""Thompson Sampling Multi-Armed Bandit for adaptive search budget allocation.

每个"臂"对应一个搜索源（DuckDuckGo / ArXiv / Tavily）。
系统维护每个臂的 Beta(α, β) 分布，通过观测到的 BGE reranker 得分更新参数，
从而在多轮迭代中自动将预算向高质量信源倾斜。

使用方式
--------
mab = ThompsonSamplingMAB()                     # 新建（均匀先验）
mab = ThompsonSamplingMAB.from_dict(state_dict) # 从状态恢复

base = {"duckduckgo": 35, "arxiv": 20, "tavily": 5}
adjusted = mab.allocate_budgets(base)           # 本轮预算分配

# ... 执行检索 + BGE rerank ...

rewards = compute_source_rewards(reranked_items, adjusted)
mab.update(rewards)                             # 更新 Beta 参数

state["mab_state"] = mab.to_dict()             # 持久化到 ResearchState
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# 三个搜索臂的名称，须与 search_tool.py / arxiv_tool.py 中的 "source" 字段一致
ARMS: List[str] = ["duckduckgo", "arxiv", "tavily"]

# 每个臂的最小保留预算（避免某臂被完全饿死，丧失探索机会）
_MIN_BUDGET: Dict[str, int] = {"duckduckgo": 3, "arxiv": 3, "tavily": 0}

# 每个臂最大允许的预算倍率（不超过基础预算的 N 倍，防止单臂过度占用）
_MAX_MULTIPLIER: float = 3.0

# Thompson Sampling 虚拟试验次数：将连续奖励转换为 Beta 更新量
_N_VIRTUAL_TRIALS: int = 10


class ThompsonSamplingMAB:
    """Thompson Sampling 多臂赌博机，用于搜索源预算自适应分配。

    原理
    ----
    - 每个臂维护 Beta(α, β) 分布，α/(α+β) 为期望奖励。
    - 每轮从各臂的 Beta 分布中采样 θ_i，归一化得到权重 w_i。
    - 按权重将总预算池重新分配给三个搜索源。
    - 用实际观测到的 BGE reranker 均分更新 Beta 参数。
    - 奖励高的臂 α 增大 → 后续被采样概率更高 → 获得更多预算。
    """

    ARMS: List[str] = ARMS

    def __init__(self, alpha_init: float = 1.0, beta_init: float = 1.0) -> None:
        # Beta 分布参数，初始为均匀先验 Beta(1,1)
        self.alpha: Dict[str, float] = {arm: alpha_init for arm in self.ARMS}
        self.beta: Dict[str, float] = {arm: beta_init for arm in self.ARMS}
        self.round: int = 0
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 预算分配
    # ------------------------------------------------------------------

    def sample_weights(self) -> Dict[str, float]:
        """从每个臂的 Beta 分布采样并归一化为分配权重。"""
        samples: Dict[str, float] = {}
        for arm in self.ARMS:
            samples[arm] = random.betavariate(self.alpha[arm], self.beta[arm])
        total = sum(samples.values())
        if total == 0.0:
            return {arm: 1.0 / len(self.ARMS) for arm in self.ARMS}
        return {arm: v / total for arm, v in samples.items()}

    def allocate_budgets(
        self,
        base_budgets: Dict[str, int],
        min_budgets: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """将总预算池按 Thompson Sampling 权重重新分配。

        参数
        ----
        base_budgets : 环境变量中配置的基础预算 {"duckduckgo": 35, ...}
        min_budgets  : 各臂的最小保留量（默认 _MIN_BUDGET）

        返回
        ----
        adjusted_budgets : 每个臂实际使用的预算
        """
        mins = min_budgets if min_budgets is not None else _MIN_BUDGET
        total_pool = sum(base_budgets.values())
        if total_pool == 0:
            return {arm: 0 for arm in self.ARMS}

        weights = self.sample_weights()
        allocated: Dict[str, int] = {}

        arms = list(self.ARMS)
        for i, arm in enumerate(arms):
            base = base_budgets.get(arm, 0)
            if base == 0:
                # 该源被用户明确禁用（基础预算=0），跳过
                allocated[arm] = 0
                continue

            # 按权重从总池中分配
            raw = round(total_pool * weights[arm])
            lo = mins.get(arm, 0)
            hi = int(base * _MAX_MULTIPLIER)
            allocated[arm] = max(lo, min(hi, raw))

        return allocated

    # ------------------------------------------------------------------
    # 参数更新
    # ------------------------------------------------------------------

    def update(self, rewards: Dict[str, Optional[float]]) -> None:
        """根据本轮观测奖励更新 Beta 分布参数。

        参数
        ----
        rewards : {arm: mean_reranker_score (0~1)} 或 None（该臂本轮未查询）

        实现细节
        --------
        用虚拟试验模型将连续奖励 r 转换为 Beta 更新：
          - 成功次数 = round(r * N_VIRTUAL)
          - 失败次数 = N_VIRTUAL - 成功次数
        这等价于从一个参数为 r 的伯努利分布中观测了 N_VIRTUAL 次独立试验。
        """
        self.round += 1
        record: Dict[str, Any] = {"round": self.round, "rewards": {}, "params_after": {}}

        for arm in self.ARMS:
            r = rewards.get(arm)
            if r is None:
                record["rewards"][arm] = None
                record["params_after"][arm] = None
                continue

            r_clamped = max(0.0, min(1.0, float(r)))
            successes = round(r_clamped * _N_VIRTUAL_TRIALS)
            failures = _N_VIRTUAL_TRIALS - successes
            self.alpha[arm] += successes
            self.beta[arm] += failures

            record["rewards"][arm] = round(r_clamped, 4)
            record["params_after"][arm] = {
                "alpha": round(self.alpha[arm], 2),
                "beta": round(self.beta[arm], 2),
                "expected_reward": round(
                    self.alpha[arm] / (self.alpha[arm] + self.beta[arm]), 4
                ),
            }

        self.history.append(record)

    # ------------------------------------------------------------------
    # 统计查询
    # ------------------------------------------------------------------

    def expected_rewards(self) -> Dict[str, float]:
        """返回各臂当前期望奖励 E[θ] = α/(α+β)。"""
        return {
            arm: round(self.alpha[arm] / (self.alpha[arm] + self.beta[arm]), 4)
            for arm in self.ARMS
        }

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "round": self.round,
            "expected_rewards": self.expected_rewards(),
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThompsonSamplingMAB":
        mab = cls()
        mab.alpha = {arm: float(d.get("alpha", {}).get(arm, 1.0)) for arm in cls.ARMS}
        mab.beta = {arm: float(d.get("beta", {}).get(arm, 1.0)) for arm in cls.ARMS}
        mab.round = int(d.get("round", 0))
        mab.history = list(d.get("history", []))
        return mab


# ------------------------------------------------------------------
# 奖励计算辅助函数（在 researcher_node 中调用）
# ------------------------------------------------------------------

def compute_source_rewards(
    reranked_items: List[Dict[str, Any]],
    allocated_budgets: Dict[str, int],
) -> Dict[str, Optional[float]]:
    """根据 BGE reranker 得分计算各搜索源的 MAB 奖励。

    逻辑
    ----
    - 对最终进入 top-k 的每条文档，按其 source 字段归属到对应的臂。
    - 奖励 = 该臂所有文档的 bge_reranker_score 均值。
    - 若某臂本轮未分配预算（=0）→ reward=None（不更新该臂参数）。
    - 若某臂有预算但无文档存活于 top-k → reward=0.0（实质惩罚）。

    返回
    ----
    {arm_name: reward_or_None}
    """
    source_scores: Dict[str, List[float]] = {arm: [] for arm in ARMS}

    for item in reranked_items:
        src = str(item.get("source", "")).lower()
        # source 字段已在 search_tool.py / arxiv_tool.py 中规范化，直接匹配
        if src in source_scores:
            score = float(
                item.get("bge_reranker_score", item.get("bge_retriever_score", 0.0))
            )
            source_scores[src].append(score)

    rewards: Dict[str, Optional[float]] = {}
    for arm in ARMS:
        budget = allocated_budgets.get(arm, 0)
        if budget == 0:
            rewards[arm] = None  # 未查询，跳过更新
        elif source_scores[arm]:
            rewards[arm] = sum(source_scores[arm]) / len(source_scores[arm])
        else:
            rewards[arm] = 0.0  # 有预算但无文档存活

    return rewards
