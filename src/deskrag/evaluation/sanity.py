"""WS4 (feat/eval-metrics): systems that test the harness itself.

- oracle: answers every answerable case with its reference answer and gold docs, abstains on the
  rest. It should score ~100% end-to-end; if not, the metrics or the labels are broken.
- always_abstain / never_abstain: the two degenerate policies. Reports show them as reference lines,
  so a real system has to beat both.
"""

from deskrag.core.config import SystemConfig
from deskrag.core.interfaces import RAGSystem
from deskrag.core.schemas import TestCase


def build_sanity_system(cfg: SystemConfig, cases: list[TestCase]) -> RAGSystem:
    raise NotImplementedError(f"WS4 (feat/eval-metrics): build the {cfg.system!r} sanity system")
