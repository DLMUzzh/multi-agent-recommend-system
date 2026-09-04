"""本地开发使用的安全、显式 DeepSeek 连通性检查。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage

from app.infrastructure.llm.client import create_chat_llm, public_llm_config


async def check_connection() -> tuple[dict[str, Any], int]:
    """返回不含密钥的连通性报告和进程退出码。"""

    report = public_llm_config()
    report["reachable"] = False
    report["latency_ms"] = None
    if not report["configured"]:
        return report, 2

    started = time.perf_counter()
    llm: Any | None = None
    try:
        llm = create_chat_llm(
            temperature=0.0,
            max_tokens=8,
            enable_llm=True,
        )
        if llm is None:
            return report, 2
        response = await llm.ainvoke(
            [HumanMessage(content="Connectivity check. Reply with OK only.")]
        )
        content = getattr(response, "content", "")
        report["reachable"] = bool(str(content).strip())
    except Exception:
        # 提供方异常可能包含请求详情，因此终端只报告连通性和耗时。
        report["reachable"] = False
    finally:
        close = getattr(llm, "aclose", None)
        if close is not None:
            await close()
        report["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)

    return report, 0 if report["reachable"] else 1


def main() -> None:
    """执行连通性检查并以退出码表示诊断结果。"""

    report, exit_code = asyncio.run(check_connection())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
