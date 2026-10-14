"""民事诉讼流程最小可运行示例：民间借贷纠纷一审普通程序。

用法：
    .venv/bin/python -m src.civil.example_run
"""
import asyncio

from langchain_core.messages import HumanMessage

from src.civil.court_flow import app
from src.civil.state import (
    CourtState,
    PhaseEnum,
    ProviderEnum,
    Evidence,
    Evidence_Show_Enum,
    case_info,
)


def build_initial_state() -> CourtState:
    meta = case_info(
        abstract=(
            "2023年3月，被告张三以生意周转为由向原告李四借款人民币10万元，"
            "约定借期一年，年利率10%。原告通过银行转账交付借款，被告出具借条。"
            "借款到期后，被告仅归还2万元，剩余本金及利息未付，原告遂诉至法院。"
        ),
        cause_of_action="民间借贷纠纷",
        claims=(
            "一、请求判令被告偿还原告借款本金8万元；"
            "二、请求判令被告支付借款利息（以8万元为基数，按年利率10%，"
            "自2024年3月起计算至实际清偿之日止）；三、本案诉讼费由被告承担。"
        ),
        plaintiff_name="李四",
        plaintiff_agent="王律师",
        statement_claim=(
            "民事起诉状：原告李四诉称，2023年3月15日，被告张三以生意周转为由"
            "向原告借款人民币10万元，原告于当日通过银行转账交付全部借款，"
            "被告出具借条一张，载明借期一年，年利率10%。2024年3月借款到期后，"
            "经原告多次催要，被告仅于2024年5月归还2万元，其余本金8万元及利息"
            "至今未付。请求法院判如所请。"
        ),
        defendant_name="张三",
        defendant_residence="某市某区某街道1号",
        defendant_occupation="个体经营者",
        defendant_ID_number="110101199001011234",
        defendant_agent="赵律师",
        answer=(
            "被告辩称：对借款事实无异议，但已归还的2万元系本金，"
            "剩余8万元本金因生意亏损暂时无力偿还；借条中约定的年利率10%"
            "过高，请求法院依法调整。"
        ),
        court_name="某市某区人民法院",
        judge_name="陈法官",
        judge_name_2="刘陪审员",
        clerk_name="小周",
        case_id="（2026）某0101民初1234号",
    )

    evidence_list = [
        Evidence(
            id="E1",
            name="借条",
            content=(
                "2023年3月15日张三出具的借条：今借到李四人民币10万元整，"
                "借期一年，年利率10%，到期本息一并归还。借款人：张三。"
            ),
            provider=ProviderEnum.PLAINTIFF,
        ),
        Evidence(
            id="E2",
            name="银行转账记录",
            content=(
                "2023年3月15日李四名下尾号8888的银行账户向张三名下尾号6666的"
                "银行账户转账人民币10万元的交易明细。"
            ),
            provider=ProviderEnum.PLAINTIFF,
        ),
        Evidence(
            id="E3",
            name="微信催款聊天记录",
            content=(
                "2024年3月至2024年6月期间，李四多次通过微信向张三催要还款，"
                "张三回复称已还2万元，剩余款项尽快归还。"
            ),
            provider=ProviderEnum.PLAINTIFF,
        ),
        Evidence(
            id="E4",
            name="还款转账凭证",
            content=(
                "2024年5月20日张三向李四转账人民币2万元的银行回单，"
                "附言注明“还款”。"
            ),
            provider=ProviderEnum.DEFENDANT,
        ),
    ]

    return CourtState(
        messages=[HumanMessage(content="现在开庭。", name="system")],
        focus=[],
        phase=PhaseEnum.OPENING,
        evidence_list=evidence_list,
        current_evidence=None,
        evidence_show_type=Evidence_Show_Enum.QUIT,
        meta=meta,
    )


async def main() -> None:
    initial_state = build_initial_state()
    print(f"{initial_state}")


if __name__ == "__main__":
    asyncio.run(main())
