from langchain_core.messages import AIMessage
from ..prompts.prompt_pro import *
from ..state import *

async def clerk_rules(state:CourtState) -> CourtState:
    '''
    【静态节点】书记员宣读法庭规则，入庭报告
    '''
    return {
        "phase": PhaseEnum.OPENING, # 以防万一初始状态没有正确传入，兜底处理
        "messages" : [
            AIMessage(content = CLERK_RULES,
                        name=f"书记员{state.meta.clerk_name}"),
            AIMessage(content = CLERK_ANNOUNCEMENT,
                        name = f"书记员{state.meta.clerk_name}")
                    ]
        }
