from src.llmconfig import get_llm
from ..prompts.prompt_plaintiff import *
from ..prompts.prompt_defendant import *
from ..state import *
from langchain.agents import create_agent
from pydantic import BaseModel,Field
from langgraph.types import interrupt,Command
from src.llm_wrapper import llm_wrapper

from langchain.agents.structured_output import ToolStrategy

class evidence_response_format(BaseModel):
    current_evidence : List[Evidence] | None = Field(description= "这里的取值有三种情况，分别对应三种举证模式")
    evidence_show_type : Evidence_Show_Enum = Field(description= "这是一个枚举类型，只有三种字符串取值，分别是“independent”，“union”,和“quit”")



class defense_evidence_format(BaseModel):
    current_evidence: list[Evidence] | Evidence
    messages: str


async def defense_objection(state: CourtState)-> CourtState:
    """
    【动态AI节点】这个节点被告诉讼代理人可以发表自己对于起诉状的异议（答辩意见），
    在当前模式中，被告诉讼代理人为AI.
    """
    defense_objection = await DEFENSE_OBJECTION.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages,
            "indictment" : state.meta.statement_claim
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_objection
    )
    reply = response.get("messages",[])[-1]
    reply.name = f"被告诉讼代理人{state.meta.defendant_agent}"
    return {
        "messages": [reply]
    }

async def defense_reply(state: CourtState)-> CourtState:
    """
    【动态AI节点】AI充当被告接受双方提问
    """
    defense_reply = await DEFENSE_REPLY.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_reply
    )
    reply = response.get("messages",[])[-1]
    reply.name = f"被告{state.meta.defendant_name}"
    return {
        "messages": [reply]
    }

def reply_control(state: CourtState)-> str:
    """
    【路由函数】检查状态参数中的pros_question_rounds,de_question_rounds,
    决定在被告发言结束后由哪一方发言
    """
    if state.pros_question_rounds == -1:
        return "defense_question"
    else:
        return "pros_question"

def de_round_control(state: CourtState) -> str:
    """
    【路由函数】通过state的一个临时参数de_question_rounds
    来控制提问人是谁
    """
    if state.de_question_rounds >=0:
        return "defense_reply"
    else:
        return "pros_summary"

async def defense_question(state: CourtState)-> CourtState:
    """
    【动态AI节点】如果还有提问次数，被告诉讼代理人向被告针对案件事实进行提问；如果没有，则进入问被告小结。
    """
    # 先判断当前剩余提问次数
    if state.de_question_rounds > 0:
        rounds = state.de_question_rounds - 1 # 扣除一次提问次数
        de_question = await DE_QUESTION.ainvoke(
            {
                "case_info" : state.meta,
                "messages" : state.messages
            }
        )
        defendant = create_agent(model=get_llm())
        response = await llm_wrapper.ainvoke_with_retry(
            defendant.ainvoke,
            de_question
        )
        question = response.get("messages",[])[-1]
        question.name = f"被告诉讼代理人{state.meta.defendant_agent}"
        return {
            "messages" : [question],
            "de_question_rounds" : rounds
        }
    else:
        return {
            "messages" : [AIMessage(content= "合议庭，被告诉讼代理人对被告的提问到此结束。",
                                    name= f"被告诉讼代理人{state.meta.defendant_agent}")],
            "de_question_rounds" : -1   # 返回一个-1，以便条件边（>=0)成功路由
        }

async def defense_summary(state: CourtState)-> CourtState:
    """【动态AI节点】被告诉讼代理人根据双方提问进行问被告小结"""
    de_summary = await DE_SUMMARY.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        de_summary
    )
    summary = response.get("messages",[])[-1]
    summary.name = f"被告诉讼代理人{state.meta.defendant_agent}"
    return {
        "messages" : [summary]
    }

async def defense_cross(state: CourtState)-> CourtState:
    """
    【动态AI节点】在原告提出证据时，被告诉讼代理人需要对证据进行质证
    """
    de_cross = await DE_CROSS.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages,
            "current_evidence": state.current_evidence
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        de_cross
    )
    cross = response.get("messages",[])[-1]
    cross.name = f"被告诉讼代理人{state.meta.defendant_agent}"
    return {
        "messages": [cross]
    }

async def defense_evidence_control(state: CourtState)-> Command:
    """
    【路由节点】被告诉讼代理人根据场上情形，在正式举证之前要先做一步判断：
    是应当单一证据举证、还是联合举证、还是放弃举证？
    放弃举证的话直接跳转到审判长宣布开始法庭辩论，进行举证则跳转到被告诉讼代理人宣读证据
    """
    evidence_decision = await DE_EVIDENCE_DECISION.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages,
            "evidence_list" : state.evidence_list
        }
    )
    defendant = create_agent(model=get_llm(),response_format=ToolStrategy(evidence_response_format))
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        evidence_decision
    )
    decision: evidence_response_format = response["structured_response"]
    if decision.evidence_show_type == "quit" or state.de_evidence_rounds <= 0:
        return Command(update={
            "current_evidence": decision.current_evidence,
            "evidence_show_type": decision.evidence_show_type,
            "messages": AIMessage(content= "感谢合议庭，被告诉讼代理人的补充举证到此结束。",
                                  name= f"被告诉讼代理人{state.meta.defendant_agent}")
        },
        goto= "judge_start_debate")
    else:
        return Command(
            update={
            "current_evidence": decision.current_evidence,
            "evidence_show_type": decision.evidence_show_type
            },
            goto= "defense_show_evidence"
        )

async def defense_show_evidence(state: CourtState) -> CourtState:
    """
    【动态AI节点】被告诉讼代理人根据前序的决定，宣读证据并作说明，后续由原告诉讼代理人质证
    """
    show_evidence = await DE_SHOW_EVIDENCE.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages,
            "current_evidence": state.current_evidence
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        show_evidence
    )
    show = response.get("messages",[])[-1]
    show.name = f"被告诉讼代理人{state.meta.defendant_agent}"
    rounds = state.de_evidence_rounds - 1
    return {
        "messages": [show],
        "de_evidence_rounds": rounds
    }

async def defense_self_statement(state: CourtState)-> CourtState:
    """
    【动态AI节点】被告自行辩论
    """
    defense_self_statement = await DEFENSE_SELF_STATEMENT.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model= get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_self_statement
    )
    statement = response.get("messages",[])[-1]
    statement.name = f"被告{state.meta.defendant_name}"
    return {
        "messages": [statement]
    }

async def defense_statement(state: CourtState)-> CourtState:
    """
    【动态AI节点】被告诉讼代理人进行辩论意见的发表
    """
    defense_statement = await DEFENSE_STATEMENT.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_statement
    )
    statement = response.get("messages",[])[-1]
    statement.name = f"被告诉讼代理人{state.meta.defendant_agent}"
    return {
        "messages" : [statement]
    }

async def defense_focus(state: CourtState)-> CourtState:
    """
    【动态AI节点】被告诉讼代理人对争议焦点的回应
    """
    current_focus = state.focus[state.focus_index]
    defense_focus = await DEFENSE_FOCUS.ainvoke(
        {
            "case_info": state.meta,
            "messages": state.messages,
            "current_focus": current_focus
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_focus
    )
    focus = response.get("messages",[])[-1]
    focus.name = f"被告诉讼代理人{state.meta.defendant_agent}"
    return {
        "messages": [focus],
        "pros_focus_rounds": state.pros_focus_rounds - 1
    }

async def defense_sumup(state: CourtState)-> CourtState:
    """
    【动态AI节点】被告诉讼代理人作总结的辩论意见发表
    """
    defense_sumup = await DEFENSE_SUMUP.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_sumup
    )
    sumup = response.get("messages",[])[-1]
    sumup.name = f"被告诉讼代理人{state.meta.defendant_agent}"
    return {
        "messages" : [sumup]
    }

async def defense_final_statement(state: CourtState)-> CourtState:
    """
    【动态AI节点】由被告作最终陈述
    """
    defense_final_statement = await DEFENSE_FINAL_STATEMENT.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model=get_llm())
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_final_statement
    )
    sumup = response.get("messages",[])[-1]
    sumup.name = f"被告{state.meta.defendant_name}"
    return {
        "messages" : [sumup]
    }
