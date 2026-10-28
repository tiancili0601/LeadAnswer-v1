import base64
import binascii
import json
import os
import re
import secrets
import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from database.manager import DatabaseManager

# 从项目根目录的 .env 读取配置（.env 已被 .gitignore 忽略）
load_dotenv()

# ==========================================
# 核心配置（密钥只从环境变量 / .env 读取）
# ==========================================
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-flash").strip()

# 演示模式：无需登录，所有对话与诊断统一使用该学生档案
DEFAULT_STUDENT_ID = "demo_student"

# Base64 体积上限（约对应 6MB 原始图片），避免请求体过大导致网关超时
MAX_IMAGE_BASE64_LENGTH = 8 * 1024 * 1024
DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"
ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}

# 《材料力学》课程专用的多模态苏格拉底式导师提示词。
# 设计要点：课程边界硬约束 + 图像先行描述 + 禁止直接给答案/直接否定 + LaTeX + 单问引导。
SYSTEM_PROMPT = r"""你是一名精通《材料力学》的多模态 AI 助教，名字叫「力学引导者」，服务对象是正在学习材料力学的本科生。你的教学风格是苏格拉底式：不灌输结论，而是用证据和问题引导学生自己完成推理。

【一、课程聚焦（硬性边界）】
1. 你只回答《材料力学》及其直接先修内容（静力学平衡、几何关系、材料基本假定）。覆盖范围：受力分析与隔离体、轴力/剪力/弯矩图、应力应变与胡克定律、扭转、平面弯曲、强度理论与组合变形、梁的位移与刚度、压杆稳定。
2. 不得回答非本课程问题：编程、外语、其他学科、闲聊、娱乐、考试作弊、直接索要作业成品等，一律用一句话说明职责并引回材料力学，不展开任何题外内容。
3. 不编造学生未提供的载荷、尺寸、支座类型或材料参数。信息不足时，先问清楚再分析。

【二、视觉识别：先客观描述，再提问】
当学生上传图片（手写受力图、剪力图、弯矩图、公式截图等）时：
1. 第一段必须先用一两句话客观描述你看到的内容，只陈述可见事实，不做对错判断。示例：「我看到了你画的简支梁受力图，包含两个支座反力和一个集中力偶。」
2. 如果图像模糊、遮挡、裁切或关键标注缺失，明确指出看不清的位置，请学生补拍或补充，绝不凭空猜测。
3. 描述之后，从图中可见证据里挑选最关键的一处，提出一个引导性问题。

【三、苏格拉底式引导：禁止直接给答案，禁止直接否定】
1. 严禁给出最终数值、完整解题步骤或成稿的受力图/剪力图/弯矩图。
2. 严禁直接说「你错了」「不对」「错误」「答案是」。发现图像或推导有误时，必须用「中性描述 + 图中/文本证据 + 一个开放问题」引导学生自己发现。
   示例：「我注意到你在 A 点画的约束反力是向上的，结合你画的集中力偶，能再列一下 A 点的力矩平衡方程吗？」
3. 每轮只提出一个核心问题，问题必须指向具体位置、符号、方程或图线，避免连环追问。
4. 学生答对时，追问依据或迁移场景；学生卡住时，缩小问题粒度或给出提示性反问，但不要代答。
5. 表扬必须具体（指出哪一步推理有效），禁止空泛鼓励。

【四、公式与排版】
1. 所有数学公式必须使用 LaTeX：行内用 $...$，独立公式用 $$...$$。示例：$\sigma = \frac{F}{A}$，$\frac{dM}{dx}=F_S$。
2. 符号遵循材料力学惯例：轴力 $F_N$、剪力 $F_S$、弯矩 $M$、正应力 $\sigma$、切应力 $\tau$、应变 $\varepsilon$、弹性模量 $E$。
3. 使用 Markdown 排版；每次回复一般控制在 2–5 句话，并以一个引导性问题结束。

【五、诊断协同】
你的可见回复只输出自然语言，严禁输出 JSON。系统会在每轮对话结束后由独立诊断 Agent 提取结构化诊断。你只需围绕本轮最关键的薄弱点提问，使学生暴露出可被诊断的证据。"""

# 独立的学情诊断 Agent 提示词。与导师回复分离，保证课堂对话体验不被 JSON 打断。
DIAGNOSIS_SYSTEM_PROMPT = """你是一名《材料力学》学情诊断专家。你的唯一任务是依据本轮「学生输入 + 图片（可选）+ 助教回复」提取可验证的学情诊断，并严格输出 JSON 对象。

诊断原则：
1. 只诊断《材料力学》范围内的薄弱点；若本轮没有可核查的薄弱点，四个字段全部输出空字符串，不得为了填充而编造。
2. diagnosis 必须具体到概念、规则或步骤，例如「未掌握剪力图在集中力作用处的突变规则」「遗漏均布载荷对弯矩图的二次项影响」；禁止写「基础差」「不认真」等空泛评价。
3. evidence 必须是对学生文本的可核查引用或概括，例如「学生认为集中力偶两侧剪力会发生突变」；没有文本证据则留空。
4. visual_evidence 仅在提供了图片且图中确有可核查问题时填写，例如「上传的受力图中缺少固定端的反力偶」；没有图片或图中无问题则必须留空。
5. suggestion 是针对该薄弱点的下一步行动指令，必须具体可执行，例如「帮我出两道带有集中力偶的简支梁剪力图练习题」；无薄弱点时留空。
6. 只输出 JSON 对象，键固定为 diagnosis、evidence、visual_evidence、suggestion，值均为字符串。不要输出 Markdown 代码块、前后缀说明或任何额外文字。"""

# 诊断结果的唯一合法结构（无弱点时所有字段为空字符串）
DIAGNOSIS_JSON_SCHEMA = {
    "diagnosis": "",
    "evidence": "",
    "visual_evidence": "",
    "suggestion": "",
}

DIAGNOSIS_FIELDS = tuple(DIAGNOSIS_JSON_SCHEMA.keys())

# 会话结束时的汇总诊断提示词：基于完整对话历史提取本次学习的代表性薄弱点
PROFILER_PROMPT_TEMPLATE = """请分析以下《材料力学》师生对话历史，提取本次学习中最值得记录的一条学情诊断。

【对话历史】
{chat_content}

【输出要求】
严格输出 JSON 对象，且只包含以下四个键，值均为字符串；若无法从对话中确认某项信息，则该项输出空字符串：
{{
  "diagnosis": "具体的薄弱点，例如：未掌握剪力图突变规则、遗漏均布载荷",
  "evidence": "对话中的文本证据，例如：学生回复的推导过程错误",
  "visual_evidence": "视觉证据，例如：上传的受力图中缺少固定端反力偶（历史无图片时留空）",
  "suggestion": "下一步行动指令，例如：帮我出两道带有集中力偶的简支梁剪力图练习题"
}}

注意：不得编造对话中不存在的信息，不得输出 JSON 以外的任何文字。"""

# 会话结束汇总诊断的 system prompt（与单轮诊断共用四字段结构）
PROFILER_SYSTEM_PROMPT = """你是一名《材料力学》学情诊断专家，擅长从完整师生对话中识别可验证的薄弱点。你只输出符合指定四字段结构的 JSON 对象，不输出任何解释性文字。"""

# 学情报告页专用：根据诊断结果生成 3–5 道针对性样卷
QUIZ_SYSTEM_PROMPT = """你是《材料力学》课程的命题教师。请根据学生的学情诊断结果，生成一份针对性样卷。

命题要求：
1. 题目数量为 3–5 道，难度递进，必须覆盖 diagnosis 指出的薄弱点。
2. 题型可包含判断题、简答题、计算题、作图题；每题给出完整参数与作答要求。
3. 所有公式使用 LaTeX；题目应能独立作答。
4. 只输出一个 JSON 对象，键固定为 quiz_markdown 与 answer_markdown，值均为字符串。
5. quiz_markdown 只包含题目、分值与作答要求；answer_markdown 包含简要解答与评分要点。
6. 不要输出 JSON 以外的任何文字，不要使用 Markdown 代码围栏。"""


def parse_image_payload(
    image_base64: Optional[str],
    image_mime_type: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    清洗并校验前端传入的图片数据。

    支持两种写法：
    1. 纯 Base64 字符串 + image_mime_type；
    2. 完整 Data URL（data:image/png;base64,...），此时自动解析 MIME。

    返回 (清洗后的 Base64 或 None, MIME 类型)。
    校验失败时抛出 ValueError，由接口层转换为 400/413。
    """
    mime_type = (image_mime_type or DEFAULT_IMAGE_MIME_TYPE).strip().lower()
    if mime_type == "image/jpg":
        mime_type = DEFAULT_IMAGE_MIME_TYPE

    if not image_base64 or not image_base64.strip():
        return None, mime_type

    value = image_base64.strip()
    if value.startswith("data:"):
        header, separator, payload = value.partition(",")
        if not separator:
            raise ValueError("image_base64 的 Data URL 格式不合法")
        if ";base64" not in header.lower():
            raise ValueError("仅支持 Base64 编码的图片 Data URL")
        header_mime = header[5:].split(";")[0].strip().lower()
        if header_mime:
            mime_type = DEFAULT_IMAGE_MIME_TYPE if header_mime == "image/jpg" else header_mime
        value = payload

    # 去除换行、空格等传输过程中可能引入的空白字符
    value = "".join(value.split())

    if not value:
        return None, mime_type
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError(f"不支持的图片类型: {mime_type}，请使用 JPEG/PNG/WebP/GIF")
    if len(value) > MAX_IMAGE_BASE64_LENGTH:
        raise ValueError("图片体积过大，请压缩到约 6MB 以内后重试")

    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image_base64 不是合法的 Base64 编码")

    return value, mime_type


def build_user_content(
    user_text: str,
    image_base64: Optional[str] = None,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
) -> Union[str, List[Dict[str, Any]]]:
    """
    构造 OpenAI 多模态 messages 中的 user.content。

    - 纯文本：直接返回字符串，保持与原有接口完全兼容；
    - 含图片：返回 text + image_url 的内容块列表。
    """
    text = (user_text or "").strip()

    if not image_base64:
        return text

    if not text:
        text = "这是我上传的《材料力学》作业图片，请先描述你看到的内容，再开始提问。"

    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
        },
    ]


def build_chat_messages(
    user_text: str,
    image_base64: Optional[str] = None,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
) -> List[Dict[str, Any]]:
    """构造本轮对话的完整 messages（system + user）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(user_text, image_base64, image_mime_type),
        },
    ]


def history_to_text(history: List[dict]) -> str:
    """
    将前端回传的对话历史扁平化为纯文本，供 Profiler Agent 诊断。

    多模态消息的 content 是内容块列表，这里把图片替换为占位符，
    避免把 Base64 数据写入提示词、撑爆上下文或污染诊断结果。
    """
    lines: List[str] = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown")
        content = message.get("content")

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    parts.append("[学生上传了一张图片]")
            content = "\n".join(part for part in parts if part)

        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# 诊断输入中允许拼接的历史文本上限，避免提示词过长
MAX_DIAGNOSIS_HISTORY_CHARS = 6000
MAX_DIAGNOSIS_FIELD_LENGTH = 1200


def empty_diagnosis() -> Dict[str, str]:
    """返回一份四字段为空的诊断结果（无弱点时的标准结构）。"""
    return dict(DIAGNOSIS_JSON_SCHEMA)


def has_diagnosis_signal(diagnosis: Dict[str, str]) -> bool:
    """判断诊断中是否存在可记录的有效信息。"""
    return any(str(diagnosis.get(field, "")).strip() for field in DIAGNOSIS_FIELDS)


def parse_json_object(raw: str) -> Dict[str, Any]:
    """
    稳健地解析模型返回的 JSON，防止多余字符导致解析崩溃。

    依次尝试：
    1. 去掉 Markdown 代码围栏后直接 json.loads；
    2. 用正则提取最外层 {...} 后再解析；
    3. 用 JSONDecoder.raw_decode 从第一个 { 开始解析，忽略尾随文字。

    三种方式都失败时抛 ValueError，由接口层转成可读的错误响应。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("模型返回了空内容")

    # 1) 去掉可能存在的 ```json / ``` 围栏
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # 2) 直接解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3) 正则兜底：提取最外层 JSON 对象（DOTALL 允许跨行）
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            # 4) 最后一个兜底：从第一个 { 开始做增量解析，自动忽略尾部说明文字
            start = candidate.find("{")
            if start != -1:
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

    raise ValueError("模型返回内容中不包含合法 JSON 对象")


def normalize_diagnosis(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    将模型输出规整为固定的四字段结构。

    - 只保留 diagnosis / evidence / visual_evidence / suggestion；
    - 所有值强制为去空白的字符串，缺失或非字符串值转为空字符串；
    - 兼容少量常见别名，降低模型偶发改键名导致的解析失败。
    """
    if not isinstance(payload, dict):
        return empty_diagnosis()

    aliases = {
        "diagnosis": ("diagnosis", "weakness", "weak_point"),
        "evidence": ("evidence", "text_evidence"),
        "visual_evidence": ("visual_evidence", "image_evidence"),
        "suggestion": ("suggestion", "action", "next_action"),
    }

    normalized = empty_diagnosis()
    for field, keys in aliases.items():
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            value = value.strip()
            if value:
                normalized[field] = value[:MAX_DIAGNOSIS_FIELD_LENGTH]
                break
    return normalized


def build_diagnosis_messages(
    student_text: str,
    tutor_reply: str,
    image_base64: Optional[str] = None,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
    history_text: str = "",
) -> List[Dict[str, Any]]:
    """
    构造诊断 Agent 的多模态 messages。

    图片会随本轮学生输入一起送入诊断模型，用于提取 visual_evidence；
    历史文本仅截取末尾一段，控制上下文长度。
    """
    sections = [
        "【本轮学生输入】",
        (student_text or "").strip() or "（本轮学生没有文字输入，仅上传了图片）",
        "",
        "【助教本轮回复】",
        (tutor_reply or "").strip() or "（无）",
    ]

    if history_text.strip():
        sections.extend([
            "",
            "【此前对话摘要（供判断连续性，可忽略）】",
            history_text.strip()[-MAX_DIAGNOSIS_HISTORY_CHARS:],
        ])

    sections.extend(["", "请按系统要求输出诊断 JSON。"])
    prompt = "\n".join(sections)

    return [
        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(prompt, image_base64, image_mime_type),
        },
    ]


# 历史诊断只保留最近若干轮，且最多携带最近 2 张图片，防止请求体与上下文失控
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_IMAGES = 2


def _parse_history_image(block: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    从历史消息的 image_url 内容块中解析图片。

    仅接受 data URL 或纯 Base64 形式；外部 http 链接直接忽略，
    避免服务端被诱导去请求不受控的地址。
    """
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
    else:
        url = image_url
    if not isinstance(url, str) or not url.strip():
        return None

    try:
        base64_data, mime_type = parse_image_payload(url, DEFAULT_IMAGE_MIME_TYPE)
    except ValueError:
        return None
    if not base64_data:
        return None
    return base64_data, mime_type


def build_history_diagnosis_messages(history: List[dict]) -> List[Dict[str, Any]]:
    """
    将最近几轮对话（可含图片内容块）转换为多模态诊断 messages。

    - 只取最后 MAX_HISTORY_MESSAGES 条 user/assistant 消息；
    - 学生消息中的图片会保留最近 MAX_HISTORY_IMAGES 张送入模型；
    - 被省略的图片以 [图片已省略] 占位，保证文本上下文不断裂；
    - 最后追加一条 user 指令，强制模型输出四字段 JSON。

    历史为空或没有可诊断文本时抛出 ValueError。
    """
    if not isinstance(history, list) or not history:
        raise ValueError("history 不能为空")

    recent = [message for message in history if isinstance(message, dict)][-MAX_HISTORY_MESSAGES:]

    # 第一遍：解析文本与图片，统计图片总数，便于只保留最近几张
    prepared: List[Dict[str, Any]] = []
    total_images = 0

    for message in recent:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        content = message.get("content")
        text_parts: List[str] = []
        image_payloads: List[Tuple[str, str]] = []

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url" and role == "user":
                    parsed_image = _parse_history_image(block)
                    if parsed_image:
                        image_payloads.append(parsed_image)
                        total_images += 1

        prepared.append({
            "role": role,
            "text": "\n".join(part for part in text_parts if part).strip(),
            "images": image_payloads,
        })

    if not prepared:
        raise ValueError("history 中没有可诊断的对话消息")

    # 只保留最近 MAX_HISTORY_IMAGES 张图片
    first_keep_index = max(0, total_images - MAX_HISTORY_IMAGES)
    image_cursor = 0

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT}
    ]

    for item in prepared:
        kept_images: List[Tuple[str, str]] = []
        omitted_count = 0

        for image in item["images"]:
            if image_cursor >= first_keep_index:
                kept_images.append(image)
            else:
                omitted_count += 1
            image_cursor += 1

        text = item["text"]
        if omitted_count:
            placeholder = "[图片已省略] " * omitted_count
            text = f"{placeholder}{text}".strip()

        if not text and not kept_images:
            continue

        if kept_images:
            content_blocks: List[Dict[str, Any]] = [
                {"type": "text", "text": text or "（学生上传了图片）"}
            ]
            for base64_data, mime_type in kept_images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_data}"
                    },
                })
            messages.append({"role": item["role"], "content": content_blocks})
        else:
            messages.append({"role": item["role"], "content": text})

    if len(messages) == 1:
        raise ValueError("history 中没有可诊断的文本或图片")

    # 追加一条明确指令，强化 JSON 输出约束
    messages.append({
        "role": "user",
        "content": "请基于以上最近几轮《材料力学》对话（包含学生上传的图片）输出诊断 JSON，只输出 JSON 对象。",
    })
    return messages


def latest_user_text(history: List[dict]) -> str:
    """提取最近一条学生文字消息，用于诊断日志归档。"""
    for message in reversed(history or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = "\n".join(part for part in parts if part).strip()
            if text:
                return text
    return ""


async def extract_diagnosis_from_messages(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """对任意已构造好的诊断 messages 发起调用，并做 JSON 正则兜底解析。"""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    return normalize_diagnosis(parse_json_object(content))


async def extract_history_diagnosis(history: List[dict]) -> Dict[str, str]:
    """
    将最近几轮对话（含图片）发给诊断模型，解析并返回四字段 JSON。

    任一环节失败都会抛出异常，由 /api/get_diagnosis 统一转换为
    400（输入问题）或 502（模型/解析问题），不会让进程崩溃。
    """
    messages = build_history_diagnosis_messages(history)
    return await extract_diagnosis_from_messages(messages)


def persist_diagnosis(
    student_id: str,
    diagnosis: Dict[str, str],
    source: str,
    student_text: str = "",
) -> bool:
    """
    将有效诊断写入学生档案的 weakness_log。

    无任何有效信息的诊断不会落库，避免污染学情轨迹。
    同时保留 topic / logic_gap 字段，兼容教师端旧逻辑。
    """
    if not has_diagnosis_signal(diagnosis):
        return False

    student_data = db.read_student(student_id)
    if not student_data:
        return False

    log_entry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
    }
    for field in DIAGNOSIS_FIELDS:
        log_entry[field] = diagnosis.get(field, "")

    # 向后兼容：旧版教师端分析与导出可能读取 topic / logic_gap
    log_entry["topic"] = diagnosis.get("diagnosis") or "未识别明显薄弱点"
    log_entry["logic_gap"] = diagnosis.get("evidence") or diagnosis.get("diagnosis", "")

    if student_text.strip():
        log_entry["student_input"] = student_text.strip()[:300]

    logs = student_data.setdefault("weakness_log", [])
    logs.append(log_entry)
    return db.write_student(student_id, student_data, operator="DiagnosisAgent")


async def extract_diagnosis(
    student_text: str,
    tutor_reply: str,
    image_base64: Optional[str] = None,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
    history_text: str = "",
) -> Dict[str, str]:
    """调用诊断 Agent，返回规范化的四字段 JSON（失败时抛异常）。"""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=build_diagnosis_messages(
            student_text, tutor_reply, image_base64, image_mime_type, history_text
        ),
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    return normalize_diagnosis(parse_json_object(content))


async def run_diagnosis_agent(
    student_id: str,
    student_text: str,
    tutor_reply: str,
    image_base64: Optional[str] = None,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
    history_text: str = "",
    persist: bool = True,
    source: str = "chat_auto",
) -> Dict[str, str]:
    """
    后台诊断任务：失败只记录日志，不影响主对话。

    仅当诊断包含有效信息时才写入学生档案。
    """
    try:
        diagnosis = await extract_diagnosis(
            student_text, tutor_reply, image_base64, image_mime_type, history_text
        )
    except Exception as e:
        print(f"Diagnosis Agent Error: {e}")
        return empty_diagnosis()

    if persist:
        try:
            persist_diagnosis(student_id, diagnosis, source=source, student_text=student_text)
        except Exception as e:
            print(f"Diagnosis persist error: {e}")

    return diagnosis

TEACHER_CONFIG = {
    "username": os.getenv("TEACHER_USERNAME", "T001"),
    "password": os.getenv("TEACHER_PASSWORD", "123456"),
    "name": os.getenv("TEACHER_NAME", "张老师"),
}

# 教师端接口 Token：优先从环境变量读取；未配置时每次启动随机生成，
# 避免在源码中写死可被公开复用的固定 Token。
TEACHER_TOKEN = os.getenv("TEACHER_TOKEN", "").strip() or secrets.token_urlsafe(32)

# 未配置真实 Key 时仍允许服务启动（便于本地预览 UI），但模型调用会返回鉴权错误
if not API_KEY:
    print("警告: 未配置 DEEPSEEK_API_KEY，请在 .env 或系统环境变量中设置后再调用模型。")

client = AsyncOpenAI(api_key=API_KEY or "not-configured", base_url=BASE_URL)
app = FastAPI(title="K12 智能教育系统 - 模块化重构版")
db = DatabaseManager()


def ensure_demo_student():
    """确保演示学生档案存在，避免无登录模式下的诊断落库失败。"""
    try:
        if not db.read_student(DEFAULT_STUDENT_ID):
            db.create_student(DEFAULT_STUDENT_ID, "演示同学", ["比赛演示"])
    except Exception as e:
        print(f"初始化演示学生失败: {e}")


ensure_demo_student()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用脚本所在目录定位 HTML，避免从其他工作目录启动 uvicorn 时 404
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def serve_landing():
    """默认首页：LeadAnswer 启动页。"""
    return FileResponse(os.path.join(BASE_DIR, "landing.html"))


@app.get("/index.html")
async def serve_chat_app():
    """原系统聊天页。"""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# ==========================================
# 数据模型
# ==========================================
class CreateStudentRequest(BaseModel):
    student_id: str
    name: str
    tags: List[str]

class ChatRequest(BaseModel):
    student_id: str = DEFAULT_STUDENT_ID
    message: str = ""
    # 手写受力图 / 剪力图 / 公式截图的 Base64（可带 data URL 前缀）
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = DEFAULT_IMAGE_MIME_TYPE
    # 每轮回复结束后自动触发学情诊断（默认开启，可通过请求体关闭）
    auto_diagnosis: bool = True

class DiagnosisRequest(BaseModel):
    student_id: str = DEFAULT_STUDENT_ID
    message: str = ""
    reply: str = ""
    history: Optional[List[dict]] = None
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = DEFAULT_IMAGE_MIME_TYPE
    persist: bool = True

class DiagnosisHistoryRequest(BaseModel):
    student_id: str = DEFAULT_STUDENT_ID
    # 最近几轮对话，元素形如 {"role": "user", "content": "..."} 或
    # {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
    history: List[dict]
    persist: bool = True

class ChatEndRequest(BaseModel):
    student_id: str = DEFAULT_STUDENT_ID
    history: List[dict]

class QuizRequest(BaseModel):
    student_id: str = DEFAULT_STUDENT_ID
    diagnosis: Optional[Dict[str, str]] = None
    count: int = 4

# 权限校验中间件
def verify_teacher_token(x_token: str = Header(None)):
    if x_token != TEACHER_TOKEN:
        raise HTTPException(status_code=403, detail="无权执行此操作")
    return True

# ==========================================
# 接口实现
# ==========================================
# 演示模式：已移除登录接口。访问 /index.html 即可直接进入对话，
# 所有对话与诊断统一使用 DEFAULT_STUDENT_ID。

@app.get("/teacher/students", dependencies=[Depends(verify_teacher_token)])
async def list_students():
    return db.list_students()

@app.post("/teacher/students", dependencies=[Depends(verify_teacher_token)])
async def add_student(req: CreateStudentRequest):
    success, msg = db.create_student(req.student_id, req.name, req.tags)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.delete("/teacher/students/{student_id}", dependencies=[Depends(verify_teacher_token)])
async def delete_student(student_id: str):
    success, msg = db.delete_student(student_id, operator="张老师")
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/teacher/analysis", dependencies=[Depends(verify_teacher_token)])
async def analyze_student(student_id: str = Body(..., embed=True)):
    student_data = db.read_student(student_id)
    if not student_data:
        raise HTTPException(status_code=404, detail="找不到该学生")
    
    prompt = f"""
请基于以下《材料力学》学生档案生成学情诊断报告。

学生档案（weakness_log 中每条记录包含 diagnosis 薄弱点、evidence 文本证据、visual_evidence 图像证据、suggestion 下一步行动指令）：
{json.dumps(student_data, ensure_ascii=False)}

要求：
1. **薄弱点归类**：把 diagnosis 按受力分析、内力图、应力应变、强度/刚度/稳定性等模块归类，指出出现频次最高的薄弱点。
2. **证据核验**：区分文本证据（evidence）与视觉证据（visual_evidence），说明哪些结论证据充分、哪些仍需课堂观察确认。
3. **认知障碍分析**：结合 cognitive_profile 与 historical_evaluations，解释薄弱点背后的思维习惯成因。
4. **训练优先级**：按“先修依赖 + 出现频次”给出下一步的干预优先级。

请使用 Markdown 输出，禁止空泛评价。
"""
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是《材料力学》课程的教学诊断专家，擅长基于证据做学情归因，输出必须专业、具体。"},
                {"role": "user", "content": prompt},
            ]
        )
        return {"analysis": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 生成诊断报告失败: {str(e)}")

@app.post("/teacher/recommendation", dependencies=[Depends(verify_teacher_token)])
async def get_recommendation(student_id: str = Body(..., embed=True)):
    student_data = db.read_student(student_id)
    if not student_data:
        raise HTTPException(status_code=404, detail="找不到该学生")
    
    # 获取最近 5 条记录
    logs = student_data.get("weakness_log", [])
    recent_logs = logs[-5:] if logs else []
    
    context = {
        "basic_info": student_data.get("basic_info"),
        "recent_weaknesses": recent_logs
    }

    prompt = f"""
    作为一名《材料力学》课程负责人和资深教育心理学专家，请基于以下学生档案（特别是最近的 weakness_log）生成一份高水准的教学建议。

    学生背景及近期记录：
    {json.dumps(context, ensure_ascii=False)}

    每条 weakness_log 包含四个诊断字段：
    - diagnosis：具体薄弱点（如：未掌握剪力图突变规则、遗漏均布载荷）
    - evidence：文本证据（如：学生回复的推导过程错误）
    - visual_evidence：视觉证据（如：上传的受力图中缺少固定端的反力偶）
    - suggestion：下一步行动指令（如：帮我出两道带有集中力偶的简支梁剪力图练习题）

    要求输出以下三个部分，保持学术严谨，禁止使用空洞的词汇（如“努力学习”、“加油”等）：
    1. **当前认知水平评估**：逐条引用 diagnosis 与 evidence，精准刻画学生目前的思维深度、知识盲区及认知障碍点；若 visual_evidence 非空，说明图像暴露的作图习惯问题。
    2. **预测学习难点**：基于薄弱点的先修依赖，预测在接下来的进阶学习中，该生最可能在哪些具体概念或逻辑环节遇到瓶颈。
    3. **推荐教学方案**：把 suggestion 汇总为可执行的训练计划，明确材料力学题型（如带集中力偶的简支梁剪力图、均布载荷下的弯矩图）与检查标准，并说明如何验证学生是否真正掌握。

    请使用 Markdown 格式输出。
    """

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一位严谨的《材料力学》教育专家，输出内容需具备专业深度，并严格基于诊断证据。"},
                {"role": "user", "content": prompt}
            ]
        )
        return {"recommendation": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 生成建议失败: {str(e)}")

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    苏格拉底式对话接口。

    兼容原有纯文本调用；当请求携带 image_base64 时，
    自动切换为 OpenAI 多模态消息格式，支持手写受力图、
    剪力图/弯矩图与公式截图的辅导。
    """
    try:
        image_base64, image_mime_type = parse_image_payload(
            req.image_base64, req.image_mime_type
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user_text = (req.message or "").strip()
    if not user_text and not image_base64:
        raise HTTPException(status_code=400, detail="message 与 image_base64 至少需要提供一个")

    return StreamingResponse(
        stream_socratic_reply(
            req.student_id,
            user_text,
            image_base64,
            image_mime_type,
            background_tasks,
            req.auto_diagnosis,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

async def stream_socratic_reply(
    student_id: str,
    user_input: str,
    image_base64: Optional[str] = None,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
    background_tasks: Optional[BackgroundTasks] = None,
    auto_diagnosis: bool = True,
) -> AsyncGenerator[str, None]:
    """
    以流式方式返回导师回复。

    无图片时 user.content 为字符串；有图片时为 text + image_url 内容块，
    与 OpenAI 多模态接口格式一致。流式回复结束后，若 auto_diagnosis=True，
    会自动把「学生输入 + 图片 + 完整回复」交给诊断 Agent 提取四字段 JSON。
    """
    messages = build_chat_messages(user_input, image_base64, image_mime_type)
    full_reply = ""
    completed = False

    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                full_reply += content
                yield content
        completed = True
    except Exception as e:
        yield f"【连接中断】: {str(e)}"
    finally:
        # 仅在正常完成后诊断，避免把中断/报错内容当作学习证据；
        # 无有效回复或无背景任务时直接跳过。
        if completed and auto_diagnosis and background_tasks and full_reply.strip():
            background_tasks.add_task(
                run_diagnosis_agent,
                student_id,
                user_input,
                full_reply,
                image_base64,
                image_mime_type,
                "",
                True,
                "chat_auto",
            )

@app.post("/chat/diagnose")
async def diagnose_endpoint(req: DiagnosisRequest):
    """
    学情诊断接口（单轮或手动触发）。

    输入本轮学生消息、助教回复、可选图片与历史，输出固定四字段 JSON：
    diagnosis / evidence / visual_evidence / suggestion。
    默认将有效诊断写入学生档案，可通过 persist=false 只取结果不落库。
    """
    try:
        image_base64, image_mime_type = parse_image_payload(
            req.image_base64, req.image_mime_type
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    student_text = (req.message or "").strip()
    tutor_reply = (req.reply or "").strip()
    if not student_text and not tutor_reply and not image_base64:
        raise HTTPException(
            status_code=400,
            detail="message、reply、image_base64 至少需要提供一个",
        )

    history_text = history_to_text(req.history or [])
    try:
        diagnosis = await extract_diagnosis(
            student_text,
            tutor_reply,
            image_base64,
            image_mime_type,
            history_text,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"学情诊断失败: {str(e)}")

    persisted = False
    if req.persist:
        persisted = persist_diagnosis(
            req.student_id, diagnosis, source="manual", student_text=student_text
        )

    return {
        "diagnosis": diagnosis,
        "has_weakness": has_diagnosis_signal(diagnosis),
        "persisted": persisted,
    }

@app.post("/api/get_diagnosis")
async def get_diagnosis(req: DiagnosisHistoryRequest):
    """
    学情诊断接口（前端每轮对话后静默调用）。

    请求体中的 history 可包含多模态内容块，后端会保留最近几张图片
    一并送入多模态模型，解析出四字段 JSON 后返回。

    容错策略：
    - history 结构问题 -> 400；
    - 模型调用或 JSON 解析失败 -> 502，服务进程不受影响；
    - parse_json_object 内部已用正则 + raw_decode 处理多余字符。
    """
    try:
        messages = build_history_diagnosis_messages(req.history)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"history 格式不合法: {str(e)}")

    try:
        diagnosis = await extract_diagnosis_from_messages(messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"学情诊断失败: {str(e)}")

    persisted = False
    if req.persist:
        persisted = persist_diagnosis(
            req.student_id,
            diagnosis,
            source="history_api",
            student_text=latest_user_text(req.history),
        )

    return {
        "diagnosis": diagnosis,
        "has_weakness": has_diagnosis_signal(diagnosis),
        "persisted": persisted,
    }

@app.post("/api/generate_quiz")
async def generate_quiz(req: QuizRequest):
    """
    根据学情诊断生成 3–5 道针对性样卷，供“学情报告”页面使用。

    复用现有的模型调用与 JSON 解析逻辑；失败时返回 502，
    不影响前端重新对话或重新生成。
    """
    count = max(3, min(5, req.count or 4))
    diagnosis = req.diagnosis or {}

    prompt = f"""
请为以下《材料力学》学情诊断结果生成 {count} 道针对性练习题。

诊断结果：
{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

要求：
1. 题目数量必须为 {count} 道，难度递进，覆盖 diagnosis 指出的薄弱点。
2. 至少包含一道作图题或受力分析题，题目参数完整、可独立作答。
3. quiz_markdown 只包含题目、分值与作答要求；answer_markdown 包含简要解答与评分要点。
4. 公式使用 LaTeX。
5. 只输出 JSON 对象，键为 quiz_markdown 与 answer_markdown，不要输出其他文字。
"""

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        payload = parse_json_object(response.choices[0].message.content or "")
        quiz = payload.get("quiz_markdown") or payload.get("quiz") or ""
        answers = payload.get("answer_markdown") or payload.get("answers") or ""
        quiz = str(quiz).strip()
        answers = str(answers).strip()

        if not quiz:
            raise ValueError("模型未返回有效题目")

        return {
            "quiz": quiz,
            "answers": answers,
            "count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"样卷生成失败: {str(e)}")

@app.post("/chat/end")
async def end_chat(req: ChatEndRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_profiler_agent, req.student_id, req.history)
    return {"status": "processing"}

async def run_profiler_agent(student_id: str, history: List[dict]):
    """
    会话结束后的 Profiler Agent：基于完整历史生成一条汇总诊断。

    注意：历史中的图片可能已不可用（前端通常只回传文字），
    因此这里的 visual_evidence 往往为空；逐轮诊断才是视觉证据的主要来源。
    """
    if not history:
        return

    chat_content = history_to_text(history)
    if not chat_content.strip():
        return

    prompt = PROFILER_PROMPT_TEMPLATE.format(chat_content=chat_content)

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROFILER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        analysis = normalize_diagnosis(
            parse_json_object(response.choices[0].message.content or "")
        )
        persist_diagnosis(student_id, analysis, source="session_profiler")
    except Exception as e:
        print(f"Profiler Agent Error: {e}")

@app.get("/health")
async def health():
    return {
        "status": "online",
        "model": MODEL_NAME,
        "api_key_configured": bool(API_KEY),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
