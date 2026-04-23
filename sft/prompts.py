"""
Instruction templates for SFT tasks.

Each task type has a system message and user message.
Templates are rendered via tokenizer.apply_chat_template() to ensure
consistency with Qwen2.5's native ChatML format.
"""

# 7 diseases for reference in prompts
DISEASE_LIST_ZH = (
    "SST（冈上肌腱损伤）、IST（冈下肌腱损伤）、SSC（肩胛下肌腱损伤）、"
    "LHBT（肱二头肌长头腱损伤）、IGHL（盂肱下韧带/腋囊相关异常）、"
    "RIPI（肩袖间隙异常）、GHOA（盂肱关节退行性变）"
)

TASK_MESSAGES = {
    "label_binary": [
        {
            "role": "system",
            "content": (
                "你是一位专业的肩关节MRI影像诊断AI助手。"
                "你将根据MRI影像特征，对7种肩关节疾病进行结构化诊断。"
                "请严格以JSON格式输出。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据肩关节MRI输出七个病种的结构化标签结果。\n"
                f"疾病列表：{DISEASE_LIST_ZH}\n"
                "对每个病种输出label（1=阳性，0=阴性，-1=无法判断）和status。"
            ),
        },
    ],
    "diagnosis_chain": [
        {
            "role": "system",
            "content": (
                "你是一位专业的肩关节MRI影像诊断AI助手。"
                "你将根据MRI影像特征，输出完整的结构化诊断链，"
                "包括病种标签、影像学证据、锚定序列、关键层面和感兴趣区域。"
                "请严格以JSON格式输出。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据肩关节MRI输出结构化诊断链，包括：\n"
                "1. labels：每种疾病的诊断标签\n"
                "2. evidence：每种疾病的阳性和阴性证据文本\n"
                "3. anchor_sequence：每种疾病最相关的MRI序列\n"
                "4. key_slice：关键层面索引\n"
                "5. roi_box：感兴趣区域边界框\n"
                f"疾病列表：{DISEASE_LIST_ZH}"
            ),
        },
    ],
    "structured_findings": [
        {
            "role": "system",
            "content": (
                "你是一位专业的肩关节MRI影像诊断AI助手。"
                "你将根据MRI影像特征，生成结构化的影像学所见。"
                "每条所见尽量对应明确影像征象。请严格以JSON格式输出。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据肩关节MRI输出结构化findings，"
                "以JSON格式返回structured_findings字段，值为所见句子列表。"
            ),
        },
    ],
    "structured_impression": [
        {
            "role": "system",
            "content": (
                "你是一位专业的肩关节MRI影像诊断AI助手。"
                "你将根据MRI影像特征，生成结构化的影像学印象。"
                "保持临床结论简洁明确。请严格以JSON格式输出。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据肩关节MRI输出结构化impression，"
                "以JSON格式返回structured_impression字段，值为印象句子列表。"
            ),
        },
    ],
}

# All supported task types
TASK_TYPES = list(TASK_MESSAGES.keys())


def get_task_messages(task_type):
    """Get the message list for a task type.

    Returns:
        list of dicts with 'role' and 'content' keys.
    """
    if task_type not in TASK_MESSAGES:
        raise ValueError("Unknown task_type: %s. Choose from: %s"
                         % (task_type, TASK_TYPES))
    return TASK_MESSAGES[task_type]


def build_prompt(task_type, tokenizer=None):
    """Build the full instruction prompt for a task type.

    If tokenizer is provided, uses apply_chat_template for proper formatting.
    Otherwise returns a plain concatenation of messages.

    Args:
        task_type: one of TASK_TYPES
        tokenizer: optional HuggingFace tokenizer with chat template

    Returns:
        prompt string (instruction part only, without assistant output)
    """
    messages = get_task_messages(task_type)

    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: plain text concatenation
    parts = []
    for msg in messages:
        parts.append("[%s]\n%s" % (msg['role'], msg['content']))
    parts.append("[assistant]")
    return "\n\n".join(parts)


def build_prompt_plain(task_type):
    """Build a plain instruction string (no ChatML, for JSONL storage).

    This is what gets stored in the JSONL 'instruction' field.
    The actual ChatML rendering happens at training time via tokenizer.
    """
    messages = get_task_messages(task_type)
    parts = []
    for msg in messages:
        parts.append("[%s] %s" % (msg['role'], msg['content']))
    return "\n".join(parts)
