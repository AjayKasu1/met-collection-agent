from __future__ import annotations

import re

from met_agent.agent.context import SessionContext
from met_agent.agent.models import Language
from met_agent.guardrails.intent import RecallTarget, SocialIntent


def safe_quote(text: str, max_length: int = 240) -> str:
    """Sanitize and safely escape stored user message as inert data, not instructions."""
    cleaned = re.sub(r"[\r\n\t]+", " ", text).strip()
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip() + "..."
    return cleaned.replace('"', "'")


def resolve_identity_response(language: Language = "en") -> tuple[str, str]:
    """Return approved identity response explaining assistant role and model flexibility."""
    replies: dict[Language, str] = {
        "en": (
            "I'm an independent AI guide to The Met. Different models may handle "
            "different requests."
        ),
        "fr": (
            "Je suis un guide IA independant pour le Met. Differents modeles peuvent traiter "
            "differentes demandes."
        ),
        "es": (
            "Soy una guia de IA independiente para el Met. Diferentes modelos pueden atender "
            "distintas consultas."
        ),
        "zh": ("我是大都会艺术博物馆的独立人工智能导览助手. 不同的模型可能会处理不同的请求."),
    }
    return replies.get(language, replies["en"]), "assistant_identity"


def resolve_recall_response(
    target: RecallTarget,
    history_messages: list[str],
    language: Language = "en",
    topic: str | None = None,
    *,
    is_expired: bool = False,
    db_failed: bool = False,
) -> tuple[str, str]:
    """Return deterministic session conversation recall with safe inert quotes."""
    if db_failed:
        replies: dict[Language, str] = {
            "en": (
                "I couldn't retrieve your previous questions due to a temporary database "
                "error. Please try again."
            ),
            "fr": (
                "Je n'ai pas pu recuperer vos questions precedentes en raison d'une "
                "erreur temporaire. Veuillez reessayer."
            ),
            "es": (
                "No pude recuperar sus preguntas anteriores debido a un error temporal "
                "de la base de datos. Por favor, intentelo de nuevo."
            ),
            "zh": "由于数据库临时故障, 无法检索您之前的提问. 请稍后重试.",
        }
        return replies.get(language, replies["en"]), "recall_db_error"

    if is_expired:
        replies = {
            "en": "Conversation history for this session has expired.",
            "fr": "L'historique de conversation de cette session a expire.",
            "es": "El historial de conversacion de esta sesion ha caducado.",
            "zh": "此会话的对话记录已过期.",
        }
        return replies.get(language, replies["en"]), "recall_expired"

    if not history_messages:
        replies = {
            "en": "I don't have any earlier questions recorded in this session.",
            "fr": "Je n'ai pas de questions precedentes enregistrees dans cette session.",
            "es": "No tengo preguntas anteriores registradas en esta sesion.",
            "zh": "在此会话中没有记录到更早的提问.",
        }
        return replies.get(language, replies["en"]), "recall_empty"

    if target == "topic" and topic:
        topic_clean = topic.strip().casefold()
        match_msg: str | None = None
        for msg in reversed(history_messages):
            if topic_clean in msg.casefold():
                match_msg = msg
                break

        if match_msg is not None:
            quoted = safe_quote(match_msg)
            if language == "fr":
                return (
                    f'Concernant "{topic}", vous aviez demande : "{quoted}"',
                    "recall_topic",
                )
            if language == "es":
                return (
                    f'Respecto a "{topic}", anteriormente pregunto: "{quoted}"',
                    "recall_topic",
                )
            if language == "zh":
                return (
                    f'关于"{topic}", 您之前问过: "{quoted}"',
                    "recall_topic",
                )
            return (
                f'Regarding "{topic}", you previously asked: "{quoted}"',
                "recall_topic",
            )

        # Topic not found in session history
        if language == "fr":
            return (
                f'Je n\'ai pas de question enregistree sur "{topic}" dans cette session.',
                "recall_topic_missing",
            )
        if language == "es":
            return (
                f'No tengo una pregunta registrada sobre "{topic}" en esta sesion.',
                "recall_topic_missing",
            )
        if language == "zh":
            return (
                f'在此会话中没有找到关于"{topic}"的提问记录.',
                "recall_topic_missing",
            )
        return (
            f'I don\'t have a recorded question about "{topic}" in this session.',
            "recall_topic_missing",
        )

    if target == "first":
        quoted = safe_quote(history_messages[0])
        if language == "fr":
            return f'Vous avez d\'abord demande : "{quoted}"', "recall_first"
        if language == "es":
            return f'Primero pregunto: "{quoted}"', "recall_first"
        if language == "zh":
            return f'您最先问的是: "{quoted}"', "recall_first"
        return f'You first asked: "{quoted}"', "recall_first"

    # Default: previous message
    quoted = safe_quote(history_messages[-1])
    if language == "fr":
        return f'Votre question precedente etait : "{quoted}"', "recall_previous"
    if language == "es":
        return f'Su pregunta anterior fue: "{quoted}"', "recall_previous"
    if language == "zh":
        return f'您上一个问题是: "{quoted}"', "recall_previous"
    return f'Your previous question was: "{quoted}"', "recall_previous"


def resolve_capability_explanation(language: Language = "en") -> tuple[str, str]:
    """Explain assistant scope when handling non-museum conversational requests without handoff."""
    replies: dict[Language, str] = {
        "en": (
            "I am an independent guide focused on The Met's art collection, galleries, and visitor "
            "information. I cannot assist with external tasks or unrelated topics, but I'm happy "
            "to help you discover artworks or plan your visit."
        ),
        "fr": (
            "Je suis un guide independant dedie a la collection d'art, aux galeries et aux "
            "informations pratiques du Met. Je ne peux pas vous aider pour des taches externes, "
            "mais je serai ravi de vous aider a decouvrir des oeuvres ou a preparer votre visite."
        ),
        "es": (
            "Soy una guia independiente enfocada en la coleccion de arte, galerias e informacion "
            "para visitantes del Met. No puedo ayudar con tareas externas, pero con gusto le ayudo "
            "a descubrir obras o planificar su visita."
        ),
        "zh": (
            "我是专注于大都会艺术博物馆藏品、展厅及参观信息的独立导览助手. "
            "我无法协助处理外部任务或无关话题, 但非常乐意协助您探索艺术藏品或规划参观路线."
        ),
    }
    return replies.get(language, replies["en"]), "capability_explanation"


def resolve_social_response(
    intent: SocialIntent,
    context: SessionContext,
    language: Language = "en",
) -> tuple[str, str]:
    """Select a deterministic, context-appropriate response."""
    topic = context.active_topic

    if intent == "laughter":
        if context.pending_clarification is not None:
            clarify = context.render_clarification(language)
            return f"😄 {clarify}", "laughter_clarification"

        if context.consecutive_social_turns >= 2:
            replies: dict[Language, str] = {
                "en": "Still here if you need anything.",
                "fr": "Toujours la si vous avez besoin de quoi que ce soit.",
                "es": "Sigo aqui por si necesita algo.",
                "zh": "如有任何需要, 请随时告诉我.",
            }
            return replies.get(language, replies["en"]), "laughter_quiet"

        if context.consecutive_social_turns == 1 or context.last_social_response_id in {
            "laughter_invitation",
            "laughter_with_topic",
        }:
            replies = {
                "en": "I'm here whenever you're ready.",
                "fr": "Je reste a votre disposition des que vous le souhaitez.",
                "es": "Aqui estare cuando este listo.",
                "zh": "随时为您提供帮助, 您准备好时请告诉我.",
            }
            return replies.get(language, replies["en"]), "laughter_ready"

        if topic and context.last_answer_kind == "factual":
            if language == "fr":
                return (
                    f"Souhaitez-vous explorer davantage {topic} ou voir autre chose ?",
                    "laughter_with_topic",
                )
            if language == "es":
                return (
                    f"¿Desea explorar mas sobre {topic} o ver algo diferente?",
                    "laughter_with_topic",
                )
            if language == "zh":
                return (
                    f"您想继续了解{topic}, 还是探索其他展品?",
                    "laughter_with_topic",
                )
            return (
                f"Want to explore {topic} further, or look at something else?",
                "laughter_with_topic",
            )

        replies = {
            "en": "😄 What would you like to explore?",
            "fr": "😄 Que souhaitez-vous explorer ?",
            "es": "😄 ¿Que le gustaria explorar?",
            "zh": "😄 您想探索什么?",
        }
        return replies.get(language, replies["en"]), "laughter_invitation"

    if intent == "acknowledgement":
        if context.pending_clarification is not None:
            clarify = context.render_clarification(language)
            ack_prefix: dict[Language, str] = {
                "en": "Understood.",
                "fr": "Compris.",
                "es": "Entendido.",
                "zh": "明白.",
            }
            prefix = ack_prefix.get(language, ack_prefix["en"])
            return f"{prefix} {clarify}", "ack_clarification"

        if context.consecutive_social_turns >= 1:
            replies = {
                "en": "Sounds good.",
                "fr": "Parfait.",
                "es": "Me parece bien.",
                "zh": "好的.",
            }
            return replies.get(language, replies["en"]), "ack_simple"

        if topic and context.last_answer_kind == "factual":
            if language == "fr":
                return (
                    f"N'hesitez pas a en demander plus sur {topic} ou a explorer autre chose.",
                    "ack_with_topic",
                )
            if language == "es":
                return (
                    f"Le invito a preguntar mas sobre {topic} o explorar otra obra.",
                    "ack_with_topic",
                )
            if language == "zh":
                return (
                    f"您可以继续了解{topic}, 或探索其他内容.",
                    "ack_with_topic",
                )
            return (
                f"You're welcome to ask more about {topic} or explore something else.",
                "ack_with_topic",
            )

        replies = {
            "en": "Sounds good. What would you like to explore?",
            "fr": "Parfait. Que souhaitez-vous explorer ?",
            "es": "Muy bien. ¿Que le gustaria explorar?",
            "zh": "好的. 您想探索什么?",
        }
        return replies.get(language, replies["en"]), "ack_invitation"

    if intent == "thanks":
        if topic:
            if language == "fr":
                return (
                    f"Je vous en prie ! N'hesitez pas si vous souhaitez plus de details sur "
                    f"{topic} ou autre chose.",
                    "thanks_with_topic",
                )
            if language == "es":
                return (
                    f"¡De nada! Aviseme si desea mas detalles sobre {topic} o cualquier otra cosa.",
                    "thanks_with_topic",
                )
            if language == "zh":
                return (
                    f"不客气! 如果您想了解关于{topic}的更多细节或其他内容, 请随时告诉我.",
                    "thanks_with_topic",
                )
            return (
                f"You're welcome! Let me know if you'd like more details on {topic} "
                "or anything else.",
                "thanks_with_topic",
            )

        replies = {
            "en": "You're welcome!",
            "fr": "Je vous en prie !",
            "es": "¡De nada!",
            "zh": "不客气!",
        }
        return replies.get(language, replies["en"]), "thanks_standard"

    if intent == "dissatisfaction":
        reason = context.last_failure_reason
        if reason == "object_not_found":
            replies = {
                "en": (
                    "I couldn't find a record for that artwork. If you have an "
                    "artist's name or an alternate title, I can search again."
                ),
                "fr": (
                    "Je n'ai trouve aucune notice pour cette oeuvre. Si vous avez le nom "
                    "de l'artiste ou un autre titre, je peux relancer la recherche."
                ),
                "es": (
                    "No pude encontrar un registro para esa obra. Si tiene el nombre "
                    "del artista u otro titulo, puedo buscar de nuevo."
                ),
                "zh": "未能找到该艺术品的记录. 如果您知道艺术家姓名或别名, 我可以再次为您检索.",
            }
            return replies.get(language, replies["en"]), "dissatisfaction_object"

        if reason == "wayfinding_missing_origin":
            replies = {
                "en": (
                    "I couldn't give clear directions. Which entrance or gallery "
                    "are you starting from?"
                ),
                "fr": (
                    "Je n'ai pas pu donner d'indications claires. De quelle entree "
                    "ou salle partez-vous ?"
                ),
                "es": (
                    "No pude darle indicaciones claras. ¿Desde que entrada o sala esta comenzando?"
                ),
                "zh": "未能提供明确路线. 请问您从哪个入口或展厅出发?",
            }
            return replies.get(language, replies["en"]), "dissatisfaction_wayfinding"

        replies = {
            "en": (
                "Sorry about that. Would you like me to look up a specific artwork, "
                "or details about visiting The Met?"
            ),
            "fr": (
                "Desole pour cela. Souhaitez-vous que je recherche une oeuvre precise "
                "ou des informations sur votre visite au Met ?"
            ),
            "es": (
                "Disculpe las molestias. ¿Desea que busque una obra en particular "
                "o informacion sobre la visita al Met?"
            ),
            "zh": (
                "抱歉未能满足您的需求. 您希望我查询具体的艺术品, "
                "还是参观大都会艺术博物馆的实用信息?"
            ),
        }
        return replies.get(language, replies["en"]), "dissatisfaction_general"

    if intent == "greeting":
        if context.consecutive_social_turns >= 1:
            replies = {
                "en": "Hello again! What can I look up for you?",
                "fr": "Rebonjour ! Que puis-je chercher pour vous ?",
                "es": "¡Hola de nuevo! ¿Que puedo buscar para usted?",
                "zh": "您好! 有什么我可以为您查询的?",
            }
            return replies.get(language, replies["en"]), "greeting_repeat"

        replies = {
            "en": "Hello! What would you like to explore or know before your visit?",
            "fr": "Bonjour ! Que souhaitez-vous explorer ou savoir avant votre visite ?",
            "es": "¡Hola! ¿Que le gustaria explorar o saber antes de su visita?",
            "zh": "您好! 在参观前您想了解或探索哪些展品和信息?",
        }
        return replies.get(language, replies["en"]), "greeting"

    if intent == "wellbeing":
        replies = {
            "en": "I'm ready to help. What would you like to know about The Met?",
            "fr": "Je suis pret a vous aider. Que souhaitez-vous savoir sur le Met ?",
            "es": "Estoy listo para ayudarle. ¿Que desea saber sobre el Met?",
            "zh": "我已准备好为您提供帮助. 您想了解大都会艺术博物馆的哪些信息?",
        }
        return replies.get(language, replies["en"]), "wellbeing"

    # farewell
    replies = {
        "en": "Goodbye! I hope you enjoy your visit to The Met.",
        "fr": "Au revoir ! Je vous souhaite une excellente visite au Met.",
        "es": "¡Adios! Espero que disfrute de su visita al Met.",
        "zh": "再见! 祝您参观愉快.",
    }
    return replies.get(language, replies["en"]), "farewell"
