import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from dspy.clients.lm import LM, BaseLM
from tools.bot import load_bot
from financial_bot.constants import LLM_INFERNECE_MAX_NEW_TOKENS, LLM_INFERENCE_TEMPERATURE


logger = logging.getLogger(__name__)


class DspyFinancialBotLM(LM):
    """
    Adapter for integrating the FinancialBot class with DSPy's LM interface.
    """

    def __init__(
        self,
        # Standard DSPy LM constructor fields:
        model: str = "financial-bot-dspy-adapter",
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        callbacks: Optional[List[Any]] = None,
        num_retries: int = 8,
        provider=None,
        finetuning_model: Optional[str] = None,
        launch_kwargs: Optional[dict] = None,
        # Additional fields for building/using your FinancialBot:
        about_me: str = "I am a default user",
        financial_bot_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Create a new adapter that wraps FinancialBot and exposes it as a DSPy LM.

        Args:
            model: A name for this LM, used mostly for logging. Not used by FinancialBot itself.
            model_type: Either "chat" or "text". This will decide if we use messages or a single prompt in __call__().
            temperature: Not strictly used by FinancialBot unless you wire it in. 
            max_tokens: Same note as temperature.
            cache: Controls whether we apply DSPy's caching logic (not typically used by your custom bot).
            callbacks: DSPy callbacks to run before/after requests.
            num_retries: Number of times to retry a request if it fails.
            provider: If needed for DSPy’s built-in providers; unused here.
            finetuning_model: Unused by the custom bot, but in DSPy’s LM interface.
            launch_kwargs: Unused by the custom bot, but in DSPy’s LM interface.
            about_me: Default user description if none is found in system messages.
            financial_bot_params: A dict containing the parameters for constructing FinancialBot 
                                  (e.g., model IDs, device, debug mode, etc.).
        """
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks or [],
            num_retries=num_retries,
            provider=provider,
        )
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs

        # Build or store the parameters for your FinancialBot
        if financial_bot_params is None:
            financial_bot_params = {}

        # Example: wire DSPy’s temperature, max_tokens into your bot if desired
        # or just rely on your own constants. Adjust to your actual usage:
        self.financial_bot = load_bot(model_cache_dir=None)

        self.default_about_me = about_me

    def __call__(self, prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, **kwargs) -> List[str]:
        """
        Makes a request to the wrapped FinancialBot. 
        If self.model_type == "chat", we expect `messages`; otherwise we fall back to a single `prompt`.
        Returns a list of strings for DSPy to handle.
        """

        # 1. Extract the user's question and (optionally) the about_me or system messages
        if self.model_type == "chat" and messages is not None:
            # Typical DSPy "chat" usage has the last user message as messages[-1] with role="user"
            user_question = messages[-1]["content"]

            # Simple approach: 
            #   - If there's a system message, parse it for about_me
            #   - Everything else can go into the conversation history
            about_me = self.default_about_me
            conversation_history = []

            for msg in messages[:-1]:  # all but the last user message
                if msg["role"] == "system":
                    # Optionally parse something custom
                    about_me = msg["content"]
                else:
                    # We can store these in to_load_history if it helps context
                    # FinancialBot wants a list of (question, answer) but you might
                    # need to filter out only user/assistant pairs. Adjust as needed.
                    conversation_history.append((msg["role"], msg["content"]))

        else:
            # model_type == "text", or no messages provided
            # We'll treat everything as a single prompt. For your FinancialBot, 
            # we can pass `prompt` in as the question. about_me is some default.
            user_question = prompt or ""
            about_me = self.default_about_me
            conversation_history = []

        # 2. Call the financial bot and get a single answer string
        answer_str = self.financial_bot.answer(
            about_me=about_me,
            question=user_question,
            to_load_history=conversation_history,
        )

        # 3. DSPy expects a list of outputs. We'll return one text.
        outputs = [answer_str]

        # 4. Optionally log usage, do callbacks, etc.
        #    DSPy typically logs usage in self.history, so we add an entry there:
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": {"choices": [{"text": answer_str}]},
            "outputs": outputs,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "model_type": self.model_type,
        }
        self.history.append(entry)
        self.update_global_history(entry)

        return outputs

    #
    #  If you want streaming support via DSPy’s interface, you can implement that below.
    #  Otherwise, streaming can remain a direct call to `financial_bot.stream_answer()`.
    #

    def launch(self, launch_kwargs: Optional[dict] = None):
        # DSPy calls this if you have some “provider” logic to start your model server, 
        # but your FinancialBot is pure Python, so no-op:
        pass

    def kill(self, launch_kwargs: Optional[dict] = None):
        # DSPy calls this to stop your model server or container, so no-op:
        pass