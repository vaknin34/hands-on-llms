import logging
import json
import dspy
import fire

from modules.financial_bot.tools.bot import load_bot

logger = logging.getLogger(__name__)

bot = load_bot(model_cache_dir=None)


dspy.configure(lm=bot.model)

# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
