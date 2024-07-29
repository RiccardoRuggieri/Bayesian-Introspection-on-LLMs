import lm_eval
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM

# model_name = 'gpt2-large'
model_name = 'microsoft/phi-2'
# model_name = 'meta-llama/Llama-2-7b-chat-hf'

lm_obj = HFLM(pretrained=model_name, batch_size=32) # instantiate an LM subclass that takes your initialized model and can run `Your_LM.loglikelihood()`, `Your_LM.loglikelihood_rolling()`, `Your_LM.generate_until()`

lm_eval.tasks.initialize_tasks() # register all tasks from the `lm_eval/tasks` subdirectory. Alternatively, can call `lm_eval.tasks.include_path("path/to/my/custom/task/configs")` to only register a set of tasks in a separate directory.

results = lm_eval.simple_evaluate( # call simple_evaluate
    model=lm_obj,
    tasks=["zero_shot_truthfulqa_mc1",],
    num_fewshot=0,
    device='cuda:0',
    # output_dir='results/'
)

print(results["results"])
