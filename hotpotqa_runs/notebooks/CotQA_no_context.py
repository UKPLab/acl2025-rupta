import sys, os

sys.path.append('..')
root = '../root/'

from hotpotqa_runs.util import summarize_trial, log_trial, save_agents
import joblib
from hotpotqa_runs.agents import CoTAgent, ReflexionStrategy
from hotpotqa_runs.prompts import cot_simple_reflect_agent_prompt, cot_simple_reflect_prompt, cot_simple_agent_prompt
from hotpotqa_runs.fewshots import COTQA_SIMPLE6, COT_SIMPLE_REFLECTION

hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop=True)
print(ReflexionStrategy.__doc__)
strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION

agents = [CoTAgent(question=row['question'],
                   context='',
                   key=row['answer'],
                   agent_prompt=cot_simple_agent_prompt if strategy == ReflexionStrategy.NONE else cot_simple_reflect_agent_prompt,
                   cot_examples=COTQA_SIMPLE6,
                   reflect_prompt=cot_simple_reflect_prompt,
                   reflect_examples=COT_SIMPLE_REFLECTION,
                   ) for _, row in list(hotpot.iterrows())[:3]]

n = 5
trial = 0
log = ''
for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        agent.run(reflexion_strategy=strategy)
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_trial(agents, trial)
    correct, incorrect = summarize_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

with open(os.path.join(root, 'CoT', 'no_context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'),
          'w') as f:
    f.write(log)
save_agents(agents, os.path.join(root, 'CoT', 'no_context', strategy.value, 'agents'))
