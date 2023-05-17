import openai
import numpy as np

def gpt3(prompt, engine="text-davinci-003",logprobs=1,temperature=0,**kwargs):
   out=openai.Completion.create(prompt=prompt, engine=engine, logprobs=logprobs, temperature=temperature, **kwargs)
   return out


# def gpt3(prompt, engine="text-davinci-003", logprobs=1, echo_probs=False, echo_response=False, temperature=0, **kwargs):
#   out=openai.Completion.create(prompt=prompt, engine=engine, logprobs=logprobs, temperature=temperature, **kwargs)
#   if echo_probs and logprobs>0:
#      x=out.choices[0]["logprobs"]["top_logprobs"][0]
#      print("Completion probabilities: ")
#      for i in x.keys(): print(i, round(np.exp(x[i]),2))
#   if echo_response:
#      print("Completion: ")
#      print(out.choices[0]["text"])
#   return([out.choices[0]["text"].strip(),round(np.exp(out.choices[0]["logprobs"]["token_logprobs"][0]),2)])

def gpt3_probs(out):
   return([out.choices[0]["text"].strip(),round(np.exp(out.choices[0]["logprobs"]["token_logprobs"][0]),2)])
   x = out.choices[0]["logprobs"]["top_logprobs"][0]
   # x = out.choices[1-n]["logprobs"]["top_logprobs"][]
   return dict([(i,round(np.exp(x[i]))) for i in x])

# def gpt3(prompt, engine="text-davinci-003", logprobs=1, echo_probs=False, echo_response=False, temperature=0, **kwargs):
#   out=openai.Completion.create(prompt=prompt, engine=engine, logprobs=logprobs, temperature=temperature, **kwargs)
#   if echo_probs and logprobs>0:
#      x=out.choices[0]["logprobs"]["top_logprobs"][0]
#      print("Completion probabilities: ")
#      for i in x.keys(): print(i, round(np.exp(x[i]),2))
#   if echo_response:
#      print("Completion: ")
#      print(out.choices[0]["text"])
#   res = [(out.choices[i]["text"].strip(),round(np.exp(out.choices[i]["logprobs"]["token_logprobs"][i]),2)) for i in range ]
#   return([out.choices[0]["text"].strip(),round(np.exp(out.choices[0]["logprobs"]["token_logprobs"][0]),2)])