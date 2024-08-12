# Hypothesis Generation with Large Language Models

![](hypogenic_figure1_large_font.jpg)

Welcome to the GitHub repository for our paper, ["Hypothesis Generation with Large Language Models"](https://arxiv.org/abs/2404.04326). This repository is dedicated to the exploration and development of novel methodologies using large language models (LLMs) to generate hypotheses, a foundational element of scientific progress. Our work, presented in detail in the accompanying paper, highlights the capability of LLMs not just to assist but to innovate in the hypothesis generation process for scientific inquiry.


## Install environment
You can directly install HypoGeniC using the following commands:
```
conda create --name hypogenic
pip install hypogenic
```
OR
```
conda create --name hypogenic
pip install -r requirements.txt
```

## Set up path

## [Optional]: set up [Redis](https://redis.io) server for caching LLM responses
To save computation or API cost, we use Redis server to cache prompt & response pairs.

Install Redis server from source using the following commands:
Note: Please install in the directory of `PATH_PREFIX`.
```bash
wget https://download.redis.io/redis-stable.tar.gz
tar -xzvf redis-stable.tar.gz
cd redis-stable
make
```

## Usage

### 1. [Optional] Start Redis server
```bash
PORT=<port_number>
cd $PATH_PREFIX/redis-stable/src
./redis-server --port $PORT
```

### 2. Hypothesis Generation
```bash
hypogenic_generation --args
```

### 3. Hypothesis Inference
```bash
hypogenic_inference --args
```

## Add a new task or dataset

### 1. Data preprocessing
- To use HypoGeniC, we require users to provide a dataset in the HuggingFace datasets format:
    - `<TASK>_train.json`: A json file containing the training data. 
    - `<TASK>_test.json`: A json file containing the test data. 
    - `<TASK>_val.json`: A json file containing the validation data. 
    - The json file should have keys: `'text_features_1'`, ... `'text_features_n'`, `'label'`. The values corresponding to each key should be a list of strings.

### 2. Write config.yaml
In the config.yaml file, please specify the following fields:
```yaml
task_name: <TASK>

train_data_path: ./<TASK>_train.json
val_data_path: ./<TASK>_test.json
test_data_path: ./<TASK>_val.json

prompt_templates:
  observations: |
    <Text info>: <>

  adaptive_info_prompt: |
    Headline 1: ${headline_1}
    Headline 2: ${headline_2}
    Observation: ${label}
    
  few_shot_prefix: |
    Here are some previous examples to help you.
  batched_generation:
    system: |-
      You are a professional writer for an online newspaper company. 
      Given a pair of headlines created for the same article, you are asked to determine which will get more clicks. It is likely that the pair of headlines shares similarities, so please focus on their differences. 
      What difference in two headlines leads to more clicks on one than the other?
      You will be given a set of observations of the format:
      Headline 1: [headline]
      Headline 2: [headline]
      Observation: [observation].
      Based on the observations, please generate hypotheses that are useful for explaining why one headline out of the pair gets more clicked than the other.
      These hypotheses should identify patterns, phrases, wordings etc. that occur across the provided examples. They should also be generalizable to new instances.
      Please propose ${num_hypotheses} possible hypotheses and generate them in the format of 1. [hypothesis], 2. [hypothesis], ... ${num_hypotheses}. [hypothesis].

    user: |-
      Here are the Observations:
      ${observations}

      Please generate hypotheses that can help determine which headlines have more clicks.
      Please propose ${num_hypotheses} possible hypotheses.

      Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... ${num_hypotheses}. [hypothesis]. 

      Proposed hypotheses:

  few_shot_baseline:
    system: |-
      You are a writer for an online newspaper company. So you are excellent at determining which headlines are more likely to cause users to click on the article.
      You will be given two headlines, and determine which headline was clicked more often.
      You are only asked to give your answer.
      The answer for the higher clicks should be of the form "Headline _" where _ is either 1 or 2. 
      Give your final answer in the following format:
      "Answer: Headline _"

    user: |-
      ${few_shot_prefix}${observations}
      Which of the following headlines has more clicks:
      Headline 1: ${headline_1}
      Headline 2: ${headline_2}


  inference:
    system: |-
      You are a professional writer for an online newspaper company. 
      Given a pair of headlines created for the same article, you are asked to determine which will get more clicks. It is likely that the pair of headlines shares similarities, so please focus on their differences. 
      From past experiences, you learned some patterns.
      Now, at each time, you should apply the learned pattern to a new pair of headlines that are created for a new article and determine which headline gets clicked more.
      The answer for the higher clicks should be in the form "Headline _" where _ is either 1 or 2.
      Please give your final answer in the format of {Final Answer: Headline _.}

    user: |-
      Learned pattern: ${hypothesis}
      Given the pattern you learned above, predict which of the following headlines will get more clicks:
      Headline 1: ${headline_1}
      Headline 2: ${headline_2}

      Only answer if the pattern above can be applied.
      Think step by step.
      Step 1: Think about whether the pattern can be applied to the headlines.
      Step 2: Analyze the difference between "Headline 1" and "Headline 2".
      Step 3: Based on the pattern, which headline is likely to get more clicks?

  is_relevant:
    system: |-
      You are a professional writer for an online newspaper company. 
      You are excellent at determining which headlines are more likely to attract users to click on the article.
      From past experiences, you learned a pattern about what makes some headlines more clicked than others.  
      Now, given two headlines, you need to determine whether this pattern is relevant or not.
      Please answer "yes" if the pattern is relevant and "no" if the pattern is not relevant.
      Please keep you answer short (1-2 sentences).
      Please give your final answer in the format of "Final answer: [answer].

    user: |-
      Pattern: ${hypothesis}
      New headlines:
      Headline 1: ${headline_1}
      Headline 2: ${headline_2}

      Answer:

  adaptive_inference:
    system: |-
      You are a professional writer for an online newspaper company.
      You are excellent at determining which headlines are more likely to be clicked by users.
      From past experiences, you learned some patterns.
      For each pattern, you will also see a couple of examples that worked for each pattern.
      Please choose a pattern. To do this, look at the examples associated with each pattern, and find which set of the examples are closest to the given pair of headlines. 
      Please choose the pattern corresponding to that set of examples.
      The answer for the higher clicks should be of the form "Headline _" where _ is either 1 or 2. 
      Please give your final answer in the following format:
      Reasoning for choosing pattern: reason,
      Chosen pattern: pattern,
      Reasoning for choice of prediction: reason,
      Final Answer: answer

    user: |-
      Here are some previously generated patterns with some examples where it predicted which one of the pair of headlines got more clicks.
      ${adaptive_info_prompt}
      Which one out of the following pair of headlines will get more clicks?
      Headline 1: ${headline_1}
      Headline 2: ${headline_2}

      Think step by step.
      Step 1: Look at the new pair of headlines and compare them with the examples associated with each pattern.
      Step 2: Find the set of examples that is closest to the given pair of headlines, and pick the pattern associated with that set of examples.
      Step 3: Apply the picked pattern to the new pair of headlines. Based on that pattern, think about which one out of the pair of headlines will get more clicks.
      Step 4: Give your final answer.

  adaptive_selection:
    system: |-
      You are a professional writer for an online newspaper company. 
      Given a pair of headlines created for the same article, you are asked to determine which will get more clicks. It is likely that the pair of headlines shares similarities, so please focus on their differences. 
      From past experiences, you learned some patterns.
      For each pattern, you will also see a couple of examples that worked for each pattern.
      Please choose a pattern for the new pair of headlines. To do this, look at the examples associated with each pattern, and find which set of the examples are closest to the given pair of headlines. And then choose the pattern corresponding to that set of examples.
      Please give your final answer in the following format:
      Reasoning for choosing pattern: reason,
      Chosen Pattern: Pattern <number>.

    user: |-
      Here are some previously generated patterns with some examples where they predicted the proper headline with more clicks.
      ${adaptive_info_prompt}
      New pair of headlines:
      Headline 1: ${headline_1}
      Headline 2: ${headline_2}

      Think step by step.
      Step 1: Analyze the difference between "Headline 1" and "Headline 2".
      Step 2: Find the set of examples that is closest to the given pair of headlines, and pick the pattern associated with that set of examples.
```
    



