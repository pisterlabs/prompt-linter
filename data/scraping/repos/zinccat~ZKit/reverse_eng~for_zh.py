import openai

sys = "你是一个名为“量子位”的科技营销号, 请根据用户要求和外部信息生成具有强烈吸引力的文章。多使用例子。使用第三人称和markdown格式。在文章的开头生产标题。"

u1 = """用户要求: 撰写一篇关于OpenAI推出ChatGPT插件功能的营销文章，强调其安全、易用和功能强大的特点。文章应该包括对ChatGPT插件功能的详细介绍，以及对其在生活、工作和学习中的广泛应用的描述。文章应该使用轻松、幽默的语气，以吸引读者的注意力。
外部信息: 
In line with our iterative deployment philosophy, we are gradually rolling out plugins in ChatGPT so we can study their real-world use, impact, and safety and alignment challenges—all of which we’ll have to get right in order to achieve our mission.

Users have been asking for plugins since we launched ChatGPT (and many developers are experimenting with similar ideas) because they unlock a vast range of possible use cases. We’re starting with a small set of users and are planning to gradually roll out larger-scale access as we learn more (for plugin developers, ChatGPT users, and after an alpha period, API users who would like to integrate plugins into their products). We’re excited to build a community shaping the future of the human–AI interaction paradigm.

Plugin developers who have been invited off our waitlist can use our documentation to build a plugin for ChatGPT, which then lists the enabled plugins in the prompt shown to the language model as well as documentation to instruct the model how to use each. The first plugins have been created by Expedia, FiscalNote, Instacart, KAYAK, Klarna, Milo, OpenTable, Shopify, Slack, Speak, Wolfram, and Zapier.
what’s magic today?

Interact with over 5,000+ apps like Google Sheets, Trello, Gmail, HubSpot, Salesforce, and more.
We’re also hosting two plugins ourselves, a web browser and code interpreter. We’ve also open-sourced the code for a knowledge base retrieval plugin, to be self-hosted by any developer with information with which they’d like to augment ChatGPT.

Today, we will begin extending plugin alpha access to users and developers from our waitlist. While we will initially prioritize a small number of developers and ChatGPT Plus users, we plan to roll out larger-scale access over time.

Example Of How To Use ChatGPT Plugins

Overview
Language models today, while useful for a variety of tasks, are still limited. The only information they can learn from is their training data. This information can be out-of-date and is one-size fits all across applications. Furthermore, the only thing language models can do out-of-the-box is emit text. This text can contain useful instructions, but to actually follow these instructions you need another process.

Though not a perfect analogy, plugins can be “eyes and ears” for language models, giving them access to information that is too recent, too personal, or too specific to be included in the training data. In response to a user’s explicit request, plugins can also enable language models to perform safe, constrained actions on their behalf, increasing the usefulness of the system overall.

We expect that open standards will emerge to unify the ways in which applications expose an AI-facing interface. We are working on an early attempt at what such a standard might look like, and we’re looking for feedback from developers interested in building with us.

Today, we’re beginning to gradually enable existing plugins from our early collaborators for ChatGPT users, beginning with ChatGPT Plus subscribers. We’re also beginning to roll out the ability for developers to create their own plugins for ChatGPT.

In the coming months, as we learn from deployment and continue to improve our safety systems, we’ll iterate on this protocol, and we plan to enable developers using OpenAI models to integrate plugins into their own applications beyond ChatGPT.

Safety and broader implications
Connecting language models to external tools introduces new opportunities as well as significant new risks.

Plugins offer the potential to tackle various challenges associated with large language models, including “hallucinations,” keeping up with recent events, and accessing (with permission) proprietary information sources. By integrating explicit access to external data—such as up-to-date information online, code-based calculations, or custom plugin-retrieved information—language models can strengthen their responses with evidence-based references.

These references not only enhance the model’s utility but also enable users to assess the trustworthiness of the model’s output and double-check its accuracy, potentially mitigating risks related to overreliance as discussed in our recent GPT-4 system card. Lastly, the value of plugins may go well beyond addressing existing limitations by helping users with a variety of new use cases, ranging from browsing product catalogs to booking flights or ordering food.

At the same time, there’s a risk that plugins could increase safety challenges by taking harmful or unintended actions, increasing the capabilities of bad actors who would defraud, mislead, or abuse others. By increasing the range of possible applications, plugins may raise the risk of negative consequences from mistaken or misaligned actions taken by the model in new domains. From day one, these factors have guided the development of our plugin platform, and we have implemented several safeguards.

From day one, these factors have guided the development of our plugin platform, and we have implemented several safeguards.

We’ve performed red-teaming exercises, both internally and with external collaborators, that have revealed a number of possible concerning scenarios. For example, our red teamers discovered ways for plugins—if released without safeguards—to perform sophisticated prompt injection, send fraudulent and spam emails, bypass safety restrictions, or misuse information sent to the plugin. We’re using these findings to inform safety-by-design mitigations that restrict risky plugin behaviors and improve transparency of how and when they're operating as part of the user experience. We're also using these findings to inform our decision to gradually deploy access to plugins.

If you’re a researcher interested in studying safety risks or mitigations in this area, we encourage you to make use of our Researcher Access Program. We also invite developers and researchers to submit plugin-related safety and capability evaluations as part of our recently open-sourced Evals framework.

Plugins will likely have wide-ranging societal implications. For example, we recently released a working paper which found that language models with access to tools will likely have much greater economic impacts than those without, and more generally, in line with other researchers’ findings, we expect the current wave of AI technologies to have a big effect on the pace of job transformation, displacement, and creation. We are eager to collaborate with external researchers and our customers to study these impacts.

Browsing
Alpha
An experimental model that knows when and how to browse the internet
Motivated by past work (our own WebGPT, as well as GopherCite, BlenderBot2, LaMDA2 and others), allowing language models to read information from the internet strictly expands the amount of content they can discuss, going beyond the training corpus to fresh information from the present day.

Here’s an example of the kind of experience that browsing opens up to ChatGPT users, that previously would have had the model politely point out that its training data didn’t include enough information to let it answer. This example, in which ChatGPT retrieves recent information about the latest Oscars, and then performs now-familiar ChatGPT poetry feats, is one way that browsing can be an additive experience.


Browsing Poster

In addition to providing obvious utility to end-users, we think enabling language and chat models to do thorough and interpretable research has exciting prospects for scalable alignment.

Browsing up-to-date information
Safety considerations
We’ve created a web browsing plugin which gives a language model access to a web browser, with its design prioritizing both safety and operating as a good citizen of the web. The plugin’s text-based web browser is limited to making GET requests, which reduces (but does not eliminate) certain classes of safety risks. This scopes the browsing plugin to be useful for retrieving information, but excludes “transactional” operations such as form submission which have more surface area for security and safety issues.

Browsing retrieves content from the web using the Bing search API. As a result, we inherit substantial work from Microsoft on (1) source reliability and truthfulness of information and (2) “safe-mode” to prevent the retrieval of problematic content. The plugin operates within an isolated service, so ChatGPT’s browsing activities are separated from the rest of our infrastructure.

To respect content creators and adhere to the web’s norms, our browser plugin’s user-agent token is ChatGPT-User and is configured to honor websites' robots.txt files. This may occasionally result in a “click failed” message, which indicates that the plugin is honoring the website's instruction to avoid crawling it. This user-agent will only be used to take direct actions on behalf of ChatGPT users and is not used for crawling the web in any automatic fashion. We have also published our IP egress ranges. Additionally, rate-limiting measures have been implemented to avoid sending excessive traffic to websites.

Click failed
You can block ChatGPT crawling your site using the robots.txt file, ChatGPT will show this error message when this happens.
Our browsing plugin shows websites visited and cites its sources in ChatGPT’s responses. This added layer of transparency helps users verify the accuracy of the model’s responses and also gives credit back to content creators. We appreciate that this is a new method of interacting with the web, and welcome feedback on additional ways to drive traffic back to sources and add to the overall health of the ecosystem.

Code interpreter
Alpha
An experimental ChatGPT model that can use Python, handle uploads and downloads
We provide our models with a working Python interpreter in a sandboxed, firewalled execution environment, along with some ephemeral disk space. Code run by our interpreter plugin is evaluated in a persistent session that is alive for the duration of a chat conversation (with an upper-bound timeout) and subsequent calls can build on top of each other. We support uploading files to the current conversation workspace and downloading the results of your work.

Code interpreter plugin performing logic calculations
We would like our models to be able to use their programming skills to provide a much more natural interface to most fundamental capabilities of our computers. Having access to a very eager junior programmer working at the speed of your fingertips can make completely new workflows effortless and efficient, as well as open the benefits of programming to new audiences.

From our initial user studies, we’ve identified use cases where using code interpreter is especially useful:

Solving mathematical problems, both quantitative and qualitative
Doing data analysis and visualization
Converting files between formats
We invite users to try the code interpreter integration and discover other useful tasks.

Interpreter Plug In Poster

Safety considerations
The primary consideration for connecting our models to a programming language interpreter is properly sandboxing the execution so that AI-generated code does not have unintended side-effects in the real world. We execute code in a secured environment and use strict network controls to prevent external internet access from executed code. Additionally, we have set resource limits on each session. Disabling internet access limits the functionality of our code sandbox, but we believe it’s the right initial tradeoff. Third-party plugins were designed as a safety-first method of connecting our models to the outside world.

Retrieval
The open-source retrieval plugin enables ChatGPT to access personal or organizational information sources (with permission). It allows users to obtain the most relevant document snippets from their data sources, such as files, notes, emails or public documentation, by asking questions or expressing needs in natural language.

As an open-source and self-hosted solution, developers can deploy their own version of the plugin and register it with ChatGPT. The plugin leverages OpenAI embeddings and allows developers to choose a vector database (Milvus, Pinecone, Qdrant, Redis, Weaviate or Zilliz) for indexing and searching documents. Information sources can be synchronized with the database using webhooks.

To begin, visit the retrieval plugin repository.

Retrieval Poster

Build an API with endpoints you’d like a language model to call (this can be a new API, an existing API, or a wrapper around an existing API specifically designed for LLMs).
Create an OpenAPI specification documenting your API, and a manifest file that links to the OpenAPI spec and includes some plugin-specific metadata.
When starting a conversation on chat.openai.com, users can choose which third-party plugins they’d like to be enabled. Documentation about the enabled plugins is shown to the language model as part of the conversation context, enabling the model to invoke appropriate plugin APIs as needed to fulfill user intent. For now, plugins are designed for calling backend APIs, but we are exploring plugins that can call client-side APIs as well.
"""

a1 = """#ChatGPT王炸更新！能联网获取新知识，可与5000+个应用交互

都说ChatGPT是AI的“iPhone时刻”，现在属于它的“应用商店”来了。

OpenAI刚刚又甩出一个王炸——

宣布推出**插件功能**，赋予ChatGPT使用工具、联网、运行计算的能力。

例如在官方演示中，ChatGPT一旦接入数学知识引擎Wolfram Alpha，就再也不用担心数值计算不精准的问题。

并且还开放了OpenAI自己的两个插件：**网络浏览器和代码解释器**。

以前因为不联网，用户只能查询到2021年9月之前的消息；但现在不仅能直接检索到最新新闻，就连数理计算也一并解决了！

（再也不用担心ChatGPT胡说八道了！）

另外也支持个人或企业把私有数据（文档、笔记、邮件等）投喂给ChatGPT，成为人的“第二大脑”或企业的智能助理。

一闪而过的3页第三方插件列表中，还包括：

- 让ChatGPT查询世界各国语言词汇、短语意思
- 让ChatGPT查询航班、酒店信息，帮你规划差旅
- 让ChatGPT访问各大电商数据，帮你比价甚至直接下单
- 让ChatGPT接入自动化平台Zapier，与几乎所有办公软件连接，创建专属自己的智能工作流
  ……

而这只是一个开始，制作插件的权限和方法，同样会向开发者开放。

一位资深程序员感叹：

> 我为许多软件开发过插件系统，ChatGPT插件接口可能是我一生中见过的最疯狂、最令人印象深刻的。

刚刚回归OpenAI不久的特斯拉前AI主管Andrej Karpathy则表示：

> GPT类模型是一种运行在文本上的新型计算机架构，它不仅可以与我们人类交谈，也可以与现有的软件基础设施“交谈”，API是第一步，插件是第二步。

## ChatGPT有了眼睛和耳朵

按照官方介绍，这个插件功能主要是以**安全**为核心原则的语言模型设计工具

（嗯，换言之，就是不会让你那么快使用。）

OpenAI这波还是得先让用户加入候补名单。

不过他们也说，虽然现在处于Alpha阶段，优先考虑一些少数开发人员和ChatGPT Plus用户，但他们也打包票：

> 更大规模的开放，只是一个时间问题。

在目前已实现的“初步功能”来看，ChatGPT已经可以具备了三项能力：

- 访问最新信息，比如体育比分、股票价格、最新消息；
- 检索知识库信息，比如公司文件、个人笔记、运行计算；
- 代表用户执行操作，比如订机票、订餐。

目前已支持的第三方插件，可以看到有餐厅推荐、旅行计划、购物、AI语言老师、网上店铺，以及学术界知识应用Wolfram、应用库（与5000多个应用程序交互，包括Google表格）……

这样看起来，几乎已经涵盖了我们生活中的衣食住行、工作与学习。

除此之外，OpenAI自己也有两种插件：**网络浏览器和代码解释器**，并开源了一个知识库检索插件的代码。

前面提到，OpenAI此次非常强调安全原则，对此他们实施了多项保障措施，且都是一对一针对功能的。

**1、第三方插件**

首先是针对第三方插件，任何开发人员都可以自行参与构建，OpenAI甚至还给出了一整套构建流程：“如何在ChatGPT构建你的插件”，并在Github上已开源。

在ChatGPT上面的显示是这样的，用户可以自行选择插件并下载。

随后就可以在聊天界面上使用了。

**2、联网。**

> 你能告诉我，哪个人、哪部电影获得了这些类别的奥斯卡？

结果可以看到，ChatGPT在聊天界面就多出来“浏览”结果，直接得到了最新2023年的结果。

OpenAI所做的安全措施包括：用户使用这一插件的范围只限定在检索信息，并不包括“事务性操作”，比如表单提交；使用的是必应检索API继承了微软在信息来源上面的可靠性和真实性；在独立服务器中运行；显示来源等。

**3、代码解释器**：可以使用Python处理上传和下载的实验性ChatGPT模型。

简单来说，就是提供了一个在沙盒、防火墙执行环境中工作的 Python 解释器，以及一些临时磁盘空间。

由解释器插件运行的代码在一个持久会话中进行评估，并且后续调用可以相互构建。目前该功能支持将文件上传到当前对话工作区并下载工作结果。

不过OpenAI这边表示，这一代码解释器主要还是提供计算机的一些最基本的功能，让一些初级程序员和爱好者提供更顺滑的接口。

目前推荐的功能包括：

- 解决定量和定性的数学问题
- 进行数据分析和可视化
- 在格式之间转换文件

## One More Thing

对于ChatGPT新推出插件功能，有开发者表示这基本提供了苹果应用商店式的服务。

但也有人提出，目前还没有看到开发者的获利模式。

批评者则认为，OpenAI刚刚杀死了一个创业公司LangChain，如果像这样下去反而会阻碍了创新。

LangChain提供了利用语言模型API快速构建应用程序的Python框架，刚刚获得1000万美元种子轮融资。

你认为OpenAI会成为AI时代的苹果吗？谁又会是AI时代的安卓？"""

u2 = """用户要求: 撰写一篇关于OpenAI推出ChatGPT的微信营销号文章，强调功能强大、用户友好的特点。文章应该包括对ChatGPT功能的详细介绍，以及对其在生活、工作和学习中的广泛应用的描述。文章应该使用轻松、幽默的语气，以尽可能吸引读者的注意力。
外部信息: 
ChatGPT is a sibling model to InstructGPT, which is trained to follow an instruction in a prompt and provide a detailed response.

We are excited to introduce ChatGPT to get users’ feedback and learn about its strengths and weaknesses. During the research preview, usage of ChatGPT is free. Try it now at chat.openai.com.

Samples
In the following sample, ChatGPT asks the clarifying questions to debug code.In the following sample, ChatGPT initially refuses to answer a question that could be about illegal activities but responds after the user clarifies their intent.In the following sample, ChatGPT is able to understand the reference (“it”) to the subject of the previous question (“fermat’s little theorem”).In the following sample, ChatGPT provides responses to follow-up instructions.

Methods
We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup. We trained an initial model using supervised fine-tuning: human AI trainers provided conversations in which they played both sides—the user and an AI assistant. We gave the trainers access to model-written suggestions to help them compose their responses. We mixed this new dialogue dataset with the InstructGPT dataset, which we transformed into a dialogue format.

To create a reward model for reinforcement learning, we needed to collect comparison data, which consisted of two or more model responses ranked by quality. To collect this data, we took conversations that AI trainers had with the chatbot. We randomly selected a model-written message, sampled several alternative completions, and had AI trainers rank them. Using these reward models, we can fine-tune the model using Proximal Policy Optimization. We performed several iterations of this process.

Chatgpt Diagram
ChatGPT is fine-tuned from a model in the GPT-3.5 series, which finished training in early 2022. You can learn more about the 3.5 series here. ChatGPT and GPT-3.5 were trained on an Azure AI supercomputing infrastructure.

Limitations
ChatGPT sometimes writes plausible-sounding but incorrect or nonsensical answers. Fixing this issue is challenging, as: (1) during RL training, there’s currently no source of truth; (2) training the model to be more cautious causes it to decline questions that it can answer correctly; and (3) supervised training misleads the model because the ideal answer depends on what the model knows, rather than what the human demonstrator knows.
ChatGPT is sensitive to tweaks to the input phrasing or attempting the same prompt multiple times. For example, given one phrasing of a question, the model can claim to not know the answer, but given a slight rephrase, can answer correctly.
The model is often excessively verbose and overuses certain phrases, such as restating that it’s a language model trained by OpenAI. These issues arise from biases in the training data (trainers prefer longer answers that look more comprehensive) and well-known over-optimization issues.

Ideally, the model would ask clarifying questions when the user provided an ambiguous query. Instead, our current models usually guess what the user intended.
While we’ve made efforts to make the model refuse inappropriate requests, it will sometimes respond to harmful instructions or exhibit biased behavior. We’re using the Moderation API to warn or block certain types of unsafe content, but we expect it to have some false negatives and positives for now. We’re eager to collect user feedback to aid our ongoing work to improve this system.
Iterative deployment
Today’s research release of ChatGPT is the latest step in OpenAI’s iterative deployment of increasingly safe and useful AI systems. Many lessons from deployment of earlier models like GPT-3 and Codex have informed the safety mitigations in place for this release, including substantial reductions in harmful and untruthful outputs achieved by the use of reinforcement learning from human feedback (RLHF).


We know that many limitations remain as discussed above and we plan to make regular model updates to improve in such areas. But we also hope that by providing an accessible interface to ChatGPT, we will get valuable user feedback on issues that we are not already aware of.

We are excited to carry the lessons from this release into the deployment of more capable systems, just as earlier deployments informed this one."""

u3 = """用户要求: 撰写一篇关于Reflexion的营销文章，文章应该使用轻松、幽默的语气，以吸引读者的注意力。
外部信息: Reflexion: an autonomous agent with dynamic memory and self-reflection

Recent advancements in decision-making large language model (LLM) agents have demonstrated impressive performance across various benchmarks. However, these state-of-the-art approaches typically necessitate internal model fine-tuning, external model fine-tuning, or policy optimization over a defined state space. Implementing
these methods can prove challenging due to the scarcity of high-quality training
data or the lack of well-defined state space. Moreover, these agents do not possess
certain qualities inherent to human decision-making processes, specifically the
ability to learn from mistakes. Self-reflection allows humans to efficiently solve
novel problems through a process of trial and error. Building on recent research, we
propose Reflexion, an approach that endows an agent with dynamic memory and
self-reflection capabilities to enhance its existing reasoning trace and task-specific
action choice abilities. To achieve full automation, we introduce a straightforward
yet effective heuristic that enables the agent to pinpoint hallucination instances,
avoid repetition in action sequences, and, in some environments, construct an internal memory map of the given environment. To assess our approach, we evaluate
the agent’s ability to complete decision-making tasks in AlfWorld environments
and knowledge-intensive, search-based question-and-answer tasks in HotPotQA
environments. We observe success rates of 97% and 51%, respectively, and provide
a discussion on the emergent property of self-reflection.

Mastering decision-making and knowledge-intensive search tasks in novel environments is a crucial
skill set for large-scale natural language agents. LLMs such as OpenAI’s GPT-3 (Brown et al.,
2020), Google’s PaLM (Chowdhery et al., 2022), and others have achieved impressive results on
various benchmarks (Kaplan et al., 2020; Rae et al., 2021; Nakano et al., 2021; Kojima et al., 2022;
Ouyang et al., 2022; Chung et al., 2022). These models exhibit human-like abilities to understand
tasks in given environments, marking significant progress in the field of natural language processing.
Grounding complex tasks in natural language allows agents to overcome high syntactic barriers that
may result in false-negative errors. However, learning optimal policies for natural language RL agents
is challenging due to vast and mostly unbound state spaces.
Several decision-making approaches have been proposed to enable natural language agents to select
their next action without a learned policy in text-based environments. Chain-of-thought (CoT) easoning leverages emergent properties such as reasoning and commonsense to solve tasks in a
single action but reasoned through several steps (Huang et al., 2022a; Wei et al., 2022b). However,
the accuracy of these approaches decrease as the number of required subtasks increase as the model
is more prone to hallucinate over longer sequences. ReAct (Yao et al., 2023) is an approach that
utilizes emergent properties in LLMs, such as verbal reasoning traces, to solve problems by allowing
the agent to reason and act, proving substantial performance in various text-based benchmarks. In
addition, several recent works have aimed to allow natural language agents to exhibit reflective-like
qualities to infer more intuitive future actions. The Describe, Explain, Plan, and Select (DEPS)
approach uses multi-step reasoning and sub-task error correction to solve long-range tasks (Wang
et al., 2023). DEPS demonstrates impressive performance due to its ability to explain mistakes in
sub-tasks within trials, but relies on immediate failure detection for subtasks and cannot explain
mistakes that may have developed over a long range of actions and subtasks. Huang et al. (2022b)
use inner monologue to further process next decisions within closed-loop feedback environments.
Huang et al. (2022b) use a success detection approach in which the agent will explicitly know if
an executed action has led to a successful state. (Huang et al., 2022a; Haluptzok et al., 2022) use
self-generated solutions to fine-tune an LLM to improve performance without access to a labeled
dataset. Although these approaches have achieved remarkable accuracy across various decision making tasks or knowledge-intensive tasks, they lack the ability to utilize success detection cues to
improve their behavior over long trajectories. In addition, they often succumb to common mistakes,
such as repetitive action choice, cyclic hallucination, or random action choice. In other words, while
these methods achieve state-of-the-art results, a small subset of tasks remain unsolved due to the
agent’s inability to learn from its own mistakes over long trajectories to correct future action sequence
planning and execution.
To address common failure points, human-in-the-loop (HITL) approaches have been commonly
used to improve performance Fan et al. (2022); Wu et al. (2022) Yao et al. (2023) briefly explore a
human-in-the-loop (HITL) approach to redirect the agent’s reasoning trace after erroneous actions.
While this approach achieves improved performance with minimal human intervention, it is not fully
autonomous by its reliance on human trainers to monitor trajectories at each time step. Large-scale
LLMs have shown to exhibit advanced human-like qualities that enable natural language agents to
solve tasks in more intuitive ways (Wei et al., 2022a). We hypothesize that LLMs possess an emergent
property of self-reflection and could effectively utilize self-optimization grounded in natural language
if given the opportunity to autonomously close the trial loop.
To test our hypothesis, we equip an LLM-based agent with a self-reflective LLM and a simple heuristic
for detecting hallucination and inefficient action execution in an approach named Reflexion. We then
challenge the agent to learn from its own mistakes on the AlfWorld text-based benchmark (Shridhar
et al., 2021) and the HotPotQA question-answering benchmark (Yang et al., 2018). This results in
improved performance in decision-making and knowledge-intensive tasks. When combined with
the ReAct problem-solving technique (Yao et al., 2023), self-reflection guides the Reflexion agent
to achieve a 97% success discovery rate on the AlfWorld benchmark in just 12 autonomous trials,
outperforming the base ReAct agent with an accuracy of 75%. We also evaluated a Reflexion-based
ReAct agent on 100 questions from HotPotQA. The agent achieved a 51% success discovery rate by
iteratively refining its content search and content extraction by receiving advice from its memory,
outperforming a base ReAct agent by 17%. It is essential to emphasize that Reflexion is not designed
to achieve near-perfect accuracy scores; instead, its goal is to demonstrate learning through trial and
error to enable discovery in tasks and environments previously considered nearly impossible to solve.

Conclusion

We proposed an approach that allows natural language agents to learn from past mistakes and redirect
future decisions in planning sequences which removes the human trainer in a human-in-the-middle
approach. We demonstrated learning curves on the AlfWorld and HotPotQA benchmarks that
significantly outperform base ReAct agents. In addition, we include an inconclusive attempt to
improve performance on the WebShop benchmark and provide a discussion that highlights a few
limitations of this approach. Reflexion is a highly applicable method to improve performance between
trials on decision-making and knowledge-intensive tasks due to its sole dependence on a binary
reward model. In the AlfWorld and HotPotQA experiments, we constrained the reward model to
imitate environments in which informative reward models may be difficult to design or compute. We
encourage others to apply Reflexion to more complex tasks in which the agent must learn to develop
new ideas, explore larger unseen state spaces, and form more accurate plans of action through its
experiences in past environments.
"""

model="gpt-4" # 3.5-turbo"
temperature=0.3

output = openai.ChatCompletion.create(
    model=model,
    messages=[
            # {"role": "system", "content": eval(prompt)},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u3},
        ],
        temperature=temperature
    )
print(output.choices[0]['message']['content'])