# 微调

**基本流程:**
- 选择预训练模型
- 准备好用于模型微调的数据集
- 准备微调前的测试问题集，并且测试结果（方便微调后进行对比）
- 设定模型微调需要的超参数
- 执行模型微调
- 微调前的测试问题集再度进行测试，对比效果
- 效果不满意，调整数据集/超参数，再度进行执行模型微调
- 结束：得到满意的模型

## 通过平台微调大模型
- 硅基流动


## 常见问题
* 为什么我的微调效果不好？跟数据集有关系吗？
* 数据集的格式是固定的吗？我要弄成什么样子？
* 数据集还分很多种类？测试集、训练集、验证集的区别是啥？
* 我想要微调特定领域的模型？去哪获取这个领域公开的数据集？
* 手动整理数据集太累了，有没有什么快速标注数据集的方法？
* 数据集可以用 AI 生成吗？怎么把领域的文献转成可供模型微调的数据集？


## 寻找数据集
* 前置知识：了解常见的微调任务类型，根据特定任务选择适合的数据集
* 前置知识：了解常见的数据集格式，数据集的类型
* 学会怎么找：一些推荐的获取公开数据集的途径
* 学会这么标：基于标注工具半自动标注数据集
* 学会怎么做：将特定领域的文献转换为目标格式的数据集
* 学会怎么做：基于 AI 全自动生成模型蒸馏数据集


## 监督微调

- 指令微调：输入格式、输出格式等转化的微调
- 对话微调：包含角色身份、多轮对话上下文，让模型学会在不同场景下如何生成合适的回复
- 领域适配：模型在特定领域的数据上进行微调，使其更好地适应特定领域的任务和需求
- 文本分类：学习文本特征 => 类别的关系，比如 `"text": "这款手机续航长达48小时，拍照效果惊艳", "label": "positive"`
- 思考链推理能力的微调：学会分布思考+复杂逻辑推理


## 强化学习微调

在监督微调的基础上，通过人类来主动反馈优化模型生成质量来进行微调




# 不进行微调如何强化模型的方法

## 蒸馏
如果大模型已经完全可以满足你在特定任务上的诉求，但是部署成本又太高了，你完全可以选择一个小模型，然后从大模型里把你任务里需要用到的领域知识提取出来，构造成数据集，再去微调小模型，从而让这个小模型也能在你的特定领域完成任务


