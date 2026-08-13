+++
date = '2025-07-09T23:06:03+08:00'
title = 'Quantization'
draft = 'false'
description = ""   
categories = ["LLM"]
image = ""
+++

## 疑惑

### Pytorch平台上对RWKV的困惑度计算模块进行量化的经历  

Q:更稳妥的方式是在模型加载和量化后，将整个模型转为float32，再量化后转换成float32的目的是什么呢，这样不是把原来量化好的给毁了吗  
A：  

#### pytorch上量化做了什么  

当调用torch.quantization.quantize_dynamic时，PyTorch并不仅仅是把权重张量的数据类型（dtype）改了一下。它做了一件更根本的事：它将模型中的标准nn.Linear层替换成了一个全新的、专门的模块，叫做DynamicQuantizedLinear。  
在这个新的DynamicQuantizedLinear模块内部，权重确实被压缩并以**torch.qint8（8位整型）**的格式存储了起来。  
同时，这个模块还保存了将这些INT8整数反量化回浮点数所需的缩放因子（scale）和零点（zero-point）（它们是浮点数）。  

#### .to(torch.float32) 到底做了什么？  

当您对一个模型调用.to(dtype)方法时，PyTorch会遍历模型的所有参数（parameters）和缓冲区（buffers），并尝试将它们的类型转换为您指定的dtype。  
这里的关键在于：.to() 方法只会转换浮点类型的张量（例如 bfloat16, float16, float64）。  
它会跳过那些已经是**整型（Integer）**的张量，比如我们量化后存储为torch.qint8的权重。 
因此，INT8权重本身保持不变，量化的成果得以保留。  

#### 那我们为什么要执行这个转换呢？  

这个操作的目的，正是为了解决您在上一个问题中遇到的 `RuntimeError: expected scalar type Float but found BFloat16` 错误。  
让我们回顾一下当时的情况：  
(1)**原始模型**：您加载模型时，因为设置了`'dataformat': 'bf16'`，所以模型中所有的浮点参数都是`bfloat16`类型。这包括`Embedding`层、`LayerNorm`层，以及`Linear`层的权重和偏置。
(2)**量化之后**：`quantize_dynamic`函数只替换了`Linear`层。但是，模型中的**其他层**（如`Embedding`和`LayerNorm`）的参数**依然是`bfloat16`类型**。
(3)**冲突发生**：当数据流经模型时，`Embedding`层输出一个`bfloat16`的张量，这个张量被送入到我们新替换的`DynamicQuantizedLinear`层。然而，这个量化层的CPU后端实现**期望接收一个标准的`float32`张量**作为输入，而不是`bfloat16`。于是，类型不匹配，程序崩溃。

**所以，调用`model.to(torch.float32)`的真正目的，是将模型中**除了INT8权重之外的所有其他浮点参数和缓冲区**（比如仍然是`bfloat16`格式的`Embedding`层或`LayerNorm`层）全部统一转换为标准的`float32`格式。

**总结一下，这个操作可以理解为一次“内务整理”：**

* **INT8权重**： untouchable，保持量化状态。
* **其他所有浮点部分（`bfloat16`）**：全部转换为`float32`。

这样一来，整个模型的数据流就统一了：非量化层输出`float32`张量，量化层接收`float32`张量，内部进行计算，从而完美解决了类型冲突的问题，同时保留了INT8量化带来的内存节省优势。

#### python中的enumereate

(1) 没有 `enumerate` 的情况
您的原始循环是这样的：
`for begin_loc in tqdm(range(0, 1536, 512)):`

这个循环会执行三次，在每一次循环中：
* **第1次循环**：`begin_loc` 的值是 **0**
* **第2次循环**：`begin_loc` 的值是 **512**
* **第3次循环**：`begin_loc` 的值是 **1024**

可以看到，这个循环只提供了一个变量 `begin_loc`。

(2) 加入 `enumerate` 之后的情况

您修改后的正确循环是这样的：
`for idx, begin_loc in enumerate(tqdm(range(0, 1536, 512))):`

这个循环同样会执行三次，在每一次循环中：
* **第1次循环**：`idx` 的值是 **0**，`begin_loc` 的值是 **0**
* **第2次循环**：`idx` 的值是 **1**，`begin_loc` 的值是 **512**
* **第3次循环**：`idx` 的值是 **2**，`begin_loc` 的值是 **1024**

结论
通过对比可以清晰地看到：  
**核心变量 `begin_loc` 的值完全没变**：在每次对应的循环中，`begin_loc` 得到的值和原来一模一样，都是 `0, 512, 1024, ...`。所以您循环内部使用 `begin_loc` 的所有逻辑都不会受到任何影响。  
**循环次数完全没变**：`range` 函数生成多少个数字，循环就会执行多少次。`enumerate` 不会增加或减少循环的次数。  
**唯一的改变**：`enumerate` 只是为您**额外提供**了一个从0开始的计数器变量 `idx`。这个变量正是您在 `print` 语句中所需要的，它解决了 `NameError`，并且让您能方便地知道当前是第几步循环。  
所以，加入 `enumerate` 是一个纯粹的“增强”操作，它在不干扰原有功能的基础上，安全地添加了您需要的新功能（即步数索引）。这是在Python中进行带索引循环的标准且推荐的做法。

#### 如果计算RWKV中的困惑度的呢  

核心思想是**“对比每一个字（Token）的输出与下一个真实字的关系”**  
RWKV的困惑度计算的核心代码（先计算loss）：  

```python
with torch.no_grad():
    logits, state = model.forward_parallel_slices(input, state, slice_len=1024)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output)
```

#### 只进行推理计算，不计算梯度  二者有什么区别吗  

训练是为了教会模型，而推理是为了使用已经教好的模型  
当然有，这两种模式在计算流程、资源消耗和最终目的上有着本质的区别。

简单来说，**训练**是为了**教会**模型，而**推理**是为了**使用**已经教好的模型。  

| 特性 (Characteristic) | 训练模式 (Training Mode) | 推理模式 (Inference Mode) |
| :--- | :--- | :--- |
| **核心目标** | **学习和优化**模型参数（权重），让模型变得更“聪明”。 | **使用**已训练好的模型对新数据进行**预测**和**评估**。 |
| **计算流程** | **前向传播 + 反向传播** (Forward Pass + Backward Pass) | **仅有前向传播** (Forward Pass ONLY) |
| **梯度计算** | **必须计算梯度**。梯度是反向传播的核心，它告诉每个权重应该如何调整才能减小误差（loss）。 | **完全不需要计算梯度**。因为模型参数是固定的，我们不打算更新它，所以计算梯度就成了不必要的开销。 |
| **内存占用** | **高**。为了能够进行反向传播，PyTorch需要构建一个“计算图”，并**保存所有中间步骤的计算结果**（激活值）。这会占用大量内存。 | **低**。因为不需要反向传播，PyTorch可以**用完一个中间结果后立即丢弃**，无需为计算图保存任何额外信息，大大节省了内存。 |
| **计算速度** | **慢**。既要跑一遍前向传播，又要跑一遍更复杂的反向传播。 | **快**。只执行一次前向传播，省去了所有与梯度相关的计算。 |
| **在PyTorch中的体现** | 默认模式。执行`loss.backward()`和`optimizer.step()`来更新权重。 | 使用`with torch.no_grad():`代码块包裹。同时常配合`model.eval()`来关闭Dropout等只在训练时使用的层。 |

因此，在计算困惑度时，使用 `with torch.no_grad():` 是一个重要的优化步骤，它能让评估过程变得更快、更节省资源。  
