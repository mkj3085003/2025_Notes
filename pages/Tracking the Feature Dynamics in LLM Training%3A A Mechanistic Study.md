## SAE - Track方法
collapsed:: true
	- **实验目的**：提出一种高效的方法来追踪LLMs训练过程中特征的演变，通过在训练检查点之间获得连续的SAE序列。
	- **实验设置**：
		- 使用Pythia-410m-deduped模型，层=4。
		- 在特定的Transformer层之前的残差流上训练SAEs。
		- 采用递归初始化和训练方法，每个检查点的SAE用前一个检查点的SAE参数进行初始化，并在当前检查点的激活上进行训练。
	- **结果分析（对应图 1）**：成功构建了连续的SAE系列，有效跟踪了特征演变，后续SAE训练时间大幅缩短，约为初始训练步骤的1/20，体现了该方法的高效性与可行性，为后续分析奠定基础。
	  
	  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=NjY1NTIxMTRjNTY5ODljMDI4YzM5NWM1M2IwYmYwNGRfT3dReHFGRHJxTzV0WGVmOUlodXNKNTFIMURSZDY1MnVfVG9rZW46SEdFdGJIaUIyb3NtaXp4a2lIYmNvSTZpbkpoXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
- ## **特征演变阶段实验**
	- **实验目的**：概述特征在训练过程中如何演变，识别特征转变的不同阶段和模式。
	- **实验设置**：
		- 使用SAE-Track在Pythia模型的不同检查点上提取特征。
		- 根据激活行为将特征分为两类：token-level特征和concept-level特征。
	- **实验结果**：
		- 识别出特征演变的三个阶段：初始化与预热、涌现阶段、收敛阶段。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=MDQ0Y2U0Nzk5MDU1YmNhNjA5MzFlN2FlYThiZGFkMjRfOFdobzROOVdMdmpZcmlVREtoRkNCcWFQRlBaWHlpQmRfVG9rZW46RTRxV2JmbDhlb1NYS1h4WUlXUmNNeUF1bkZoXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
		- 在初始化和预热阶段，token - level特征出现但仅与特定token相关联，concept - level特征分散；
		- 在涌现阶段，concept - level特征从噪声状态逐渐向抽象概念对齐，token - level特征保持稳定；
		- 在收敛阶段，两种特征都达到可解释状态，表明模型在训练过程中特征语义理解逐渐成熟。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=Njc1ZDVkOWUwOGFiNDA0YzQ3ZDVhOGFjODA2MjlkMGVfb0IweW41RWhHZFlNZHN3bWtScTNNV2tjNGtzMjN0QUhfVG9rZW46TkxTWmJZeTlxb0NBaWt4aXBoZmN4d1JRbnJnXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
- ## **特征转换模式实验**
  collapsed:: true
	- **实验设置**：同样基于SAE - Track提取的特征，观察不同训练阶段特征的具体变化方式，识别其转换模式。
	- **实验结果**：
		- 观察到三种主要的特征转变模式：维持、转移、分组。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=YmU5MTUzZDNlODliM2Q4MjhhNTk4M2IzODE1YmI4M2RfMFBFQTd1WFNqTzlqOXhkZWw5WTBLdGtqYVNJT0ROempfVG9rZW46UjF6ZmIwQTJEb0g3dkR4TmhBbmNEcWtGbkdiXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
	- **结果分析（对应图 2）**：
		- 维持模式下部分token - level特征在各检查点对同一token保持稳定激活；
		- 转换模式中某些token - level特征会转变为新的token - level特征或演变为concept - level特征；
		- 分组模式使噪声特征合并为有意义的表示，展示了特征在训练过程中的动态重组和演变规律。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTJkZTUzNGMwNGI3NDQ3ODQ0Yjg3NzlmMTIwNWNjMTJfUDBMUmJQSkpHenlneDN5RG5JN2lLRUZuUHprRTIzOVNfVG9rZW46T29DcmJIb3pXb2R4NWN4TWJYWWM4OVdwblZkXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
- ## **特征形成分析实验**
  collapsed:: true
	- **实验目的**：分析特征从噪声激活到有意义表示的转变过程。
	- **实验设置**：
		- 使用SAE在最终检查点定义特征区域，并追踪这些区域内数据点的演变。
			- 通过公式 $$\mathcal{R}_{i}=\left\{x |\left(\hat{W}_{i} \cdot x+\hat{b}_{i}\right)>0\right\}$$定义特征区域
		- 利用UMAP可视化不同检查点下各种特征的激活集，并通过计算特征形成进度度量（如 $$M_{i}(t)=\overline{Sim}_{\mathcal{A}_{i}^{t}}-\overline{Sim}_{\mathcal{A}_{raw }}$$来量化特征形成过程，其中涉及对不同类型数据点（如$$\mathcal{A}_{i}^{t}$$和 $$\mathcal{A}_{random}$$）的相似度计算，可采用余弦相似度或Jaccard相似度等度量方法，同时在激活空间和特征空间分别进行分析。
		- 提出了特征形成进度度量（Progress Measure），用于量化特征在训练过程中变得完善的程度。
	- **实验结果**：
		- 可视化结果显示从训练开始时的随机激活到后期的语义连贯聚类，清晰呈现了特征形成过程；进度度量结果表明特征形成是渐进的过程，且不同类型特征在训练中有不同的动态变化，同时验证了该度量方法对不同相似度度量的稳健性，即不同度量下总体趋势一致，进一步揭示了特征形成的本质。
		- **图3**：通过玩具示例说明了特征区域和激活水平的概念。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=N2UwMGFmMmE2MWIyZjI2ZjhlZmIyZTViZDJkNTM3YjBfcnNmNXlZb2d5SkQ3eUl6QXY0VVdlWmJidnhKNGgzUXJfVG9rZW46UHFxT2JJclEyb01wUG54Tk5zWmNLR2hHbnVkXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
		- **图4**：展示了从训练开始时的随机激活到后期的语义连贯簇的转变。
		- **图5**：展示了在激活空间和特征空间中不同特征的进度度量。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=MWI5YTMzZjk1NGMwMTM3YzMyMGE3YjVmNWEzMTdkZDRfWEdLU3JlYjFBaTJDOHdTSlZyWHRER1hSVHlmck8zd0hfVG9rZW46SkJUU2JIeVVub0JjdUp4ZTdQb2NGcVAzbkhlXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
- ## **特征漂移分析实验**
  collapsed:: true
	- **实验目的**：分析特征方向在训练过程中是否漂移，或者是否早期就稳定下来。
	- **实验设置**：
		- 计算不同训练检查点的解码器向量之间的余弦相似度，从两个角度进行分析，
			- 一是以检查点为中心查看所有特征在给定时间相对于最终状态的余弦相似度分布（图 6(a)），
			- 二是以特征为中心展示特定特征与最终状态的相似度在所有训练检查点的演变（图 6(b)），
			- 同时定义特征轨迹（如 $$\mathcal{J}_{i}=\left\{W_{:, i}^{d e c}[1], W_{:, i}^{d e c}[2],..., W_{:, i}^{d e c}\left[T_{final }\right]\right\}$$来观察特征在训练过程中的方向变化，区分特征形成前后的阶段以深入理解方向漂移与特征形成的关系。
		- 从检查点中心视角和特征中心视角分析解码器向量的演变。
	- **实验结果**：
		- 发现特征方向在训练初期有显著漂移，在特征形成后仍继续变化，最终稳定在最终状态，且特征轨迹从无形成阶段到形成阶段的转变也体现了这一动态过程，表明在整个训练过程中特征几何结构不断调整直至稳定。
		- **图6**：展示了特征方向在训练过程中的全局对齐趋势和个别特征动态。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=OGFjMmRkYTdhMzY0MjY2ZDRjNGQ5ZDdjZjNmNjU5NjNfeHc4S2hwdkFoZG1UemJUTXA1cExPZjY4Q0FJZmlTOXJfVG9rZW46TFNUZWJGOUkwbzlrY294R2hIR2NScGkybjFZXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
		- **图7**：展示了特征轨迹，即解码器向量在训练检查点之间的方向变化。
		  
		  ![](https://m-a-p-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=YjlhZmQ5ZjMwYWE5NTg1MGQ3NTgzYTVkNzRlMWUxMjNfalBtQ2xsZVp1a3BicjA0ZUFkeUI3NndMM3h0VWV5Rk1fVG9rZW46QkJIcWJpMEVzb0tKU2Z4VHNIQ2NBNmNWbk5NXzE3MzY5OTk3MDQ6MTczNzAwMzMwNF9WNA)
- ## **不同模型规模验证实**
  collapsed:: true
	- **实验设置**：对Pythia - 160m - deduped和Pythia - 1.4b - deduped模型进行与上述主要实验类似的操作，包括在特定层前的残差流上训练SAE、提取特征、进行可视化（UMAP投影）、计算进度度量、分析解码器余弦相似度和特征轨迹等，以验证研究结果在不同模型规模下的一致性。
	- **结果分析**（对应图 9 - 16）：得到的结果与Pythia - 410m - deduped模型的结果紧密一致，如在特征演变阶段、转换模式、形成过程和漂移趋势等方面都表现出相似的规律，充分证明了研究方法和结论的通用性与可扩展性，适用于不同规模的语言模型。