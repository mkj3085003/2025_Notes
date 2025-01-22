- https://github.com/NainaniJatinZ/ScalableSAECircuits/blob/main/ScalableSAECircuits_Colab.ipynb
- https://www.lesswrong.com/posts/PkeB4TLxgaNnSmddg/scaling-sparse-feature-circuit-finding-to-gemma-9b
- **将 SAE 电路扩展到大型模型**：通过仅将稀疏自动编码器间隔放置在残差流中，我们可以在与 Gemma 9B 一样大的模型中找到电路，而无需为每个变压器层训练 SAE。
- **电路查找**：开发了一种更高效的电路查找算法，通过优化稀疏自编码器（SAE）潜在特征的二进制掩码来识别模型的关键子图。这种方法比现有的基于阈值的方法（如归因修补（Attribution Patching）或积分梯度（Integrated Gradients））更有效
- #### ** 研究背景**
	- **目标**：开发一种可扩展的方法，用于在大型语言模型（如Gemma 9B）中发现稀疏特征电路（Sparse Feature Circuits, SFCs），以提高模型行为的可解释性。
	- **挑战**：现有方法在扩展到全尺寸大型语言模型（LLMs）时面临显著挑战，尤其是需要在每个Transformer层放置稀疏自编码器（SAEs），导致计算成本过高。
- #### **研究方法**
	- **稀疏特征电路（Sparse Feature Circuits）**
		- **稀疏自编码器（SAEs）**：将模型激活投影到一个稀疏且可解释的基底中，每个潜在特征（latent）捕捉一个单一概念。
		- **问题**：尽管SAEs提高了可解释性，但现有方法需要在每个层和组件类型（MLP、注意力、残差流）放置SAEs，这在大型模型中不切实际。
	- **电路发现（Circuit Finding）**
		- **传统方法**：通过计算组件对任务相关损失函数的间接效应（IE）来识别电路。常用方法包括归因修补（Attribution Patching）和积分梯度（Integrated Gradients）。
		- **问题**：这些方法独立评分每个组件的重要性，忽略了组件之间的集体行为和自一致性。
	- **间隔放置残差流SAEs**：仅在模型的残差流中每隔几层放置SAEs，而不是在每个层都放置。这种方法减少了计算开销，并产生了更简洁、更可解释的电路。
	- **二进制掩码优化**：通过连续稀疏化（continuous sparsification）优化SAE潜在特征的二进制掩码，选择对任务性能至关重要的最小特征集合。这种方法比基于阈值的方法（如积分梯度）更有效。
- #### **实验设计**
	- **任务选择**：包括主谓一致（Subject-Verb Agreement, SVA）、字典键错误检测（Dictionary Key Error Detection）和Python代码输出预测等任务。
	- **电路发现**：使用间隔放置的残差流SAEs发现电路，并通过二进制掩码优化选择电路组件。
	- **性能评估**：通过以下指标评估电路的性能：
		- **保真度（Faithfulness）**：电路性能与模型性能的比值。
		- **完整性（Completeness）**：移除电路组件对模型性能的影响。
		- **稳定性（Stability）**：不同超参数设置下电路的一致性。
		- **因果故事（Causal Story）**：电路揭示的模型行为机制。
- #### **实验结果**
	- **性能恢复（Performance Recovery）**
		- **代码输出预测任务**：
			- 字典键错误检测任务中，二进制掩码优化显著优于积分梯度，能够以更少的潜在特征恢复更高的性能。
			- 列表索引任务中，二进制掩码优化同样优于积分梯度。
		- **主谓一致任务（SVA）**：
			- 二进制掩码优化和积分梯度的性能接近，但二进制掩码优化能够以更少的潜在特征实现更高的保真度。
		- **间接宾语识别任务（IOI）**：
			- 该任务依赖于注意力机制，而本方法仅使用残差流SAEs，因此需要更多的潜在特征来恢复性能。然而，二进制掩码优化仍然优于积分梯度。
	- **完整性(Completeness)**
		- 通过随机移除电路组件并比较模型和电路的性能变化，验证电路的完整性。结果显示，模型和电路在移除组件后的性能变化相似，表明电路确实捕捉了模型的关键组件。
	- **稳定性（Stability）**
		- 通过改变稀疏化超参数多次训练二进制掩码，发现不同超参数设置下电路组件具有很强的嵌套结构，表明方法能够可靠地识别核心电路组件。
- #### 案例研究：代码输出预测
	- **任务**：预测Python代码的输出，包括字典键错误检测和列表索引任务。
	- **发现的机制**：
		- 模型依赖于“检测重复”潜在特征来判断键是否存在，并输出对应的值。如果未检测到重复，则生成错误标记（如Traceback）。
		- **漏洞**：模型过度依赖重复检测潜在特征。利用这一知识，可以构造对抗性字典，使模型产生错误输出。
- #### **研究意义与局限性**
	- **意义**：
		- 提出了一种可扩展且可解释的电路发现方法，能够显著减少计算开销，同时生成更简洁、更可解释的电路。
		- 该方法在大型语言模型（如Gemma 9B）上表现出色，揭示了模型行为的清晰算法模式。
	- **局限性**：
		- 该方法仅关注“什么”是必要的，而不涉及“如何”或“何时”计算。
		- 对于某些任务（如IOI），需要在每个SAE中包含大量潜在特征以传递信息，这使得电路分析变得更加困难。
- #### **未来研究方向**
	- **扩展到更复杂的任务**：尝试理解语言模型如何执行工具使用、通用代码解释和数学推理等任务。
	- **改进SAE放置策略**：探索最佳的SAE数量和位置，以进一步优化电路发现。
	- **应用于边缘**：将方法扩展到电路中的边缘，以发现重要的连接。
	- **非模板化数据**：开发方法以处理非模板化任务，例如通过学习角色路由器和模型子集来分析不同令牌角色。
	- **迭代积分梯度**：探索迭代积分梯度方法，以选择更一致的电路组件。
- 代码
	- **`data`**：存储数据的 JSON 文件，按任务类型（如 `codereason`、`ioi`、`sva`）进行分类存储。
	- **`masks`**：存储掩码相关的数据，同样按任务类型分类存储。
	- SAEMasks 类
		- `SAEMasks` 类用于管理和应用一组掩码（masks），这些掩码在神经网络的不同钩子点（hook points）上操作。该类支持掩码的应用、统计打印以及保存和加载模型。
		- 构造方法
			- ```python
			  def __init__(self, hook_points, masks):
			      super().__init__()
			      self.hook_points = hook_points  # hook points 是钩子点的名称列表
			      self.masks = masks  # 掩码列表，长度与钩子点数量一致
			  ```
		- 前向传播
			- ```python
			  def forward(self, x, sae_hook_point, mean_ablation=None):
			      index = self.hook_points.index(sae_hook_point)  # 查找钩子点对应的掩码
			      mask = self.masks[index]  # 获取相应的掩码
			      censored_activations = torch.ones_like(x)  # 创建一个和输入张量大小相同的全1张量
			      if mean_ablation is not None:
			          censored_activations = censored_activations * mean_ablation
			      else:
			          censored_activations = censored_activations * 0
			  
			      diff_to_x = x - censored_activations  # 计算原始张量与掩蔽张量的差异
			      return censored_activations + diff_to_x * mask  # 应用掩码
			  
			  ```
		- 打印掩码统计信息
			- ```python
			  def print_mask_statistics(self):
			      for i, mask in enumerate(self.masks):
			          shape = list(mask.shape)
			          total_latents = mask.numel()
			          total_on = mask.sum().item()  # 统计掩码中值为1的元素数量
			  
			          if len(shape) == 1:
			              avg_on_per_token = total_on  # 只有一个token时直接等于总数
			          elif len(shape) == 2:
			              seq_len = shape[0]
			              avg_on_per_token = total_on / seq_len if seq_len > 0 else 0
			          else:
			              seq_len = shape[0]
			              avg_on_per_token = total_on / seq_len if seq_len > 0 else 0
			  
			          print(f"Statistics for mask '{self.hook_points[i]}':")
			          print(f"  - Shape: {shape}")
			          print(f"  - Total latents: {total_latents}")
			          print(f"  - Latents ON (mask=1): {int(total_on)}")
			          print(f"  - Average ON per token: {avg_on_per_token:.4f}\n")
			  
			  ```
		- 保存和加载
			- ```python
			  ```