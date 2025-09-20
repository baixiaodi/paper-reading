# 日期

## 2025.09.22

# 论文标题

## [EnYOLO: A Real-Time Framework for Domain-Adaptive Underwater Object Detection with Image Enhancement](https://ieeexplore.ieee.org/abstract/document/10610639)
![content](https://github.com/INDTLab/Literature-Review/blob/7e0e84450b3a1a78fa46dfd25d32359202548541/yc/20250922/IMG/author.png)
# 摘要

In recent years, significant progress has been made in the field of underwater image enhancement (UIE). However, its practical utility for high-level vision tasks, such as underwater object detection (UOD) in Autonomous Underwater Vehicles (AUVs), remains relatively unexplored. It may be attributed to several factors: (1) Existing methods typically employ UIE as a pre-processing step, which inevitably introduces considerable computational overhead and latency. (2) The process of enhancing images prior to training object detectors may not necessarily yield performance improvements. (3) The complex underwater environments can induce significant domain shifts across different scenarios, seriously deteriorating the UOD performance. To address these challenges, we introduce EnYOLO, an integrated real-time framework designed for simultaneous UIE and UOD with domain-adaptation capability. Specifically, both the UIE and UOD task heads share the same network backbone and utilize a lightweight design. Furthermore, to ensure balanced training for both tasks, we present a multi-stage training strategy aimed at consistently enhancing their performance. Additionally, we propose a novel domain-adaptation strategy to align feature embeddings originating from diverse underwater environments. Comprehensive experiments demonstrate that our framework not only achieves state-of-the-art (SOTA) performance in both UIE and UOD tasks, but also shows superior adaptability when applied to different underwater scenarios. Our efficiency analysis further highlights the substantial potential of our framework for onboard deployment.

# 本论文解决什么问题
本文旨在解决水下目标检测中的三个核心问题：

1.  **计算延迟问题**：传统串联式处理（先增强后检测）带来的计算延迟高、难以实时部署的问题。
2.  **任务不匹配问题**：图像增强目标（视觉美观）与检测目标（特征识别）不匹配，可能导致性能下降的问题。
3.  **域偏移问题**：不同水下环境（如绿、蓝、浑浊水域）间存在巨大域偏移，导致模型泛化能力差的问题。

# 已有方法的优缺点

**1. 水下图像增强 (Underwater Image Enhancement - UIE)**
*   **UDCP [10] 等传统方法**
    *   **方法**：基于物理模型（如估计后向散射和透射率）。
    *   **优点**：在特定先验假设下能产生清晰图像；计算复杂度相对较低。
    *   **缺点**：**严重依赖先验假设**，在**复杂真实场景**中效果显著下降，泛化能力差。
*   **Wang et al. [12]**
    *   **方法**：基于Swin Transformer的UIE方法。
    *   **优点**：利用Swin Transformer的强大建模能力，能同时捕获局部特征和长程依赖，生成高质量图像。
    *   **缺点**：**计算复杂度高**，不适合**实时应用**和**嵌入式部署**。
*   **Huang et al. [14]**
    *   **方法**：半监督UIE方法，引入对比正则化。
    *   **优点**：降低对大量成对数据的依赖，提升了视觉质量。
    *   **缺点**：方法复杂，且目标仍是提升主观视觉质量，未考虑对**高层任务（如检测）的实用性**。
*   **Jamieson et al. [15]**
    *   **方法**：将水下成像模型与深度学习效率相结合。
    *   **优点**：实现了**实时**色彩校正，计算效率高。
    *   **缺点**：焦点完全局限于视觉质量增强，其**对高层任务的有效性未被探索**。

**2. 水下目标检测 (Underwater Object Detection - UOD)**
*   **Jiang et al. [5]**
    *   **方法**：使用WaterNet对图像进行**预处理**，然后再进行检测。
    *   **优点**：简单直接，在某些情况下能提升检测性能。
    *   **缺点**：**串联结构引入额外延迟，破坏实时性**；增强可能引入**有害伪影**，反而导致检测性能下降。
*   **Fan et al. [19]**
    *   **方法**：在**特征层面**对退化图像进行增强。
    *   **优点**：比像素层面增强更高效。
    *   **缺点**：仍是“先增强再检测”的思路，**未能实现真正的端到端联合优化**。
*   **Cheng et al. [21] (JADSNet)**
    *   **方法**：提出一个**端到端的多任务框架**，**联合训练**UIE和UOD。
    *   **优点**：通过端到端方式**共同优化**两个任务，避免了任务不匹配和误差累积。
    *   **缺点**：依赖**复杂的网络架构**，模型笨重、训练困难，**难以在实际中部署**。

**3. 水下域适应 (Underwater Domain Adaptation)**
*   **Uplavikar et al. [24]**
    *   **方法**：使用**对抗性学习**让UIE网络处理不同水体类型。
    *   **优点**：提出了用水下域适应来解决UIE任务的泛化问题。
    *   **缺点**：工作**仅限于图像增强层面**，未涉及**目标检测任务**的域适应。
*   **Chen et al. [9]**
    *   **方法**：通过**内容和风格分离**来实现不同域的UIE。
    *   **优点**：方法设计巧妙，能更好地适应不同水域风格变化。
    *   **缺点**：研究范畴**完全集中在UIE上**，未探索对下游任务（如检测）的适应能力。
*   **Liu et al. [25]**
    *   **方法**：使用**WCT2风格迁移**来增强检测器能力（域泛化）。
    *   **优点**：通过风格化增加数据多样性，且不需要目标域数据。
    *   **缺点**：**没有直接对齐特征分布**来解决域偏移，且未集成到检测器中进行端到端优化。

# 本文采用什么方法及其优缺点

**方法**：
本文提出EnYOLO框架。
![content](https://github.com/INDTLab/Literature-Review/blob/c0d83ae1b58705a91af492c8e00138b8a906eeaa/yc/20250922/IMG/EnYOLO%20framework.png)


![content](https://github.com/INDTLab/Literature-Review/blob/c0d83ae1b58705a91af492c8e00138b8a906eeaa/yc/20250922/IMG/UIE%20network.png)
1.  采用共享主干(CSPDarkNet53)与解耦任务头(UIE头 & UOD头)的一体化轻量设计，实现高效并行推理。
2.  设计了多阶段训练策略：
    *   **Burn-In阶段**：使用合成和真实数据分别监督学习两个任务。
    *   **Mutual-Learning阶段**：引入无监督损失并利用增强图像训练检测器，促进任务间知识迁移。
    *   **Domain-Adaptation阶段**：通过特征对齐损失(MSE+协方差)简化域适应过程。
3.  采用了一种轻量级的基于特征对齐的域适应方法。

**优点**：

1.  **高效实时**：框架高效实时
2.  **性能优异**：在UIE和UOD任务上均达到SOTA性能，尤其在跨域泛化能力上提升显著。
3.  **设计巧妙**：训练策略设计巧妙，域适应方法简单有效。

**缺点**：

1.  **UIE的局限性**：为确保检测性能，在图像细节恢复和极致视觉美观上做了权衡。
2.  **假设风险**："增强特征域不变"的假设在极端环境下可能存在局限。


# 使用的数据集和性能度量
**数据集**：UIE训练使用合成配对数据集Syrea(20,688对图像)；UOD训练与测试使用真实数据集DUO(4个类别，6,671训练图，1,111测试图)；UIE测试使用UIEB和DUO Test set。

**性能度量**：UIE任务使用UIQM、UCIQE、URanker等水下图像质量评价指标；UOD任务使用mAP(平均精度均值)及其在不同合成水域(green, blue, turbid)下的值。
![content](https://github.com/INDTLab/Literature-Review/blob/c0d83ae1b58705a91af492c8e00138b8a906eeaa/yc/20250922/IMG/table1.png)

![content](https://github.com/INDTLab/Literature-Review/blob/c0d83ae1b58705a91af492c8e00138b8a906eeaa/yc/20250922/IMG/table2.png)

![content](https://github.com/INDTLab/Literature-Review/blob/c0d83ae1b58705a91af492c8e00138b8a906eeaa/yc/20250922/IMG/table3.png)

![content](https://github.com/INDTLab/Literature-Review/blob/c0d83ae1b58705a91af492c8e00138b8a906eeaa/yc/20250922/IMG/table4.png)

# 与我们工作的相关性
本研究与我们"目标检测+水下图像增强"的方向高度契合。其文献综述清晰地指出了三大研究空白：

1.  **UIE与高层任务脱节**：现有UIE研究不关心其输出对检测任务是否真正有效。
2.  **UIE-UOD联合建模的困境**：串联方法有根本缺陷，而端到端方法（JADSNet）又过于复杂、不实用。
3.  **UOD域适应研究的缺失**：这是最关键的空白，前人工作几乎未触及目标检测模型在不同水域下的泛化难题。

因此，这篇论文的动机和创新点非常明确，其**轻量一体化框架**和**针对检测的域适应方法**为我们后续的研究提供了参考。

# 英文总结

This paper proposes EnYOLO, a novel and efficient framework that unifies underwater image enhancement (UIE) and object detection (UOD) in a single network for real-time application. It addresses key challenges including computational latency, task misalignment, and domain shift across varied underwater environments. The core innovations include a lightweight architecture with a shared backbone and decoupled task heads, a multi-stage training strategy for balanced learning, and a simple yet effective feature-alignment-based domain adaptation method. Extensive experiments demonstrate that EnYOLO achieves state-of-the-art performance on both tasks and exhibits superior generalization capability, making it highly suitable for onboard deployment on autonomous underwater vehicles. This work provides a valuable reference for our research in domain-adaptive underwater vision systems.
