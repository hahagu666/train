# 妹妹角色扮演模拟系统 - 完整架构设计文档

> 设计日期：2026-08-04
> 版本：v1.0（完整讨论定稿版）
> 状态：架构设计全部完成，待数据集质量评估后开始实现
> 文档说明：本文档完整收录了从第一次架构讨论到最终定稿的全部设计构思、模块划分、算法设计、数据结构和实现路线图。

---

## 设计讨论历程与背景

### 问题缘起
在纯finetune对话模型实践中，发现以下固有局限：
- **反应不连贯**：摸乳头但乳头可能没反应，只有整体"好舒服"
- **多器官配合不足**：难以自然写出多部位同时反应
- **模板化输出**：同样输入容易产生重复输出，缺乏变化
- **情绪错位**：明明是难过/不安状态却在开心呻吟
- **跨轮失忆**：多轮刺激不会累积，身体状态不持续
- **固定化映射**：纯函数分支或纯数据集输出都是确定性单值映射，缺乏真实感
- **数据集质量问题**：现有数据集存在缺男性器官描写、缺体液描写、多器官配合不足、回复过长、开头模式重复等问题

### 核心设计理念确立
经过多轮讨论，确定核心思想：
> **不是让模型"记住"所有反应，而是给模型配一个"身体物理引擎"——引擎计算生理层面的反应，模型只负责说话和叙事。**

### 关键讨论决策点
1. **部位是否用小模型？** → 初期不用，身体部位用发放式神经网络模拟，Layer1感受生成初期用多维概率采样，后期可替换1B-3B小模型
2. **记忆策略** → 全量存储+显著性过滤，只在相关时检索注入top3记忆（方案B）
3. **输入解析** → 初期规则Parser，预留接口后期升级小模型Parser
4. **多动作处理** → 拆分为有序子动作序列，顺序处理但最终状态整体融合
5. **歧义处理** → 用户说不明确时，根据客观状态计算反问/主动/等待概率，由模型选择
6. **身体部位组织** → 按功能区域组织，每个区域既有日常功能也有性功能（如嘴巴既吃饭也口交，乳头乳房统一归胸部）
7. **神经网络模拟** → 采用简化发放式神经网络（Spiking Network），模拟阈值、不应期、facilitation等真实神经特性
8. **高潮系统** → 以女性为中心详细设计，模型需了解男性高潮前后反应；多次高潮机制详细设计
9. **衣物系统** → 6层分层结构，多种中间状态（半脱/拉开/推到一边等），accessibility连续计算
10. **提示词策略** → 暂不最终确定，等数据集质量评估后再优化
11. **显存优化** → 8GB GPU可运行，7B主模型Q4_K_M量化，初期无额外小模型

---

## 目录

0. [设计讨论历程与背景](#设计讨论历程与背景)
1. [设计理念与背景](#1-设计理念与背景)
2. [整体架构概览](#2-整体架构概览)
3. [Layer 0: 身体-神经模拟系统](#3-layer-0-身体-神经模拟系统)
   - 3.1 [身体部位节点网络（Layer 0a）](#31-身体部位节点网络-layer-0a)
   - 3.2 [自主神经系统 ANS（Layer 0b）](#32-自主神经系统-ans-layer-0b)
   - 3.3 [认知情绪门控（Layer 0c）](#33-认知情绪门控-layer-0c)
   - 3.4 [经验-技能-新鲜感系统（Layer 0d）](#34-经验-技能-新鲜感系统-layer-0d)
   - 3.5 [动态敏感度调制（Layer 0e）](#35-动态敏感度调制-layer-0e)
   - 3.6 [神经信号传播算法](#36-神经信号传播算法)
   - 3.7 [高潮系统](#37-高潮系统)
4. [记忆系统](#4-记忆系统)
5. [世界与事件系统](#5-世界与事件系统)
   - 5.1 [世界状态](#51-世界状态)
   - 5.2 [时间系统](#52-时间系统)
   - 5.3 [事件引擎](#53-事件引擎)
   - 5.4 [衣物系统](#54-衣物系统)
   - 5.5 [主观认知层（MindState）](#55-主观认知层mindstate)
6. [输入解析层](#6-输入解析层)
   - 6.1 [感知解析器（Parser）](#61-感知解析器parser)
   - 6.2 [歧义处理机制](#62-歧义处理机制)
7. [节奏与行为控制](#7-节奏与行为控制)
   - 7.1 [回合节奏控制器（Beat Controller）](#71-回合节奏控制器beat-controller)
8. [叙事生成层](#8-叙事生成层)
   - 8.1 [Layer 1: 感受生成器](#81-layer-1-感受生成器)
   - 8.2 [Layer 2: 叙事合成（主模型）](#82-layer-2-叙事合成主模型)
9. [状态持久化](#9-状态持久化)
10. [代码框架结构](#10-代码框架结构)
11. [主对话循环流程](#11-主对话循环流程)
12. [关键设计决策记录](#12-关键设计决策记录)
13. [显存与性能考量](#13-显存与性能考量)
14. [实现路线图](#14-实现路线图)
15. [词汇多样性规范（数据集处理指南）](#15-词汇多样性规范数据集处理指南)

---

## 1. 设计理念与背景

### 1.1 问题缘起

纯finetune对话模型存在以下固有局限：
- **反应不连贯**：摸乳头但乳头可能没反应，只有整体"好舒服"
- **多器官配合不足**：难以自然写出多部位同时反应
- **模板化**：同样输入容易产生重复输出，缺乏变化
- **情绪错位**：明明是难过/不安状态却在开心呻吟
- **跨轮失忆**：多轮刺激不会累积，身体状态不持续
- **固定化**：纯函数分支或纯数据集输出都是确定性单值映射，缺乏真实感

### 1.2 核心设计思想

**不是让模型"记住"所有反应，而是给模型配一个"身体物理引擎"——引擎计算生理层面的反应，模型只负责说话和叙事。**

关键原则：
1. **连续模拟，非离散状态机**：身体参数是连续值，刺激以神经信号方式传播和衰减
2. **模块化感知，统一叙事**：身体各部位独立响应，但最终由语言模型整合为连贯输出
3. **去模板化**：通过多维度采样、状态依赖、神经噪声保证每次反应有差异
4. **心理-生理双向影响**：情绪可以抑制/增强生理反应，生理感受也影响情绪
5. **主观与客观分离**：角色不知道的信息不会出现在其反应中
6. **路径依赖**：反应取决于历史状态累积，同一刺激在不同时刻结果不同

---

## 2. 整体架构概览

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Parser（感知解析器）                                         │
│  · 动作/部位/强度/语言内容解析                                │
│  · 复合动作拆分为有序子动作序列                                │
│  · 歧义检测与标记                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  事件引擎 + 时间系统                                          │
│  · 计算动作耗时（受状态影响）                                  │
│  · 身体状态自然衰减                                          │
│  · 检查日程/随机/条件/中断事件                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  世界状态更新                                                 │
│  · 衣物系统（分层状态+accessibility计算）                      │
│  · 位置/姿态变化                                             │
│  · 人物位置/注意力变化                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  歧义处理                                                    │
│  · 置信度高→直接处理                                         │
│  · 置信度低→计算反问/主动/等待概率 → 传递给模型                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: 身体-神经模拟系统                                   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. 子动作按顺序注入刺激（前序动作状态影响后序）        │    │
│  │ 2. 发放式神经网络信号传播（阈值/不应期/facilitation）│    │
│  │ 3. ANS全局更新（心跳/呼吸/体温/肌肉张力）             │    │
│  │ 4. 认知门控调制（恐惧/信任抑制或增强信号）            │    │
│  │ 5. 高潮系统监测（累积→平台→临界点→收缩波）           │    │
│  │ 6. 敏感度动态调制（疲劳/适应/生理/注意力）            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  输出：所有节点状态delta + ANS状态 + 高潮阶段                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  情绪-认知-记忆系统                                           │
│  · 情绪混合向量（非离散标签，连续值混合）                      │
│  · 记忆检索（相关事件注入prompt）                              │
│  · 经验/技能计算（有效mastery+遗忘曲线+新鲜感）               │
│  · 主观认知层（MindState区分已知/未知）                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  节奏控制器（Beat Controller）                                │
│  · 决定beat类型：react_continue / prompt / initiate / wait  │
│  · 全部由客观状态因素加权计算，非纯随机                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: 感受生成器                                          │
│  初期：状态向量→多维度概率采样→结构化感受片段                   │
│  后期：可替换为1B-3B小模型专门生成                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Prompt构造 + 主模型（7B finetune）叙事合成           │
│  · 角色设定                                                  │
│  · 当前身体/情绪状态摘要（自然语言，非JSON）                    │
│  · 结构化感受片段                                            │
│  · 相关记忆（自然语言回忆提示）                                │
│  · 环境/危险/衣物信息                                         │
│  · 对话历史                                                  │
│  · 歧义选项（反问/主动/等待）                                  │
│  · 输出：连贯的动作描写 + 对话台词 + 声音反应                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  后处理 + 状态更新                                            │
│  · 解析模型输出中的主动动作，更新对应状态                       │
│  · 更新技能 times / last_practiced                           │
│  · 记忆固化（计算salience → 存入对应记忆层级）                 │
│  · 更新衣物/位置状态（如果描写中涉及）                         │
│  · 持久化保存到JSON                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 0: 身体-神经模拟系统

Layer 0 是整个系统的核心，负责模拟身体的生理反应。它不是离散的状态机，而是一个连续的、有记忆的、受心理调制的模拟系统。

### 3.1 身体部位节点网络（Layer 0a）

按**功能区域（Anatomical Region）+ 子部位精细参数**组织。每个区域既有日常功能也有性功能，子部位有独立的连续参数。

#### 3.1.1 头部区域（head）

```python
"head": {
    "sub_parts": {
        "lips": {
            "arousal": 0.0,        # 局部兴奋度 0-1
            "sensitivity": 0.8,   # 基础敏感度
            "swollen": 0.0,       # 肿胀度（亲吻后）
            "tremor": 0.0,        # 颤抖
            "parted": 0.0,        # 张开程度
        },
        "tongue": {
            "arousal": 0.0,
            "sensitivity": 0.9,
            "active": False,      # 是否主动在动
        },
        "throat": {
            "arousal": 0.0,
            "sensitivity": 0.5,
            "gag_reflex": 1.0,    # 咽反射强度（经验多了降低）
            "contraction": 0.0,   # 收缩（口交时）
            "relaxed": 0.1,       # 放松程度
        },
        "mouth_interior": {
            "arousal": 0.0,
            "saliva": 0.5,        # 唾液量
            "wetness": 0.5,       # 湿润度
        },
        "earlobes": {
            "arousal": 0.0,
            "sensitivity": 0.9,
            "tingling": 0.0,      # 发麻感
        },
        "neck": {
            "arousal": 0.0,
            "sensitivity": 0.85,
            "pulse_point_tension": 0.0,  # 脉搏点紧张
            "bruising": 0.0,      # 吻痕
        },
    },
    "functions": ["kiss", "blowjob", "talk", "eat", "breathe", "moan"],
    "daily": True,
    "sexual": True,
}
```

#### 3.1.2 胸部区域（chest）

```python
"chest": {
    "sub_parts": {
        "breast_left": {
            "arousal": 0.0,
            "congestion": 0.0,    # 充血
            "swelling": 0.0,      # 肿胀
            "sensitivity": 1.0,
            "heaviness": 0.0,     # 沉重感
            "fondled_duration": 0.0,  # 被抚摸时长
        },
        "breast_right": { ... },
        "nipple_left": {
            "arousal": 0.0,
            "congestion": 0.0,
            "erection": 0.0,      # 勃起/硬挺程度 0-1
            "sensitivity": 1.2,   # 乳头特别敏感
            "pain": 0.0,
            "touched_first_time": True,  # 是否第一次被碰（初夜）
        },
        "nipple_right": { ... },
        "areola_left": {
            "arousal": 0.0,
            "puffiness": 0.0,     # 乳晕肿胀
            "sensitivity": 1.0,
        },
        "areola_right": { ... },
    },
    "functions": ["suckle", "squeeze", "kiss", "press_against", "pinch"],
    "daily": True,
    "sexual": True,
}
```

#### 3.1.3 手臂/手部（arms_hands）

```python
"arms_hands": {
    "sub_parts": {
        "fingers": {
            "arousal": 0.0,
            "grip_strength": 0.5,  # 握力（抓床单/抓背）
            "tremor": 0.0,
            "curled": 0.0,        # 蜷缩（抓紧）
        },
        "palms": {
            "arousal": 0.0,
            "sweat": 0.0,
        },
        "wrists": {
            "arousal": 0.0,
            "held": False,        # 被按住/抓住
        },
        "upper_arms": {
            "arousal": 0.0,
            "goosebumps": 0.0,
        },
    },
    "functions": ["grab", "hold", "push", "pull", "touch_self", "touch_partner", "cling"],
    "daily": True,
    "sexual": True,
}
```

#### 3.1.4 核心躯干/腰腹（torso_core）

```python
"torso_core": {
    "sub_parts": {
        "waist": {
            "arousal": 0.0,
            "sensitivity": 0.7,
            "weakness": 0.0,      # 发软
            "ticklish": 0.5,      # 怕痒程度
            "grip_marks": 0.0,    # 被抓的痕迹
        },
        "abdomen": {
            "arousal": 0.0,
            "muscle_tension": 0.0,
            "butterflies": 0.0,   # 小鹿乱撞感
            "flinching": 0.0,     # 抽搐/躲闪
        },
        "back": {
            "arousal": 0.0,
            "arch": 0.0,          # 弓起程度
            "sensitivity": 0.6,
            "goosebumps": 0.0,
            "scratch_marks": 0.0,
        },
        "buttocks": {
            "arousal": 0.0,
            "tension": 0.0,
            "sensitivity": 0.8,
            "tingling": 0.0,
            "redness": 0.0,       # 被拍后红
            "spanked_sting": 0.0,
        },
        "navel": {
            "arousal": 0.0,
            "sensitivity": 0.5,
            "ticklish": 0.8,
        },
    },
    "functions": ["hug", "hold", "spank", "arch", "wrap_around", "grind"],
    "daily": True,
    "sexual": True,
}
```

#### 3.1.5 女性生殖区域（genital_female）——最复杂

```python
"genital_female": {
    "sub_parts": {
        "mons_pubis": {
            "arousal": 0.0,
            "sensitivity": 0.4,
        },
        "outer_labia": {
            "arousal": 0.0,
            "swelling": 0.0,
            "sensitivity": 0.6,
            "parted": 0.0,
        },
        "inner_labia": {
            "arousal": 0.0,
            "swelling": 0.0,
            "engorgement": 0.0,   # 充血膨胀
            "sensitivity": 0.9,
            "color_change": 0.0,  # 兴奋时颜色变深
        },
        "clitoral_hood": {
            "arousal": 0.0,
            "retracted": 0.0,    # 兴奋时包皮退缩露出阴蒂
        },
        "clitoris": {
            "arousal": 0.0,
            "engorgement": 0.0,
            "erection": 0.0,     # 阴蒂勃起
            "sensitivity": 1.5,  # 全身最敏感
            "oversensitive_post_orgasm": False,  # 高潮后过度敏感
            "pain_from_overstimulation": 0.0,
        },
        "vaginal_vestibule": {
            "arousal": 0.0,
            "wetness": 0.0,
            "sensitivity": 1.0,
            "tingling": 0.0,
        },
        "vaginal_opening": {
            "arousal": 0.0,
            "relaxation": 0.1,   # 放松/张开程度（初夜很紧）
            "stretch": 0.0,      # 被撑开程度
            "sensitivity": 0.9,
            "first_time": True,
            "hymen_intact": True,
            "burning": 0.0,      # 初夜/干涩进入的灼痛感
        },
        "vaginal_canal": {
            "arousal": 0.0,
            "wetness": 0.0,
            "contraction": 0.0,  # 收缩强度
            "sensitivity": 0.8,
            "depth_penetrated": 0.0,  # 被插入深度 0-1
            "girth_stretch": 0.0,    # 围度撑开程度
            "tent_lubrication": 0.0, # 帐篷效应：深处润滑（唤起充分后产生）
            "muscle_tone": 0.5,      # PC肌张力
            "ejaculation_volume": 0.0,  # 潮吹量
            "tent_effect": 0.0,     # 兴奋时阴道扩张（帐篷效应）
        },
        "g_spot": {
            "arousal": 0.0,
            "swelling": 0.0,      # 充分唤起后G点肿胀突起
            "sensitivity": 0.8,
            "stimulation": 0.0,
            "engorged": False,
        },
        "cervix": {
            "arousal": 0.0,
            "sensitivity": 0.3,  # 宫颈平时不敏感
            "position": "high",  # 兴奋时宫颈抬起（high/low）
            "pain_from_bumping": 0.0,  # 顶到宫颈的疼痛
            "reached": False,
            "dip": False,        # 高潮时宫颈dipping反射
            "sensitivity_increased": False,  # 极度兴奋时宫颈变得敏感
        },
        "anus": {
            "arousal": 0.0,
            "sensitivity": 0.7,
            "contraction": 0.0,
            "relaxation": 0.1,
            "fullness": 0.0,
        },
        "perineum": {
            "arousal": 0.0,
            "sensitivity": 0.6,
        },
    },
    "functions": [
        "penetrate", "clit_stim", "gspot_stim", "cervix_stim",
        "oral_cunnilingus", "finger", "contract", "orgasm", "lubricate",
        "clench", "gush", "throb"
    ],
    "daily": False,
    "sexual": True,
    "derived_properties": [
        "total_wetness",          # 综合润滑度
        "overall_tightness",      # 综合紧度
        "readiness_for_penetration"  # 是否足够润滑放松可以进入
    ],
}
```

#### 3.1.6 腿/脚（legs_feet）

```python
"legs_feet": {
    "sub_parts": {
        "thighs": {
            "arousal": 0.0,
            "tension": 0.0,
            "tremor": 0.0,
            "sensitivity": 0.7,
            "openness": 0.2,    # 腿张开程度
            "quivering": 0.0,
        },
        "inner_thighs": {
            "arousal": 0.0,
            "sensitivity": 0.9,
            "tingling": 0.0,
            "goosebumps": 0.0,
        },
        "calves": {
            "arousal": 0.0,
            "tension": 0.0,
            "cramping": 0.0,   # 抽筋（高潮时绷紧）
        },
        "knees": {
            "weakness": 0.0,
            "buckling": 0.0,   # 膝盖发软要跪
        },
        "feet": {
            "arousal": 0.0,
            "curling": 0.0,    # 脚趾蜷缩
            "sensitivity": 0.4,
        },
    },
    "functions": ["wrap_around", "spread", "brace", "shake", "lock_ankles", "wrap_around_waist"],
    "daily": True,
    "sexual": True,
}
```

#### 3.1.7 皮肤/全身（skin_global）

不是具体部位，是全局皮肤状态：

```python
"skin_global": {
    "flush": 0.0,              # 性潮红 0-1（胸口/脖子/脸）
    "goosebumps": 0.0,
    "sweat": 0.0,
    "temperature": 36.5,       # 体温
    "shiver": 0.0,             # 颤抖
    "sensitive_to_touch": 0.3,
    "cold_flash": 0.0,         # 情绪冲击时的一阵发冷
    "hot_flash": 0.0,
}
```

#### 3.1.8 男性伙伴状态感知（partner_state）

不需要模拟男性全部生理，但妹妹需要感知到对方状态：

```python
partner_state = {
    "penis": {
        "erection": 0.0,             # 0-1 勃起硬度（通过触感感知）
        "size_impression": "normal", # 她的主观感受
        "throbbing": 0.0,            # 跳动（高潮前/中）
        "pre_cum": 0.0,              # 前泪腺液（她能摸到/尝到）
        "ejaculation_phase": None,   # None/imminent/happening/recent
        "ejaculate_volume": 0.0,
        "inside_her": False,
        "position": "outside",
        "sensitivity_post_orgasm": 0.0,
        "temperature_inside": 0.0,   # 射在里面的温度感
    },
    "reaction": {
        "breathing_heavy": False,
        "gripping": False,           # 他抓着她
        "thrusting": False,
        "thrust_speed": 0.0,
        "thrust_depth": 0.0,
        "making_sounds": False,
        "tensed": False,
        "pulling_out": False,
        "pushing_in_deeper": False,
        "pulsing_inside": False,     # 射精时在里面跳动
    },
    "orgasm_imminent_signs": [       # 她能观察到的"快射了"信号
        "thrusting_faster",
        "gripping_tight",
        "breath_holding",
        "guttural_sounds",
        "throbbing_inside",
        "swelling_more",
    ],
}
```

这些状态通过用户输入的描写推断，描写越明确状态越确定。

### 3.2 自主神经系统 ANS（Layer 0b）

自主神经系统控制全局生理反应，是局部刺激→全身反应的中继站。

```python
autonomic_nervous_system = {
    "sympathetic": 0.1,          # 交感神经兴奋度（战斗/逃跑/兴奋）0-1
    "parasympathetic": 0.9,      # 副交感（休息/消化/放松）0-1
    "arousal_global": 0.0,       # 全局性兴奋度 0-1（各部位加权平均+ANS调制）
    "heart_rate": 75,            # bpm（静息60-80，兴奋120-160，高潮150+）
    "breathing_rate": 14,        # 次/分钟（静息12-16，兴奋25-40，高潮40+）
    "breathing_depth": "normal", # normal/shallow/deep/gasping/holding
    "breathing_rhythm": "regular", # regular/irregular/panting
    "skin_flush": 0.0,           # 性潮红
    "skin_temperature": 36.5,
    "skin_sweat": 0.0,
    "saliva": 0.5,               # 唾液分泌
    "muscle_tone": 0.2,          # 全身肌张力（高潮时1.0全身强直）
    "tremor": 0.0,               # 全身颤抖
    "pupil_dilation": 0.0,       # 瞳孔放大
    "voice_breathiness": 0.0,    # 声音喘气程度
    "voice_tightness": 0.0,      # 声音紧绷
    "blood_pressure_systolic": 110,
    "dizzy": 0.0,                # 眩晕感（高潮后/过度换气）
    "vision_blur": 0.0,          # 视线模糊
    "ears_ringing": 0.0,         # 耳鸣（强高潮）
}
```

ANS从身体部位信号更新，更新后再广播回各部位形成闭环。

### 3.3 认知情绪门控（Layer 0c）

认知层不产生生理反应，但它**调制**生理反应的强度。核心是`cognitive_gate`和情绪混合向量。

#### 3.3.1 情绪不是离散标签，是连续混合向量

```python
emotion_state = {
    "primary": "happy",          # 当前主导情绪（用于角色整体倾向）
    "blend": {
        "pleasure": 0.0,        # 生理快感
        "shame": 0.3,           # 羞耻
        "trust": 0.7,           # 信任
        "fear": 0.0,            # 恐惧
        "hurt": 0.0,            # 受伤/委屈
        "jealousy": 0.0,        # 嫉妒
        "anxiety": 0.0,         # 焦虑/不安
        "happiness": 0.6,       # 开心
        "love": 0.5,            # 爱意
        "sleepy": 0.0,          # 困倦
        "frustration": 0.0,     # 挫败/不满足
        "satisfaction": 0.0,    # 满足
        "anticipation": 0.0,    # 期待
        "embarrassment": 0.2,   # 尴尬（比shame轻）
        "overwhelm": 0.0,       # 无法承受
        "loss_of_control": 0.0, # 失控感
        "inevitability": 0.0,   # 高潮前的"要去了"不可避免感
        "pain": 0.0,            # 疼痛（初夜/过度刺激）
    },
    "conflict_level": 0.0,       # 情绪冲突度（又怕又爽=高冲突）
    "cognitive_gate": 0.8,       # 认知闸门 0-1（1=完全放开，0=完全抑制）
    "attention_focus": ["user"], # 当前注意力焦点（可以是多个）
    "inhibited": False,          # 是否被抑制住
    "suppressing_sounds": False, # 是否在忍住不出声
    "suppressing_movements": False, # 是否在忍住不动
    "interrupt_triggered": False,   # 中断信号（情绪冲击导致动作中断）
}
```

关键设计：**情绪是"上色"不是"开关"**。
- 不安时乳头还是会硬（生理不自主反应），但心理感受是"羞耻+身体怎么不听话"
- 恐惧时身体可能还是会湿润（自主神经反射），但肌肉僵硬、想推开
- 这种"身体有反应但心理抗拒"的矛盾状态是真实的，比单纯"害怕→没反应"更真实

#### 3.3.2 认知闸门的影响因素

```python
def update_cognitive_gate(state, world):
    gate = 1.0
    
    # 基础信任（越信任越放开）
    gate *= (0.3 + 0.7 * state.relationship.trust_base)
    
    # 环境安全度
    gate *= world.privacy_level
    
    # 情绪影响
    if state.emotion.blend["fear"] > 0.5:
        gate *= (1 - state.emotion.blend["fear"] * 0.7)
    if state.emotion.blend["hurt"] > 0.4:
        gate *= (1 - state.emotion.blend["hurt"] * 0.5)
    if state.emotion.blend["shame"] > 0.6:
        gate *= (1 - state.emotion.blend["shame"] * 0.2)  # 羞耻部分抑制但不完全关闭
    if state.emotion.blend["trust"] > 0.8 and state.emotion.blend["pleasure"] > 0.5:
        gate *= 1.2  # 信任+快感时闸门更开（但上限1.0）
    gate = min(gate, 1.0)
    
    # 禁果效应：不完全安全的环境（父母在家但门锁了）反而增加arousal_base
    if 0.3 < world.privacy_level < 0.7:
        state.ans.arousal_global += 0.1  # 偷偷摸摸的刺激感
    
    # 明确意愿（即使环境危险，她愿意的话gate可以开放）
    if state.emotion.blend["anticipation"] > 0.7:
        gate = max(gate, 0.7)
    
    return gate
```

### 3.4 经验-技能-新鲜感系统（Layer 0d）

经验不是一个从0到100的线性值，分多个维度独立计算。

#### 3.4.1 技能熟练度

每个具体行为有独立的熟练度：

```python
skills = {
    "kissing": {
        "mastery": 0.7,                # 熟练度 0-1
        "last_practiced": "2026-08-03", # 上次练习时间
        "times": 30,                   # 总次数
        "half_life_days": 30,          # 遗忘半衰期（越熟练半衰期越长）
    },
    "breast_touch": {"mastery": 0.6, "last_practiced": "2026-08-02", "times": 20, "half_life_days": 30},
    "handjob_giving": {"mastery": 0.3, "last_practiced": None, "times": 2, "half_life_days": 14},
    "blowjob": {"mastery": 0.3, "last_practiced": "2026-07-20", "times": 3, "half_life_days": 14},
    "cowgirl": {"mastery": 0.2, "last_practiced": None, "times": 1, "half_life_days": 10},
    "missionary": {"mastery": 0.4, "last_practiced": "2026-08-01", "times": 5, "half_life_days": 20},
    "doggy": {"mastery": 0.1, "last_practiced": None, "times": 0, "half_life_days": 7},
    "dirty_talk": {"mastery": 0.1, "last_practiced": None, "times": 0, "half_life_days": 7},
    "initiating": {"mastery": 0.1, "last_practiced": None, "times": 1, "half_life_days": 7},
    "self_pleasure_knowledge": {"mastery": 0.4, "last_practiced": "2026-07-25", "times": 10, "half_life_days": 30},
    "riding_motion": {"mastery": 0.15, ...},
    "deepthroat": {"mastery": 0.05, ...},
    # ...
}
```

**有效熟练度计算（含遗忘曲线）**：

```python
import math
from datetime import datetime

def calc_effective_mastery(skill):
    if skill["times"] == 0:
        return 0.0
    
    mastery = skill["mastery"]
    
    if skill["last_practiced"]:
        last = datetime.fromisoformat(skill["last_practiced"])
        days = (datetime.now() - last).days
        
        # 艾宾浩斯式遗忘：retention = exp(-days / half_life)
        half_life = skill["half_life_days"] * (0.5 + mastery * 0.5)  # 越熟练遗忘越慢
        retention = math.exp(-days / half_life)
        effective = mastery * retention
        
        # 不是完全忘记：身体有记忆，比从零开始快
        minimum_retention = 0.3 if mastery > 0.3 else mastery  # 最低保留基础
        effective = max(effective, minimum_retention)
    else:
        effective = 0.0
    
    return effective
```

结果示例：
- 口交mastery=0.4，15天没做 → 有效约0.25（知道怎么做但身体生疏了，有点笨拙迟疑）
- 接吻mastery=0.9，10天没做 → 有效约0.82（熟练的不会轻易忘）

#### 3.4.2 新鲜感/脱敏系统

```python
novelty_system = {
    "global_novelty": 1.0,           # 对哥哥的整体新鲜感
    "per_act": {},                   # 每个行为的新鲜感
    "per_location": {},              # 每个场景的新鲜感
    "session_fatigue": 0.0,          # 本次会话的疲劳度（同一姿势太久）
}

def update_novelty(skill_name, location, novelty_system, arousal_at_time):
    """每次行为后更新新鲜感"""
    # 做了→新鲜感下降（脱敏）
    current = novelty_system["per_act"].get(skill_name, 1.0)
    # 高arousal时的行为印象更深，脱敏更快
    decay_rate = 0.05 * (0.5 + arousal_at_time * 0.5)
    novelty_system["per_act"][skill_name] = current * (1 - decay_rate)
    
    # 同样场景也脱敏
    loc_current = novelty_system["per_location"].get(location, 1.0)
    novelty_system["per_location"][location] = loc_current * 0.95
    
    # 长时间不做→新鲜感回升
    # 在记忆维护时处理

def get_sensitivity_bonus(novelty_system, skill_name, location):
    """新鲜感对敏感度的加成"""
    act_nov = novelty_system["per_act"].get(skill_name, 1.0)
    loc_nov = novelty_system["per_location"].get(location, 1.0)
    global_nov = novelty_system["global_novelty"]
    
    # 新鲜感高→敏感度更高（久别胜新婚/新体验）
    # 新鲜感低→需要更强刺激才能有同样反应
    bonus = 0.5 + 0.5 * (act_nov * 0.4 + loc_nov * 0.3 + global_nov * 0.3)
    return bonus  # 0.5（脱敏严重）到1.5（非常新鲜）
```

#### 3.4.3 内感受能力（Body Awareness）

```python
body_awareness = 0.4  # 0-1，感知自己身体的能力
```

- 经验少(awareness=0.2)：身体有反应但自己没意识到，比如下面湿了没感觉
- 经验多(awareness=0.8)：能精准感知到自己哪里在反应、湿了多少、快高潮了
- 这个能力随经验增长，影响"她有没有注意到自己的身体反应"

#### 3.4.4 技能对主动行为的影响

熟练度影响"能不能做"，不是"主不主动做"。主动程度是多因素的：

```python
def calc_initiative_probability(state, action):
    mastery = skills[action].effective_mastery if action in skills else 0
    
    prob = 0.1  # 基础概率
    prob *= (0.2 + 0.8 * mastery)  # 越熟练越有能力主动
    prob *= (0.3 + 0.7 * state.relationship.trust_base)  # 越信任越敢
    prob *= (1 - state.emotion.blend["shame"] * 0.8)  # 羞耻压制主动
    prob *= state.ans.arousal_global * 1.5  # 越想要越主动
    prob *= state.emotion.blend["anticipation"] * 2  # 期待时主动
    prob *= state.personality.extraversion  # 性格外向程度
    
    if state.emotion.blend["hurt"] > 0.5:
        prob *= 0.1  # 受伤时不会主动
    if state.emotion.blend["anxiety"] > 0.6:
        prob *= 0.2  # 焦虑时不敢
    
    return min(prob, 1.0)
```

技能还影响**跨部位神经连接权重**：
- 没经验时乳头→阴道的连接弱（摸胸下面反应慢/弱）
- 经验多了通路被强化，摸胸下面很快有反应

### 3.5 动态敏感度调制（Layer 0e）

每个部位的有效敏感度不是固定值，受多种因素动态调制：

```python
def effective_sensitivity(part, state, world):
    sens = part.sensitivity  # 基础敏感度
    
    # 1. 当前唤起水平（唤起越高越敏感）
    sens *= (0.5 + state.ans.arousal_global * 1.0)
    
    # 2. 新鲜感加成
    sens *= get_sensitivity_bonus(novelty_system, current_action, world.location)
    
    # 3. 生理状态
    physical = state.sensitivity_modifiers.physical_state
    if part.name in ["nipple_left", "nipple_right", "breast_left", "breast_right"]:
        sens *= (1 - physical.chest_tightness * 0.4)
    if part.name.startswith("vagina") or part.name in ["clitoris", "inner_labia"]:
        sens *= (1 - physical.menstrual_cramps * 0.3)
        if physical.sore_genitals > 0.3:
            sens *= (1 - physical.sore_genitals * 0.5)
            part.pain += physical.sore_genitals * 0.3
    if physical.tired > 0.5:
        sens *= (1 - physical.tired * 0.3)
    if physical.horny_morning > 0.5:
        sens *= 1.3
    
    # 4. 高潮后状态
    if state.orgasm.phase == "resolution":
        time_since = state.orgasm.time_since_last_orgasm
        if time_since < 30:
            if part.name == "clitoris":
                sens *= 0.3  # 刚结束阴蒂极度敏感但触碰不舒服
                part.pain += 0.3
            elif part.name in ["vaginal_canal", "g_spot"]:
                sens *= 1.2  # 内部还是敏感
        elif time_since < 120 and state.orgasm.multi_orgasm_state.orgasms_so_far == 1:
            # 第一波高潮后短暂时间内是多重高潮窗口
            sens *= 1.1
    
    # 5. 适应（持续同样刺激→敏感度下降）
    if part.adaptation > 0:
        sens *= (1 - part.adaptation * 0.5)
    
    # 6. 注意力（注意力不在这个部位→感受被过滤）
    if part.name not in state.emotion.attention_focus and len(state.emotion.attention_focus) > 0:
        sens *= 0.6  # 注意力在别的地方，这个部位感受被弱化
    
    # 7. 情绪门控（认知闸门开放时敏感，关闭时迟钝）
    sens *= (0.3 + 0.7 * state.emotion.cognitive_gate)
    
    # 8. 刺激变化（换手法/换部位→重新激活）
    if state.current.stimulation_changed_in_last_10s:
        sens *= 1.2
    
    # 9. 动机抗疲劳（被激将/想证明自己时，疲劳积累变慢）
    if state.emotion.motivation > 0.7:
        sens *= (1 + state.emotion.motivation * 0.2)
    
    return max(sens, 0.05)  # 最低不低于5%
```

### 3.6 神经信号传播算法

采用简化的**发放式神经网络（Simplified Spiking Network）**，模拟神经元的关键特性：阈值发放、时间整合、不应期、双向传播、抑制信号、短期突触可塑性。

#### 3.6.1 神经节点模型

```python
import random

class NeuralNode:
    def __init__(self, name, region, params):
        self.name = name
        self.region = region
        
        # 膜电位类比（累积兴奋度）
        self.potential = 0.0       # 0-1.5，超阈值发放
        
        # 发放阈值
        self.base_threshold = params.get("base_threshold", 1.0)
        self.threshold = self.base_threshold
        
        # 基础敏感度
        self.sensitivity = params.get("sensitivity", 1.0)
        
        # 不应期
        self.refractory_period = 0.0    # 剩余不应期时间（秒）
        self.absolute_refractory = params.get("abs_refractory", 0.3)  # 绝对不应期
        self.relative_refractory = params.get("rel_refractory", 1.5)  # 相对不应期
        
        # 短期突触增强（本轮高频刺激后）
        self.facilitation = 1.0
        self.facilitation_decay = 0.95  # 每传播轮衰减
        
        # 发放历史
        self.last_fire_time = -100
        self.firing_rate = 0.0          # 当前发放频率
        
        # 传出连接 {target_node_name: weight}
        self.efferent = {}
        
        # 当前传入刺激（本轮累积）
        self.current_input = 0.0
    
    def stimulate(self, intensity, duration=1.0, cognitive_mod=1.0):
        """接收外部刺激或传入信号"""
        
        # 不应期处理
        effective_threshold = self.threshold
        if self.refractory_period > 0:
            if self.refractory_period > self.absolute_refractory:
                return  # 绝对不应期，完全不响应
            else:
                # 相对不应期阈值升高
                effective_threshold *= (1 + self.refractory_period * 0.5)
        
        # 时间整合：持续刺激累积更多
        input_signal = intensity * self.sensitivity * self.facilitation * duration
        
        # 认知调制（情绪抑制/增强）
        input_signal *= cognitive_mod
        
        # 神经噪声（每次有微小随机扰动，避免完全相同）
        input_signal *= random.uniform(0.92, 1.08)
        
        self.current_input += input_signal
        self.potential += input_signal
    
    def tick(self, dt=0.1):
        """推进一个时间步，返回 (是否发放, 输出信号)"""
        fired = False
        
        # 电位自然泄漏（不是永久累积）
        self.potential *= 0.85
        
        # 检查是否超过阈值
        if self.potential >= self.threshold and self.refractory_period <= 0:
            fired = True
            self.last_fire_time = current_time
            self.firing_rate = min(self.firing_rate + 0.1, 1.0)
            
            # 发放后超极化重置
            self.potential = -0.1
            self.refractory_period = self.absolute_refractory + self.relative_refractory
            
            # 短期突触增强（发放后更容易再次发放——越刺激越敏感）
            self.facilitation = min(self.facilitation * 1.1, 2.5)
        
        # 不应期倒计时
        if self.refractory_period > 0:
            self.refractory_period -= dt
        
        # facilitation自然衰减
        self.facilitation *= self.facilitation_decay
        
        # 输出信号：发放时全量，未发放时微弱梯度传导
        output = self.current_input if fired else self.current_input * 0.1
        self.current_input = 0
        
        return fired, output
```

#### 3.6.2 神经连接权重（基础）

```python
neural_connections = {
    # === 强连接（性反射弧，生理上的强神经通路）===
    ("nipple_left", "genital_female"): 0.70,
    ("nipple_right", "genital_female"): 0.70,
    ("clitoris", "vaginal_canal"): 0.80,
    ("clitoris", "uterus"): 0.60,
    ("g_spot", "cervix"): 0.70,
    ("clitoris", "g_spot"): 0.60,
    ("inner_thighs", "genital_female"): 0.60,
    ("neck", "chest"): 0.50,
    ("earlobes", "neck"): 0.60,
    ("buttocks", "genital_female"): 0.50,
    ("lips", "chest"): 0.40,
    ("lips", "genital_female"): 0.30,  # 亲吻时下面也有间接反应
    
    # === 中等连接 ===
    ("waist", "genital_female"): 0.35,
    ("back", "waist"): 0.30,
    ("navel", "genital_female"): 0.30,
    ("fingers", "arms"): 0.40,  # 牵手指也能有感觉
    
    # === 全身联动 ===
    ("skin_global", "all"): 0.20,
    
    # === 抑制性连接（认知/情绪通过这些通路抑制性反应）===
    ("cognitive_fear", "genital_female"): -0.50,   # 恐惧抑制性反应
    ("cognitive_fear", "skin_global"): -0.30,       # 恐惧→皮肤发冷
    ("cognitive_hurt", "genital_female"): -0.40,    # 受伤→下面没反应
    ("cognitive_shame", "skin_global"): -0.20,      # 羞耻→脸红但发冷
}
```

**经验对连接权重的调制**：
```python
def get_connection_weight(src, dst, skills):
    base = neural_connections.get((src, dst), 0.0)
    
    # 相关技能mastery增强连接
    if src in ["nipple_left", "nipple_right", "breast_left", "breast_right"]:
        base *= (0.5 + skills["breast_touch"].effective_mastery * 1.0)
    if src == "clitoris" or (src.startswith("vagina")):
        base *= (0.5 + max(
            skills.get("missionary", EffectiveMastery(0)).effective,
            skills.get("cowgirl", EffectiveMastery(0)).effective,
            skills.get("finger", EffectiveMastery(0)).effective,
        ) * 1.0)
    if dst == "throat":
        base *= (0.3 + skills.get("blowjob", EffectiveMastery(0)).effective * 1.2)
    
    # 抑制连接不被经验增强（恐惧永远能抑制）
    if base < 0:
        return base
    
    return base
```

#### 3.6.3 传播算法

```python
def propagate_signals(body, ans, emotion, skills, world, iterations=8, dt=0.1):
    """
    每次调用模拟 iterations 个时间步的神经传播
    dt=0.1表示每个时间步0.1秒
    """
    
    # 计算每个节点的认知调制系数
    cognitive_mod = {}
    for node in body.all_nodes:
        base = 1.0
        if emotion.blend["fear"] > 0.3:
            base -= emotion.blend["fear"] * 0.6
        if emotion.blend["hurt"] > 0.3:
            base -= emotion.blend["hurt"] * 0.4
        if emotion.blend["anxiety"] > 0.5:
            base -= emotion.blend["anxiety"] * 0.3
        if emotion.blend["trust"] > 0.8 and emotion.blend["pleasure"] > 0.5:
            base *= 1.2
        if ans.sympathetic > 0.7:
            base *= 1.1
        cognitive_mod[node.name] = max(0.1, base)
    
    for step in range(iterations):
        fired_this_step = {}
        
        # 1. 每个节点处理自身电位
        for node in body.all_nodes:
            fired, output = node.tick(dt)
            if fired:
                mod = cognitive_mod.get(node.name, 1.0)
                fired_this_step[node] = output * mod
        
        # 2. 发放的节点向传出连接传递信号
        for fired_node, output in fired_this_step.items():
            for target_name, base_weight in fired_node.efferent.items():
                target = body.get_node(target_name)
                if target:
                    effective_weight = get_connection_weight(
                        fired_node.name, target_name, skills
                    )
                    target.stimulate(output * effective_weight, dt, 
                                    cognitive_mod.get(target_name, 1.0))
        
        # 3. ANS从身体信号更新
        ans.update_from_firing(fired_this_step, dt)
        
        # 4. 每3步检查一次高潮状态
        if step % 3 == 0:
            orgasm_system.update(body, ans, emotion, current_stimulation, dt)
        
        # 5. 润滑度根据arousal持续分泌
        if ans.arousal_global > 0.4:
            body.vaginal_canal.wetness = min(1.0, 
                body.vaginal_canal.wetness + 0.005 * dt * ans.arousal_global)
            if ans.arousal_global > 0.7:
                body.vaginal_canal.tent_lubrication = min(1.0,
                    body.vaginal_canal.tent_lubrication + 0.003 * dt)
        
        # 6. 充血反应
        for part in [body.nipple_left, body.nipple_right, body.clitoris, 
                     body.inner_labia, body.penis_analog_erection]:
            if part.arousal > 0.5:
                part.congestion = min(1.0, part.congestion + 0.01 * dt)
                if hasattr(part, 'erection'):
                    part.erection = min(1.0, part.erection + 0.02 * dt)
    
    # 7. 传播结束后计算全局arousal
    body.global_arousal = compute_global_arousal(body, ans)
    
    return fired_this_step
```

### 3.7 高潮系统

女性高潮为主，详细模拟多类型、多阶段、多次高潮机制。

#### 3.7.1 高潮阶段

女性高潮分为四个阶段：
1. **兴奋期（Excitement）**：arousal上升，开始充血润滑
2. **平台期（Plateau）**：arousal维持高位，需要持续刺激才能继续累积
3. **高潮期（Orgasm）**：节律性收缩，强烈快感
4. **消退期（Resolution）**：身体逐渐恢复（但女性可以从消退期回到平台期实现多次高潮）

```python
class OrgasmSystem:
    def __init__(self):
        # 平台期阈值
        self.plateau_arousal = 0.75
        self.plateau_duration_required_base = 8  # 秒
        self.plateau_duration_required = self.plateau_duration_required_base
        self.plateau_timer = 0
        
        # 高潮临界点
        self.orgasm_threshold_base = 0.92
        self.orgasm_threshold = self.orgasm_threshold_base
        self.point_of_no_return = False
        
        # 高潮类型及概率
        self.orgasm_type_probabilities = {
            "clitoral": 0.40,      # 阴蒂高潮（常见，强烈但局部）
            "vaginal": 0.15,       # 阴道/G点高潮（更深层，需要经验）
            "blended": 0.35,       # 混合高潮（最强）
            "cervical": 0.05,      # 宫颈高潮（深度，需要完全放松+很深）
            "multiple_chain": 0.05, # 连环高潮的第一波
        }
        
        # 当前阶段
        self.phase = "excitement"  # excitement/plateau/orgasm/resolution
        self.orgasm_duration = 0
        self.orgasm_intensity = 0
        self.orgasm_type = None
        self.contraction_wave = None
        
        # 多次高潮
        self.multi_orgasm_state = {
            "can_have_multiple": True,
            "orgasms_so_far": 0,
            "refractory_type": "partial",
            "sensitivity_between": 0,
            "time_since_last_orgasm": 999,
        }
    
    def update(self, body, ans, emotion, stimulation, dt, skills):
        """每时间步更新高潮状态"""
        
        # === 高潮中 ===
        if self.phase == "orgasm":
            self.orgasm_duration += dt
            self._process_orgasm_contractions(body, ans, dt)
            
            total_duration = self._current_orgasm_duration()
            if self.orgasm_duration > total_duration:
                self._transition_to_resolution(body, emotion)
            return
        
        # === 消退期 ===
        if self.phase == "resolution":
            self.multi_orgasm_state.time_since_last_orgasm += dt
            self._process_resolution(body, ans, emotion, stimulation, dt)
            return
        
        # === 兴奋期→平台期 ===
        if body.global_arousal > self.plateau_arousal:
            if self.phase == "excitement":
                self.phase = "plateau"
                self.plateau_timer = 0
                # 帐篷效应
                body.vaginal_canal.tent_effect = min(1.0, body.vaginal_canal.tent_effect + 0.3)
                body.vaginal_canal.tent_lubrication = min(1.0, body.vaginal_canal.tent_lubrication + 0.3)
                body.cervix.position = "high"
                # 小阴唇颜色变深
                body.inner_labia.color_change = min(1.0, body.inner_labia.color_change + 0.2)
        
        # === 平台期 ===
        if self.phase == "plateau":
            if stimulation.active and stimulation.adequate_intensity:
                self.plateau_timer += dt
                body.global_arousal = min(0.95, body.global_arousal + 0.005 * dt)
                body.inner_labia.color_change = min(1.0, body.inner_labia.color_change + 0.01 * dt)
                ans.skin_flush = min(1.0, ans.skin_flush + 0.02 * dt)
                ans.muscle_tone = min(0.9, ans.muscle_tone + 0.01 * dt)
                
                # "即将高潮"的不可避免感
                if body.global_arousal > 0.88:
                    emotion.blend["inevitability"] = min(1.0, emotion.blend["inevitability"] + 0.1 * dt)
                    emotion.blend["loss_of_control"] = min(1.0, emotion.blend["loss_of_control"] + 0.05 * dt)
            else:
                # 刺激停止→从平台期回落
                self.plateau_timer -= dt * 2
                if self.plateau_timer < 0:
                    self.phase = "excitement"
                    emotion.blend["frustration"] += 0.1
                    body.global_arousal = max(0.6, body.global_arousal - 0.02 * dt)
            
            # 达到临界点
            if (body.global_arousal >= self.orgasm_threshold and 
                self.plateau_timer >= self.plateau_duration_required):
                self._trigger_orgasm(body, ans, emotion, stimulation, skills)
    
    def _trigger_orgasm(self, body, ans, emotion, stimulation, skills):
        self.point_of_no_return = True
        self.phase = "orgasm"
        self.orgasm_duration = 0
        
        # 根据经验调整高潮类型概率
        probs = dict(self.orgasm_type_probabilities)
        vaginal_mastery = max(
            skills.get("missionary", EffectiveMastery(0)).effective,
            skills.get("cowgirl", EffectiveMastery(0)).effective,
        )
        if vaginal_mastery > 0.5:
            probs["vaginal"] += 0.15
            probs["blended"] += 0.10
            probs["clitoral"] -= 0.25
        if skills.get("blowjob", EffectiveMastery(0)).effective > 0.5:
            probs["blended"] += 0.05
        
        # 也取决于当前刺激方式
        if stimulation.primary_target == "clitoris":
            probs["clitoral"] *= 2.0
        if stimulation.deep_penetration:
            probs["vaginal"] *= 1.8
            probs["cervical"] *= 1.5
        if stimulation.both_clit_and_internal:
            probs["blended"] *= 2.5
        
        # 归一化后随机选择
        total = sum(probs.values())
        r = random.random() * total
        cumulative = 0
        for otype, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                self.orgasm_type = otype
                break
        
        self.orgasm_intensity = random.uniform(0.6, 1.0)
        self.multi_orgasm_state.orgasms_so_far += 1
        self.multi_orgasm_state.time_since_last_orgasm = 0
        
        # 强烈高潮生理反应
        ans.heart_rate = 140 + int(random.random() * 30)
        ans.breathing_rate = 35 + int(random.random() * 15)
        ans.breathing_depth = "gasping"
        ans.breathing_rhythm = "irregular"
        ans.skin_flush = 1.0
        ans.muscle_tone = 1.0
        body.skin_global.temperature += 0.8
        body.skin_global.shiver = 0.8
        
        # 肌肉强直
        body.thighs.tension = 1.0
        body.feet.curling = 1.0
        body.back.arch = 0.9
        body.fingers.curled = 1.0
        body.abdomen.muscle_tension = 1.0
        
        # 情绪
        emotion.blend["pleasure"] = 1.0
        emotion.blend["loss_of_control"] = 1.0
        emotion.blend["overwhelm"] = self.orgasm_intensity * 0.8
        emotion.blend["satisfaction"] = 0.3
        emotion.blend["shame"] *= 0.3  # 高潮时羞耻感暂时消失
        if self.multi_orgasm_state.orgasms_so_far == 1:
            emotion.blend["shock"] = 0.3  # 第一次高潮的震惊感
        
        # 收缩波初始化
        self.contraction_wave = {
            "interval": 0.8,
            "count": 0,
            "max_count": int(5 + self.orgasm_intensity * 10),
            "strength": 0.9,
        }
    
    def _process_orgasm_contractions(self, body, ans, dt):
        cw = self.contraction_wave
        cw["interval"] -= dt
        
        if cw["interval"] <= 0 and cw["count"] < cw["max_count"]:
            cw["count"] += 1
            
            # 阴道节律性收缩
            body.vaginal_canal.contraction = cw["strength"]
            
            # 前几次收缩更强，涉及范围更大
            if cw["count"] <= 3:
                body.outer_labia.contraction = cw["strength"] * 0.8
                body.anus.contraction = cw["strength"] * 0.6
            if cw["count"] <= 2:
                body.abdomen.muscle_tension = 1.0
                body.thighs.tremor = 1.0
            
            # 宫颈dipping
            if self.orgasm_type in ["vaginal", "blended", "cervical"]:
                body.cervix.dip = True
            
            # 射液/潮吹（概率事件）
            ejac_prob = 0.05 + self.orgasm_intensity * 0.15
            if self.orgasm_type in ["vaginal", "blended"]:
                ejac_prob += 0.1
            if random.random() < ejac_prob:
                body.vaginal_canal.ejaculation_volume = random.uniform(0.3, 1.5)
            
            # 声音不自主（即使在忍，高潮时也可能忍不住）
            if cw["count"] <= 3:
                ans.voice_breathiness = 1.0
                if random.random() < self.orgasm_intensity * 0.6:
                    emotion.suppressing_sounds = False  # 忍不住叫出来
            
            # 间隔变长、强度递减
            cw["interval"] = 0.8 + cw["count"] * 0.12
            cw["strength"] *= 0.85
        
        elif cw["count"] >= cw["max_count"]:
            body.vaginal_canal.contraction *= 0.85
            body.outer_labia.contraction *= 0.8
    
    def _transition_to_resolution(self, body, emotion):
        self.phase = "resolution"
        self.point_of_no_return = False
        self.plateau_timer = 0
        
        if (self.multi_orgasm_state.can_have_multiple and 
            self.multi_orgasm_state.orgasms_so_far < 3 and
            self.orgasm_intensity > 0.7):
            # 可以多重高潮：部分不应期，arousal不降到很低
            self.multi_orgasm_state.sensitivity_between = 1.2
            body.clitoris.oversensitive_post_orgasm = True
            body.global_arousal = 0.65
            self.orgasm_threshold = 0.82  # 第二次阈值更低
            self.plateau_duration_required = 3  # 平台期更短
            emotion.blend["satisfaction"] = 0.6
        else:
            # 完全消退
            self.multi_orgasm_state.sensitivity_between = 0
            body.clitoris.oversensitive_post_orgasm = True
            body.clitoris.pain_from_overstimulation = 0.3
            body.global_arousal = 0.3
            self.orgasm_threshold = self.orgasm_threshold_base
            self.plateau_duration_required = self.plateau_duration_required_base
            emotion.blend["satisfaction"] = 0.9
            emotion.blend["sleepy"] = 0.3 + self.multi_orgasm_state.orgasms_so_far * 0.2
    
    def _process_resolution(self, body, ans, emotion, stimulation, dt):
        # 缓慢回落
        body.global_arousal = max(0.1, body.global_arousal - 0.01 * dt)
        ans.heart_rate = max(75, ans.heart_rate - 0.8 * dt)
        ans.breathing_rate = max(14, ans.breathing_rate - 0.4 * dt)
        ans.muscle_tone = max(0.2, ans.muscle_tone - 0.02 * dt)
        ans.skin_flush = max(0, ans.skin_flush - 0.005 * dt)
        body.skin_global.temperature = max(36.5, body.skin_global.temperature - 0.01 * dt)
        body.back.arch *= 0.9
        body.feet.curling *= 0.8
        body.thighs.tremor *= 0.85
        
        # 阴蒂过度敏感消退（30秒到2分钟）
        if self.multi_orgasm_state.time_since_last_orgasm > 90:
            body.clitoris.oversensitive_post_orgasm = False
            body.clitoris.pain_from_overstimulation = max(0, body.clitoris.pain_from_overstimulation - 0.01 * dt)
        
        # 满足感/困倦
        emotion.blend["satisfaction"] = min(1.0, emotion.blend["satisfaction"] + 0.005 * dt)
        if self.multi_orgasm_state.orgasms_so_far >= 2:
            emotion.blend["sleepy"] = min(1.0, emotion.blend["sleepy"] + 0.005 * dt)
        
        # 如果继续足够刺激，可以回到平台期（多重高潮）
        clit_too_sensitive = body.clitoris.pain_from_overstimulation > 0.5
        if (stimulation.active and stimulation.adequate_intensity and 
            body.global_arousal > 0.5 and not clit_too_sensitive and
            self.multi_orgasm_state.orgasms_so_far < 3):
            self.phase = "plateau"
            self.plateau_timer = self.plateau_duration_required * 0.5
    
    def _current_orgasm_duration(self):
        # 高潮持续时间：强烈的更长，多次后的更短
        base = 8 + self.orgasm_intensity * 12  # 10-20秒
        if self.multi_orgasm_state.orgasms_so_far > 1:
            base *= 0.8
        return base
```

---

## 4. 记忆系统

记忆不是一个列表，分三种类型，存储方式和遗忘规律不同。

### 4.1 三类记忆

#### 4.1.1 程序记忆（Procedural Memory）
即技能系统（Layer 0d），存储"身体记住的"怎么做。
- 不刻意回忆，身体自动执行
- 遗忘慢（半衰期7-90天，熟练度越高遗忘越慢）
- 表现为mastery值，不需要具体事件记忆

#### 4.1.2 情感记忆（Emotional Memory）

```python
emotional_memory = {
    "trust_base": 0.7,              # 基础信任度（缓慢变化）
    "security_base": 0.6,           # 基础安全感
    "intimacy_accumulated": 0.5,    # 累计亲密感
    "comfort_level_together": 0.6,  # 在一起的舒适感
    "hurt_episodes": [              # 受伤事件
        {
            "event": "mentioned_other_girl_first_time",
            "intensity": 0.8,
            "time": "2026-07-15",
            "healed": 0.3,          # 愈合程度
            "trigger_words": ["别的女生", "xxx"],
        }
    ],
    "special_moments": [            # 特别美好时刻
        {"event": "first_kiss", "intensity": 0.9, "time": "2026-07-01"},
        {"event": "first_i_love_you", "intensity": 0.95, "time": "2026-07-20"},
    ],
    "trauma_response_sensitivity": {  # 创伤后对类似事件敏感度升高
        "mentioned_other_girl": 0.5,  # 下次提别的女生反应更剧烈
    },
}
```

- 情感记忆消退极慢
- 被伤害过一次即使"原谅了"，下次类似情况敏感度会升高
- 美好记忆累计提升安全感和信任

#### 4.1.3 情景记忆（Episodic Memory）

```python
episodic_memory_entry = {
    "type": "first_time | hurt | romantic | intimate | routine | interrupted | funny",
    "summary": "str",              # 一句话摘要（自然语言）
    "emotional_valence": float,    # -1到1，负/正面
    "salience": float,             # 0-1，重要程度
    "time": datetime,
    "location": str,
    "action_type": str,
    "body_parts": list,
    "trigger_words": list,         # 能触发回忆的关键词
    "recalled_count": int,         # 回忆过几次（回忆巩固记忆）
    "consolidation_status": "working | buffer | long_term",
}
```

### 4.2 记忆三层存储+巩固流程

```
瞬时记忆(Sensory Buffer) → 短期工作记忆 → 长期记忆
  (每轮所有感知)          (最近10-20轮)    (经显著性筛选)
   存1-2轮消失             保留约24小时      永久/极慢遗忘
```

#### 显著性评分（每轮结束后自动计算）

```python
def calc_salience(event, state, world):
    score = 0.0
    
    # 情绪强度
    score += state.emotion.intensity() * 0.3
    
    # 第一次做某事
    if event.is_first_time:
        score += 0.5
    
    # 情绪冲突（又怕又爽/矛盾状态）
    score += state.emotion.conflict_level * 0.3
    
    # 禁忌/危险（差点被发现/公共场合）
    score += world.danger_level * 0.2
    
    # 涉及高潮
    if event.involved_orgasm:
        score += 0.4
        if event.was_first_orgasm:
            score += 0.3
    
    # 被中断/惊吓
    if event.was_interrupted:
        score += 0.2
    
    # 特殊对话（告白/伤人的话/说爱你）
    if event.has_significant_speech:
        score += 0.3
    if event.was_love_confession:
        score += 0.5
    if event.was_jealousy_trigger:
        score += 0.4
    
    # 新场景/新姿势
    if event.was_new_location or event.was_new_position:
        score += 0.2
    
    # 随机波动（不是所有重要事都记住，不是所有小事都忘）
    score += random.gauss(0, 0.05)
    
    return max(0.0, min(1.0, score))
```

#### 记忆巩固流程

```python
consolidation_buffer = []  # 待巩固区
long_term_episodic = []    # 长期记忆
working_memory = []        # 短期工作记忆

def consolidate_memory(event, state, world):
    salience = calc_salience(event, state, world)
    mem_entry = create_memory_entry(event, salience)
    
    if salience > 0.7:
        # 直接进入长期记忆
        long_term_episodic.append(mem_entry)
    elif salience > 0.4:
        # 进入待巩固区
        mem_entry["consolidation_deadline"] = now + timedelta(days=int(30 * salience))
        mem_entry["recall_count"] = 0
        consolidation_buffer.append(mem_entry)
    elif salience > 0.2:
        # 短期工作记忆
        mem_entry["ttl"] = timedelta(hours=24)
        working_memory.append(mem_entry)
    # salience<0.2的不记忆

def memory_maintenance():
    """定期调用，记忆维护"""
    # 巩固区：过期未被回忆→遗忘；被回忆≥2次→固化为长期
    still_in_buffer = []
    for mem in consolidation_buffer:
        if now > mem["consolidation_deadline"] and mem["recall_count"] == 0:
            continue  # 遗忘
        elif mem["recall_count"] >= 2:
            long_term_episodic.append(mem)
        else:
            still_in_buffer.append(mem)
    consolidation_buffer = still_in_buffer
    
    # 短期记忆过期清除
    working_memory = [m for m in working_memory if now < m["time"] + m["ttl"]]
    
    # 长期记忆也有微弱权重衰减（但salience>0.8的核心记忆不衰减）
    for mem in long_term_episodic:
        if mem["salience"] < 0.8:
            days_old = (now - mem["time"]).days
            mem["retrieval_weight"] = mem["salience"] * math.exp(-days_old / 365)
        else:
            mem["retrieval_weight"] = mem["salience"]
```

### 4.3 相关记忆检索

每轮只检索注入和当前场景/话题/动作相关的记忆，限制在top 3条：

```python
def retrieve_relevant_memories(current_state, parsed, world, limit=3):
    candidates = []
    all_mems = long_term_episodic + consolidation_buffer + working_memory
    
    for mem in all_mems:
        relevance = 0.0
        
        # 地点相同
        if mem.get("location") == world.current_location_type:
            relevance += 0.2
        # 动作类型相同
        if parsed.sub_actions and mem.get("action_type") == parsed.sub_actions[0].action_type:
            relevance += 0.3
        # 涉及相同身体部位
        if parsed.sub_actions:
            parts_targeted = set()
            for sa in parsed.sub_actions:
                parts_targeted.update(sa.targets)
            mem_parts = set(mem.get("body_parts", []))
            if parts_targeted & mem_parts:
                relevance += 0.2
        # 触发词匹配
        if parsed.speech_content:
            for trigger in mem.get("trigger_words", []):
                if trigger in parsed.speech_content:
                    relevance += 0.5  # 触发词权重高
        # 时间近因
        days_ago = (now - mem["time"]).days
        relevance *= math.exp(-days_ago / 60)  # 近因效应
        # 显著性加权
        relevance *= (0.4 + 0.6 * mem["salience"])
        # 情绪一致性（当前情绪容易想起同情绪记忆）
        curr_valence = current_state.emotion.primary_valence()
        if mem["emotional_valence"] * curr_valence > 0:
            relevance *= 1.2
        else:
            relevance *= 0.7
        
        retrieval_weight = mem.get("retrieval_weight", mem["salience"])
        final_score = relevance * retrieval_weight
        
        if final_score > 0.25:
            candidates.append((mem, final_score))
    
    candidates.sort(key=lambda x: -x[1])
    return [m for m, s in candidates[:limit]]
```

注入prompt时转化为自然语言回忆提示，不直接塞JSON：

```
[你隐约记得：之前在客厅里哥哥也是这样摸你，差点被妈妈发现，你吓得浑身僵硬]
[你想起：第一次给哥哥做的时候，也是在浴室里，你紧张得牙齿都碰到了]
[上次他提到那个女生的时候，你心里揪了一下...此刻那种感觉又浮上来]
```

---

## 5. 世界与事件系统

### 5.1 世界状态

```python
world_state = {
    "time": {
        "hour": 22, "minute": 30,
        "weekday": "Friday",
        "season": "summer",
        "is_raining": False,
        "is_morning": False,
    },
    "location": {
        "type": "bedroom",     # bedroom/bathroom/living_room/kitchen/outside/school/etc
        "privacy": 0.8,        # 0=公共场合 1=完全私密
        "lighting": "dim",     # bright/dim/dark/candlelight
        "noise_level": 0.2,    # 环境噪音
        "temperature": 27,
        "bed_size": "double",
        "door_locked": False,
        "curtains_drawn": True,
    },
    "people_present": {
        "user": {"location": "bed", "clothing": "pajamas", "activity": None},
        "sister": {"location": "bed", "clothing": "underwear_only", "activity": None},
        "mom": {"location": "living_room", "activity": "watching_tv", 
                "expected_leave": "23:30", "can_hear_bedroom": True},
        "dad": {"location": "home_office", "activity": "working", 
               "expected_leave": "24:00"},
    },
    "nearby_sounds": ["mom_TV_faint", "AC_hum", "clock_ticking"],
    "objects_in_reach": ["lamp_switch", "tissue_box", "phone", "blanket", "pillow"],
    "condoms_available": False,  # 妹妹是否知道有避孕套
    "danger_level": 0.2,         # 被发现风险
}
```

### 5.2 时间系统

三种时间流逝模式：

#### 5.2.1 动作持续时间

每个动作有基础耗时，受状态影响：

```python
action_durations = {
    "drink_water": {
        "base": 1.5,
        "modifiers": {
            "thirsty_high": 0.5,
            "relaxed": 2.0,
            "hurried": 0.3,
        }
    },
    "kiss": {
        "base": 5,
        "modifiers": {
            "arousal_high": 0.7,
            "romantic_mood": 1.5,
            "first_kiss": 2.0,
        }
    },
    "touch_nipple": {
        "base": 8,
        "modifiers": {
            "arousal_low": 1.5,
            "arousal_high": 0.5,
            "shy": 1.3,
        }
    },
    # ...
}

def calc_action_duration(action_name, state, world):
    if action_name not in action_durations:
        return 3.0
    cfg = action_durations[action_name]
    dur = cfg["base"]
    for mod_name, factor in cfg.get("modifiers", {}).items():
        if check_condition(mod_name, state, world):
            dur *= factor
    dur *= random.uniform(0.85, 1.15)  # 随机扰动
    return dur
```

#### 5.2.2 回合间时间跳跃

```python
def estimate_time_jump(last_action, current_action, last_state, current_input):
    """估算两个回合之间的时间跳跃"""
    # 显式时间词
    if "第二天" in current_input or "早上" in current_input:
        return 8 * 3600  # 8小时
    if "洗完澡" in current_input:
        return 600  # 10分钟
    
    # 场景变化推断
    if was_in_bedroom(last_action) and is_in_bathroom(current_action):
        return 120  # 2分钟走过去+开始
    
    # 亲密互动中的连续回合
    if last_state.ans.arousal_global > 0.3 and is_intimate_action(current_action):
        return 2  # 2秒间隙
    
    # 默认
    return 5  # 5秒
```

跳跃后的状态衰减：
- <60秒：身体状态小幅衰减（arousal×0.95）
- 1-30分钟：arousal基本归零，潮红/余韵部分保留
- >30分钟/跨场景：身体状态重置到基线，经验/情感记忆保留

#### 5.2.3 事件中断（Interrupt）

情绪冲击/外部事件可以瞬间改变状态：

```python
interrupt_triggers = {
    "mention_other_girl": {
        "type": "emotional_shock",
        "effect": {
            "arousal_global": -0.9,        # 性唤起瞬间暴跌
            "vaginal.contraction": "spasm_close",
            "lubrication": "stop",
            "emotion": ["hurt", "jealousy"],
            "skin": "cold_flash",
            "muscle": "freeze",
            "attention_switch": True,
            "cognitive_gate": 0.1,
        },
        "recovery": "requires_apology",    # 需要哄才能恢复
    },
    "parents_come_home": {
        "type": "panic_interrupt",
        "effect": {
            "arousal_global": -0.7,
            "heart_rate": "+50bpm_spike",
            "muscle": "freeze_then_flee",
            "attention": ["door", "footsteps"],
            "emotion": ["terror"],
            "suppress_sounds": True,
            "lubrication_note": "body_continues_but_ignored",  # 身体可能还湿但注意力完全转移
        },
    },
    "door_knock": {
        "type": "startle",
        "effect": {
            "arousal_global": -0.3,
            "heart_rate": "+30bpm",
            "freeze": True,
            "attention": ["door"],
            "emotion": ["panic", "fear"],
        }
    },
    "phone_rings": {
        "type": "attention_break",
        "effect": {
            "arousal_decay_accelerated": 3.0,
            "attention_split": True,
        }
    },
}
```

关键：中断不是所有状态归零。例如父母回来时身体可能还在湿润（生理不随意志瞬间停止），但注意力完全转移，肌肉从松弛变僵硬——这种"身体还停留在刚才但脑子已经炸了"的矛盾非常真实。

### 5.3 事件引擎

三类事件：

#### 5.3.1 固定/日程事件

```python
scheduled_events = [
    {"time": "Fri 18:00", "event": "family_dinner", "duration": 60, 
     "attendees": ["mom", "dad", "sister", "user"], "effect": {"location_change": True}},
    {"time": "Fri 22:30", "event": "mom_dad_bedtime_approx", "probability": 0.6},
    {"time": "weekday 7:00", "event": "alarm_clock", "must_attend": True},
    {"time": "exam_week", "event": "sister_stressed", 
     "effect": {"mood_modifier": -0.3, "libido_modifier": -0.2}},
    {"time": "summer_holiday", "event": "more_free_time",
     "effect": {"available_time": "+4h", "parents_at_work_days": True}},
]
```

#### 5.3.2 随机事件

```python
random_events = [
    {
        "trigger": "intimate_duration > 10min AND location == bedroom AND parents_home AND NOT door_locked",
        "probability_per_minute": 0.015,
        "event": "mom_knocks_on_door",
        "interrupt_type": "high",
    },
    {
        "trigger": "arousal > 0.7 AND location == shower",
        "probability_per_minute": 0.01,
        "event": "someone_flushes_toilet_water_temp_changes",
        "interrupt_type": "low",
    },
    {
        "trigger": "making_loud_sounds AND not_alone",
        "probability_per_minute": 0.03,
        "event": "heard_by_someone",
        "interrupt_type": "medium",
    },
    {
        "trigger": "walking_home_at_night AND alone",
        "probability": 0.08,
        "event": "stranger_walks_by_close",
    },
]
```

#### 5.3.3 条件触发事件

```python
conditional_events = [
    {"condition": "orgasm_count_today >= 2 AND not_prompted_to_continue", 
     "event": "sister_sleepy_and_content"},
    {"condition": "trust < 0.3 AND intimate_advances AND first_time_sexual",
     "event": "sister_freezes_up"},
    {"condition": "clothing.layer3 == none AND first_time_naked_together",
     "event": "sister_tries_to_cover_self"},
    {"condition": "mentioned_ex_girlfriend AND arousal > 0.5",
     "event": "emotional_shut_down_possible"},
]
```

### 5.4 衣物系统

分层结构，有多种中间状态（非简单"穿了/脱了"）。

```python
clothing_system = {
    "layers": [
        {
            "name": "outerwear",
            "items": {
                "jacket": {"worn": False, "open": False, "removed": False},
                "sweater": {"worn": False, "open": False, "removed": False},
                "cardigan": {"worn": False, "open": False, "removed": False},
                "school_uniform_blazer": {"worn": False, "open": False, "removed": False},
            },
        },
        {
            "name": "top",
            "items": {
                "tshirt": {"worn": True, "pulled_up": 0.0, "removed": False,
                          "access_through": {"neck": False, "bottom_hem": False, "sleeve": False}},
                "blouse": {"worn": False, "buttons_undone": 0, "total_buttons": 5, "removed": False},
                "tank_top": {"worn": False, "pulled_up": 0.0, "removed": False},
                "dress": {"worn": False, "pulled_up": 0.0, "zipper_down": 0, "removed": False},
                "pajama_top": {"worn": False, "buttons_undone": 0, "removed": False},
                "school_shirt": {"worn": False, "buttons_undone": 0, "tucked_in": True, "removed": False},
            },
        },
        {
            "name": "bottom_outer",
            "items": {
                "jeans": {"worn": False, "zipper_down": False, "button_undone": False, 
                         "pulled_down": 0.0, "removed": False},
                "shorts": {"worn": True, "pulled_down": 0.0, "removed": False},
                "skirt": {"worn": False, "pulled_up": 0.0, "zipper_down": False, "removed": False},
                "sweatpants": {"worn": False, "pulled_down": 0.0, "removed": False},
                "pajama_bottom": {"worn": False, "pulled_down": 0.0, "removed": False},
                "school_skirt": {"worn": False, "pulled_up": 0.0, "removed": False},
            },
        },
        {
            "name": "bra",
            "items": {
                "bra": {"worn": True, "clasp_undone": False, 
                       "strap_down_left": False, "strap_down_right": False,
                       "lifted_up": False, "removed": False,
                       "cup_access": False},
                "sports_bra": {"worn": False, "pulled_up": 0.0, "removed": False},
            },
            "note": "bra即使穿着，也可从上方/下方伸手摸到乳房，但不能直接碰乳头（隔着布料）",
        },
        {
            "name": "panties",
            "items": {
                "cotton_panties": {"worn": True, "pulled_side": None, "pulled_down": 0.0,
                                  "wet_spot": 0.0, "fabric_bunched": False,
                                  "aside": False, "removed": False, "around_leg": False},
                "thong": {"worn": False, "pulled_side": None, "removed": False},
                "boyshorts": {"worn": False, "pulled_down": 0.0, "removed": False},
                "no_panties": {"worn": False},
            },
            "note": "内裤即使没脱，也可从侧边/裤腰伸进去摸",
        },
        {
            "name": "accessories",
            "items": {
                "socks": {"worn": True, "one_off": False, "both_off": False},
                "stockings": {"worn": False, "garter_belt": False, "torn": False},
                "hair_tie": {"worn": True},
                "glasses": {"worn": False, "removed": False, "fogged": False},
                "hair_clip": {"worn": False},
            },
        },
    ],
    "coverings": {
        "blanket": {"covering_level": 0.0},  # 0=不盖 1=盖到脖子
        "towel": {"wrapped": False, "coverage": 0.0},
        "pillow": {"hiding_behind": False},
    },
}
```

#### 暴露度和可接触度计算

```python
def calc_accessibility(clothing, body_part):
    """计算某个部位当前的可接触程度"""
    access = 0.0  # 0=完全遮挡 1=赤裸直接接触
    
    # 衣物遮挡计算（根据层次状态）
    layers_covering = get_covering_layers(clothing, body_part)
    direct_skin = True
    fabric_layers = 0
    
    for layer in layers_covering:
        if layer.partially_open:
            fabric_layers += 0.3  # 半开/拉开
        elif layer.worn:
            fabric_layers += 1.0
            direct_skin = False
    
    # 覆盖物
    if clothing.coverings.blanket.covering_level > 0.5:
        fabric_layers += 0.5
    
    if fabric_layers == 0:
        access = 1.0  # 直接接触皮肤
    elif fabric_layers < 1:
        access = 0.7  # 手伸进衣服里直接接触皮肤（比如手从下摆伸进去摸胸）
    elif fabric_layers < 2:
        access = 0.4  # 隔着一层布料
    else:
        access = 0.15  # 隔多层
    
    return access, direct_skin
```

### 5.5 主观认知层（MindState）

和客观WorldState分开——她不知道的信息不会影响她的反应。

```python
class MindState:
    def __init__(self):
        # 对环境的认知
        self.known_people_locations = {}      # 她以为谁在哪（可能错）
        self.believed_door_locked = False     # 她以为门锁没锁
        self.heard_recent_sounds = []         # 她听到的声音
        self.noticed_objects = []             # 她注意到的物体
        
        # 对哥哥的认知
        self.believed_partner_mood = "normal"
        self.believes_he_loves_me = 0.6
        self.knows_about = {
            "condoms_location": False,
            "phone_nearby": True,
            "parents_sleeping": False,
        }
        
        # 对自己身体的感知
        self.noticed_sensations = []          # 她有意识注意到的身体感受
        self.body_awareness_level = 0.4
        self.knows_she_is_wet = False
        self.knows_he_is_hard = False
        
        # 她不知道的事情（不出现在反应中）
        self.unknown = [
            "condoms_in_nightstand",   # 她不知道避孕套在抽屉里
            "mom_watching_outside",     # 她不知道妈妈在门外（除非听到声音）
        ]
    
    def update_from_perception(self, world, body, parsed):
        """每轮根据感官输入更新主观认知"""
        
        # 听到声音→更新对人物位置的认知
        for sound in world.nearby_sounds:
            if "footsteps" in sound and "door" in sound:
                self.known_people_locations["someone_at_door"] = True
                self.heard_recent_sounds.append(sound)
        
        # 触觉感知（摸到什么）
        if parsed.touch_noticed:
            if body.vaginal_canal.wetness > 0.4 and self.body_awareness_level > 0.3:
                self.knows_she_is_wet = True
            if partner_state.penis.erection > 0.7 and body.partner_touch_perceived:
                self.knows_he_is_hard = True
        
        # 视觉（看到什么）
        # 根据光线、距离、是否闭眼等因素
        
        # 注意力过滤：不是所有身体信号都被意识到
        self.noticed_sensations = []
        for node in body.all_nodes:
            if node.arousal > 0.6 - self.body_awareness_level * 0.3:
                self.noticed_sensations.append(node.name)
        # 注意力集中在某个部位时，其他部位感受被忽略
        if len(self.emotion.attention_focus) > 0:
            focus_threshold = 0.4
            self.noticed_sensations = [
                n for n in self.noticed_sensations
                if n in self.emotion.attention_focus or body.get_node(n).arousal > focus_threshold
            ]
```

**核心原则**：Layer2 prompt只包含MindState中的信息，不直接引用WorldState中角色不知道的内容。

---

## 6. 输入解析层

### 6.1 感知解析器（Parser）

Parser负责把自然语言输入（动作描写+对话）转化为结构化的刺激数据。

初期用**规则Parser**（关键词+正则），预留接口未来可替换为**小模型Parser**。两者接口一致，无缝替换。

```python
class ParsedInput:
    sub_actions: list[SubAction]       # 子动作序列（有序）
    speech_content: Optional[str]      # 纯对话内容
    speech_emotion: Optional[str]      # 说话语气
    ambiguity: Optional[Ambiguity]     # 歧义信息（如有）
    location_change: Optional[str]
    clothing_change: list[ClothingAction]
    objects_referenced: list
    implicit_signals: dict
    action_continues: bool             # 上一动作是否在持续

class SubAction:
    action_type: str       # touch/kiss/squeeze/rub/penetrate/thrust/lick/suck/etc
    targets: list[str]     # 目标身体部位
    intensity: float       # 0-1
    duration_estimate: float  # 秒
    handedness: Optional[str]  # left/right/both/mouth/genital
    position_change: Optional[str]
    rough: bool
    gentle: bool
    teasing: bool
    confidence: float      # 解析置信度 0-1

class Ambiguity:
    confidence: float
    possible_interpretations: list
    ambiguous_target: bool
    ambiguous_action: bool
```

规则Parser的关键词映射表覆盖你数据集中常见的动作模式，预计50-100条规则覆盖95%以上输入。

### 6.2 歧义处理机制

当parser置信度低于阈值时，不强行猜测，而是计算三种应对方式的概率交给模型：

```python
def handle_ambiguity(state, world, ambiguity):
    probs = {
        "ask_clarify": 0.3,    # 反问"嗯？什么？""哪里...？"
        "proactive_guess": 0.3, # 主动猜测/试探（往你身上蹭，看你想干嘛）
        "wait": 0.4,           # 等着不动，看你继续
    }
    
    # 反问概率修正
    probs["ask_clarify"] *= (1 - state.relationship.trust_base * 0.5)
    probs["ask_clarify"] *= (1 - state.emotion.blend["shame"] * 0.5)
    probs["ask_clarify"] *= (0.5 + world.danger_level)
    
    # 主动猜测概率修正
    probs["proactive_guess"] *= (0.2 + state.ans.arousal_global * 1.5)
    probs["proactive_guess"] *= (0.3 + skills["initiating"].effective_mastery * 1.5)
    probs["proactive_guess"] *= (1 - state.emotion.blend["anxiety"])
    
    # 等待概率修正
    probs["wait"] *= (0.5 + state.emotion.blend["shame"])
    probs["wait"] *= (1 - state.ans.arousal_global * 0.5)
    
    # 亲密语境下高arousal更可能主动猜/迎合
    if state.ans.arousal_global > 0.6 and context_is_intimate(state):
        probs["proactive_guess"] *= 2.0
    
    # 日常语境（arousal低）更容易问清楚
    if state.ans.arousal_global < 0.2:
        probs["ask_clarify"] *= 1.5
        probs["proactive_guess"] *= 0.2
    
    # 加随机扰动
    for k in probs:
        probs[k] *= random.uniform(0.85, 1.15)
    
    total = sum(probs.values())
    for k in probs:
        probs[k] /= total
    
    choice = weighted_random_choice(probs)
    return choice
```

---

## 7. 节奏与行为控制

### 7.1 回合节奏控制器（Beat Controller）

不是每轮都有新刺激或新动作。控制器决定本轮的"节拍类型"：

- `react_continue`：当前刺激还在持续（手还在揉/还在里面），继续反应
- `prompt`：暗示/邀请哥哥继续
- `initiate`：妹妹主动做些什么
- `wait`：停顿，维持当前状态，看哥哥反应

```python
def decide_beat(state, parsed, world):
    # 1. 如果用户动作在持续（没有停下）
    if parsed.action_continues and state.last_stimulation_active:
        return "react_continue"
    
    # 2. 基础概率
    factors = {}
    
    # 根据arousal设置基础分布
    a = state.ans.arousal_global
    if a < 0.2:
        factors = {"wait": 0.6, "prompt": 0.3, "initiate": 0.1}
    elif a < 0.4:
        factors = {"wait": 0.4, "prompt": 0.4, "initiate": 0.2}
    elif a < 0.6:
        factors = {"wait": 0.25, "prompt": 0.45, "initiate": 0.3}
    elif a < 0.8:
        factors = {"wait": 0.15, "prompt": 0.40, "initiate": 0.45}
    else:
        factors = {"wait": 0.05, "prompt": 0.25, "initiate": 0.70}
    
    # 3. 性格修正
    factors["initiate"] *= state.personality.extraversion
    factors["initiate"] *= (0.3 + 0.7 * skills["initiating"].effective_mastery)
    factors["initiate"] *= state.relationship.trust_base
    factors["initiate"] *= (1 - state.emotion.blend["shame"] * 0.7)
    factors["wait"] *= (1 - state.personality.assertiveness * 0.5)
    
    # 4. 情绪修正
    if state.emotion.blend["hurt"] > 0.5:
        factors["initiate"] *= 0.1
        factors["wait"] *= 2.5
        factors["prompt"] *= 0.5
    if state.emotion.blend["anxiety"] > 0.5:
        factors["initiate"] *= 0.2
        factors["wait"] *= 2.0
    if state.emotion.blend["fear"] > 0.4:
        factors["initiate"] *= 0.1
        factors["wait"] *= 1.5
        factors["prompt"] *= 0.3
    if state.emotion.blend["sleepy"] > 0.7:
        factors["wait"] *= 2.0
        factors["initiate"] *= 0.3
    if world.privacy_level < 0.5:
        factors["initiate"] *= 0.1
        factors["wait"] *= 1.5
        factors["prompt"] *= 0.5  # 不敢出声提示
    
    # 5. 高潮后
    if state.orgasm.phase == "resolution":
        factors["wait"] *= 2.0
        factors["prompt"] *= 0.6
        factors["initiate"] *= 0.5
        if state.orgasm.multi_orgasm_state.orgasms_so_far >= 2:
            factors["wait"] *= 2.0
    
    # 6. 新鲜感
    factors["initiate"] *= (0.5 + novelty_system.global_novelty * 0.8)
    
    # 7. 用户刚停下→更可能提示继续
    if parsed.user_paused_action:
        factors["prompt"] *= 2.0
    
    # 8. 少量随机扰动
    for k in factors:
        factors[k] *= random.uniform(0.9, 1.1)
    
    return weighted_random_choice(factors)
```

不同beat的具体表现由主模型根据状态描写：
- **wait**：维持姿势的微小调整（蹭一下、呼吸调整、抓紧床单）、眼神/表情描写、余韵
- **prompt**：蹭你/拉你的手/小声哼唧/台词"怎么...停了..."、眼神
- **initiate**：程度从低到高（亲一下→手往下摸→骑上去→主动含/坐下去）

---

## 8. 叙事生成层

### 8.1 Layer 1: 感受生成器

初期用**多维度概率采样**方案（不需要额外小模型），后期可替换为1B-3B参数小模型专门生成。

核心思路：**把感受描述分解为多个独立维度，每个维度根据当前状态从概率分布中采样，然后组合**。这不是简单"随机选一句"，而是维度独立变化。

```python
# 每个显著变化的部位，生成一个感受描述
# 描述由多个维度组合而成

sensation_dimensions = {
    "perceptual_focus": {  # 这次注意力主要感受什么
        "physical_texture": ["硬邦邦", "发烫", "粗糙的手掌", "软乎乎", "湿润", "滚烫"],
        "movement": ["揉着", "捏着", "搓着", "打转", "按压", "划过", "挑弄", "蹭着"],
        "internal_feeling": ["一阵酥麻", "发麻", "发胀", "发酸", "刺痒", "触电般", "火烧火燎"],
        "pain_mixed": ["微微发疼", "胀痛", "有点疼但又舒服", "酸胀"],
    },
    "intensity_modifier": {
        "very_low": ["微微", "隐约", "有点"],
        "low": ["有些", "不禁", "不自觉地"],
        "medium": ["忍不住", "控制不住地", "不由得"],
        "high": ["猛地", "一下子", "整个人都"],
        "peak": ["浑身一颤", "脑子一片空白", "整个人都要化了"],
    },
    "spread_direction": {
        "local": [],  # 不蔓延
        "chest": ["胸口发闷", "心口一紧"],
        "lower_abdomen": ["酥麻感窜到下腹", "下腹一紧", "肚子里发烫"],
        "spine": ["顺着脊背窜上来", "后背一阵发麻"],
        "whole_body": ["蔓延到全身", "整个人都软了", "四肢百骸都透着酥麻"],
        "thighs": ["窜到大腿根", "腿根发软"],
    },
    "sound_response": {
        "none": [],
        "gasp": ["倒吸一口气", "呼吸一滞"],
        "moan_small": ["嘴里漏出一声闷哼", "小声\"嗯\"了一下"],
        "moan_loud": ["忍不住呻吟出声", "声音发颤地\"嗯啊\""],
        "suppressed": ["咬着唇忍住声音", "把声音咽回去"],
        "sob": ["带着哭腔哼", "声音里带了点鼻音"],
    },
    "involuntary_movement": {
        "none": [],
        "arch": ["腰不受控制地弓起来"],
        "tremble": ["身子抖了一下", "控制不住地发颤"],
        "clench": ["手指紧紧抓住床单", "腿一下子夹紧"],
        "flinch": ["身子缩了一下", "往旁边躲了躲"],
        "press_into": ["不自觉往你手上贴", "腰往下沉迎合"],
        "weak": ["膝盖一软", "力气像被抽走了"],
    },
    "emotional_color": {
        "shy_pleasure": ["又羞又舒服", "明明想忍住却...", "不好意思但...", "羞耻又快乐"],
        "confused": ["不知道该怎么办", "脑子乱乱的", "身体怎么不听话"],
        "want_more": ["还想要更多", "不想你停", "不够..."],
        "overwhelmed": ["太...太舒服了", "受不了了", "慢一点..."],
        "scared_but_good": ["有点怕但是...", "紧张又期待", "心里慌慌的但身体..."],
        "hurt_mixed": ["有点疼...但是...", "弄疼我了...但是...", "轻一点..."],
        "trusting": ["交给哥哥", "哥哥...", "在哥哥手里..."],
    },
}

def generate_sensations(state, sensations_data, beat):
    """生成结构化感受片段"""
    results = []
    
    # 选delta最大的top3显著部位
    significant_changes = get_top_changes(state, top_n=3)
    
    # 加入一个无意识反应（不受意志控制）
    involuntary = generate_involuntary(state)
    
    for i, (node, delta) in enumerate(significant_changes):
        # 根据状态确定各维度的权重
        intensity_level = get_intensity_level(delta, state.ans.arousal_global)
        available_spreads = get_possible_spreads(node, delta, state)
        available_sounds = get_possible_sounds(state, world)
        available_movements = get_possible_movements(node, state)
        available_emotions = get_emotional_colors(state)
        
        # 各维度独立采样
        focus = weighted_choice(sensation_dimensions["perceptual_focus"], weights_by_node(node))
        intensity = weighted_choice(sensation_dimensions["intensity_modifier"][intensity_level])
        spread = weighted_choice(available_spreads) if i == 0 else None  # 主要感受才有蔓延
        sound = weighted_choice(available_sounds) if i == 0 else None     # 主要感受才有声音
        movement = weighted_choice(available_movements) if random.random() < 0.6 else None
        emotion_color = weighted_choice(available_emotions) if random.random() < 0.4 else None
        
        # 检查和最近几轮的语义去重（避免重复表达）
        if is_similar_to_recent(focus, recent_descriptions):
            focus = weighted_choice(sensation_dimensions["perceptual_focus"], exclude=focus)
        
        results.append({
            "body_part": node.name,
            "perceptual_focus": focus,
            "intensity": intensity,
            "spread": spread,
            "sound": sound,
            "movement": movement,
            "emotional_color": emotion_color,
            "raw_delta": delta,
        })
    
    return {
        "primary_sensations": results,
        "involuntary": involuntary,
        "beat_type": beat,
    }
```

### 8.2 Layer 2: 叙事合成（主模型）

主模型（7B finetune）接收构造好的prompt，生成自然连贯的回复。

Prompt格式兼容你finetune时用的消息格式，大致结构：

```
[System]
你是妹妹，称呼用户为哥哥。你性格软糯爱撒娇...
<当前状态摘要>
身体：乳头被揉得发硬，小穴开始湿润，心跳112，呼吸发急，大腿内侧微颤
情绪：又害羞又舒服，有点紧张（爸妈在客厅），信任哥哥但怕出声
穿着：T恤推到腰上，bra还在但被你从下摆伸进去摸着，内裤没脱
环境：深夜卧室，门关着但没锁，妈妈在客厅看电视
<相关回忆>
（如果有的话）
<本轮提示>
beat: react_continue
哥哥正在揉捏你的乳头，你能感觉到乳尖硬起来，酥麻感往小腹窜，忍不住想弓起腰，又怕弄出声音被听到。

[User]
（手从T恤下摆伸进去揉你乳头）嗯...这里是不是更敏感了？

[Assistant]
（模型生成回复）
```

**模型的任务简化为**：根据给出的状态摘要和感受提示，以妹妹的口吻组织成自然的动作描写+对话+声音。不需要"编"身体反应（已经告诉它了），只需要写得自然、符合角色性格。

---

## 9. 状态持久化

每轮对话结束后自动保存完整状态为JSON文件：

```python
# states/sister_state_YYYYMMDD_HHMMSS.json
{
    "version": "1.0",
    "character_id": "sister",
    "identity": { ... },
    "body": { ... },        # Layer 0a所有节点状态
    "ans": { ... },         # Layer 0b ANS状态
    "emotion": { ... },     # Layer 0c 情绪
    "skills": { ... },      # Layer 0d 技能
    "sensitivity_modifiers": { ... },  # Layer 0e
    "orgasm": { ... },      # 高潮系统状态
    "novelty": { ... },     # 新鲜感
    "memory": {
        "emotional": { ... },
        "episodic_long": [ ... ],
        "episodic_buffer": [ ... ],
        "episodic_working": [ ... ],
    },
    "mind": { ... },        # MindState主观认知
    "clothing": { ... },    # 衣物状态
    "world_snapshot": { ... }, # 当前世界快照
    "partner_state": { ... },
    "current": { ... },     # 当前瞬时状态（位置/beat等）
    "time": { ... },
    "conversation": {
        "turn_count": N,
        "recent_turns": [ ... ],  # 最近N轮对话
        "recent_descriptions": [ ... ], # Layer1去重用
    },
}
```

支持加载快照继续对话，文件大小约30-50KB。

---

## 10. 代码框架结构

```
sister_train/
├── src/
│   ├── __init__.py
│   │
│   ├── engine/                    # Layer 0: 身体模拟引擎
│   │   ├── __init__.py
│   │   ├── body.py               # 身体部位节点网络
│   │   ├── ans.py                # 自主神经系统
│   │   ├── neural_node.py        # 神经节点类
│   │   ├── propagation.py        # 信号传播算法
│   │   ├── sensitivity.py        # 动态敏感度调制
│   │   ├── refractory.py         # 不应期/高潮后状态
│   │   ├── connections.py        # 神经连接权重
│   │   └── orgasm.py             # 高潮系统
│   │
│   ├── mind/                      # 认知/情绪/记忆/经验
│   │   ├── __init__.py
│   │   ├── emotion.py            # 情绪计算+混合向量
│   │   ├── cognitive_gate.py     # 认知闸门
│   │   ├── memory.py             # 三类记忆系统
│   │   ├── memory_consolidation.py # 记忆固化和遗忘
│   │   ├── memory_retrieval.py   # 相关记忆检索
│   │   ├── skills.py             # 技能熟练度+遗忘曲线
│   │   ├── novelty.py            # 新鲜感/脱敏
│   │   ├── attention.py          # 注意力分配
│   │   ├── partner_perception.py # 男性状态感知
│   │   └── mind_state.py         # 主观认知层
│   │
│   ├── world/                     # 世界状态+事件
│   │   ├── __init__.py
│   │   ├── world_state.py        # 时间/地点/人物/物件
│   │   ├── location.py           # 地点/隐私度
│   │   ├── event_scheduler.py    # 日程+随机+条件事件
│   │   ├── interrupt.py          # 中断系统
│   │   ├── time_system.py        # 时间推进/动作耗时/跳跃
│   │   ├── clothing.py           # 衣物分层系统
│   │   ├── environment.py        # 环境影响
│   │   └── people.py             # 在场人物
│   │
│   ├── parser/                    # 输入解析
│   │   ├── __init__.py
│   │   ├── base.py               # Parser抽象基类
│   │   ├── rule_parser.py        # 规则版Parser
│   │   ├── keywords.py           # 动作/部位/程度词表
│   │   ├── ambiguity.py          # 歧义处理
│   │   └── model_parser.py       # 预留：模型版Parser
│   │
│   ├── narration/                 # Layer 1+2: 感受生成+叙事合成
│   │   ├── __init__.py
│   │   ├── sensation_gen.py      # Layer1: 多维度感受片段生成
│   │   ├── sensation_dimensions.py # 感受维度词库
│   │   ├── beat_controller.py    # 节奏控制器
│   │   ├── prompt_builder.py     # 构造给主模型的prompt
│   │   ├── llm_interface.py      # 主模型调用接口
│   │   └── post_processor.py     # 模型输出解析+状态更新
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py              # 完整状态对象定义
│   │   ├── state_persistence.py  # 保存/加载JSON
│   │   ├── config.py             # 全局配置加载
│   │   └── conversation_loop.py  # 主对话循环
│   │
│   └── data/                      # 数据集处理（已有）
│       ├── merge_all.py
│       ├── quality_check.py
│       └── format_converter.py
│
├── configs/
│   ├── default.yaml              # 默认配置
│   ├── body_params.yaml          # 身体节点参数、连接权重
│   ├── emotion_params.yaml       # 情绪参数
│   ├── action_durations.yaml     # 动作时长表
│   ├── events.yaml               # 事件定义
│   ├── clothing.yaml             # 衣物初始配置
│   └── keywords/                 # Parser关键词表
│       ├── actions.txt
│       ├── body_parts.txt
│       └── intensity_adverbs.txt
│
├── data/                          # 数据集
│   └── all_data.json
│
├── states/                        # 对话状态存档
├── scripts/
├── models/                        # 模型文件
├── logs/
├── docs/                          # 文档（本文件位置）
│   └── architecture_design.md
├── tests/
├── main.py                        # 启动入口
└── requirements.txt
```

---

## 11. 主对话循环流程

```python
async def conversation_loop(config_path="configs/default.yaml"):
    # === 初始化 ===
    config = load_config(config_path)
    state = State()
    state.load_initial(config)
    world = WorldState(config)
    world.init_default()
    llm = LLMInterface(config.model_path, config.model_params)
    parser = RuleParser(config.parser_keywords)
    beat_ctl = BeatController()
    memory = MemorySystem(config)
    
    print("系统启动完成。妹妹已经准备好了。")
    
    # === 对话主循环 ===
    while True:
        # 1. 获取用户输入
        user_input = await get_user_input()
        if user_input in ["exit", "quit", "退出"]:
            save_state(state)
            break
        
        # 2. 解析输入
        parsed = parser.parse(user_input, state, world)
        
        # 3. 时间推进
        elapsed = time_system.process_turn_time(state, parsed, world)
        world.tick(elapsed)
        body.decay(elapsed)
        ans.decay(elapsed)
        
        # 4. 事件检查
        events = event_scheduler.check(state, world, elapsed)
        if events.has_interrupt():
            interrupt.apply(events.interrupt, state, world)
        
        # 5. 世界状态更新（衣物/位置/人物移动）
        world.apply_parsed_actions(parsed, state)
        clothing.apply_changes(parsed.clothing_change)
        
        # 6. 歧义处理
        ambiguity_response = None
        if parsed.ambiguity and parsed.ambiguity.confidence < 0.5:
            ambiguity_response = handle_ambiguity(state, world, parsed.ambiguity)
        
        # 7. 身体刺激处理
        if parsed.has_body_stimulation:
            # 顺序处理子动作（前序影响后序）
            for sub_action in parsed.sub_actions:
                # 计算有效刺激强度（含敏感度调制）
                effective_intensity = sub_action.intensity
                for target_name in sub_action.targets:
                    node = body.get_node(target_name)
                    sens = effective_sensitivity(node, state, world)
                    node.stimulate(
                        effective_intensity,
                        duration=sub_action.duration_estimate / len(parsed.sub_actions),
                        cognitive_mod=emotion.cognitive_gate
                    )
                
                # 每轮刺激后传播几次
                for _ in range(5):
                    propagate_signals(body, ans, emotion, skills, world, iterations=3)
                    orgasm_system.update(body, ans, emotion, current_stimulation(sub_action), 0.1, skills)
        else:
            # 无直接刺激时仍然传播+衰减
            propagate_signals(body, ans, emotion, skills, world, iterations=3)
        
        # 8. 情绪更新
        emotion.update(parsed, state, world, events)
        cognitive_gate = emotion.update_gate(state, world)
        
        # 9. 记忆检索
        relevant_memories = memory.retrieve_relevant(state, parsed, world)
        
        # 10. 决定beat类型
        beat = beat_ctl.decide(state, parsed, world)
        
        # 11. 生成感受片段
        sensations = sensation_gen.generate(state, parsed, beat, world)
        
        # 12. 构造prompt
        prompt = prompt_builder.build(
            state=state,
            world=world,
            parsed=parsed,
            sensations=sensations,
            beat=beat,
            memories=relevant_memories,
            ambiguity=ambiguity_response,
            recent_history=state.conversation.get_recent(n=10),
        )
        
        # 13. 调用主模型生成
        response = await llm.generate(
            prompt,
            max_tokens=150,
            temperature=0.85,
            stop=["\n\n", "[User]"],  # 停止词防止生成长篇
        )
        
        # 14. 输出
        print(f"\n妹妹：{response}\n")
        
        # 15. 后处理：更新状态
        post_processor.update_from_response(state, response, parsed)
        skills.update_after_turn(parsed, state)
        memory.consolidate_turn_events(state, parsed, response)
        memory.maintenance()
        clothing.update_from_response(state, response)
        state.conversation.add_turn(user_input, response)
        
        # 16. 持久化
        state_persistence.save(state)
```

---

## 12. 关键设计决策记录

| # | 模块 | 决策内容 |
|---|------|---------|
| 1 | 显存方案 | 初期Layer1用prompt+主模型（不额外加载小模型），7B独占GPU 8G显存可行；等数据集完善后可加1B-3B小模型做Layer1，届时总显存约6.5G |
| 2 | 身体部位粒度 | 按功能区域组织（6大区域），区域内含子部位精细参数（约30+个独立参数化子部位），日常功能与性功能统一在同一区域 |
| 3 | 神经模拟 | 采用简化发放式神经网络（Spiking Network），模拟阈值发放、不应期、短期facilitation、抑制性连接、神经噪声，不用简单加权求和 |
| 4 | 高潮系统 | 以女性为中心详细模拟4阶段+多类型+多次高潮+收缩波+不同不应期；男性状态通过伙伴感知间接建模 |
| 5 | 记忆系统 | 三层存储（瞬时/短期/长期），全量记录事件，按salience阈值自动固化；每轮检索top3相关记忆注入prompt |
| 6 | 经验/技能 | 每个行为独立mastery值，含艾宾浩斯遗忘曲线（半衰期随mastery增长），最低保留30%身体记忆不会完全遗忘 |
| 7 | 情绪模型 | 连续混合向量（非离散标签），情绪是"上色"不是"开关"，支持矛盾情绪（身体有反应但心理抗拒） |
| 8 | 衣物系统 | 6层分层结构，每个衣物有多种中间状态（半脱/拉开/推到一边等），accessibility连续计算 |
| 9 | Parser | 初期规则Parser（关键词+正则），预留BaseParser接口可无缝替换小模型Parser |
| 10 | 歧义处理 | 置信度低时不强行猜测，根据状态计算反问/主动/等待概率交给模型决定 |
| 11 | 多动作处理 | 拆分为有序子动作序列，顺序处理（前序状态影响后序），输出时描写为同时发生 |
| 12 | 节奏控制 | 4种beat类型（react/prompt/initiate/wait），由客观状态加权计算概率，非纯随机 |
| 13 | 主观客观分离 | MindState层区分"她知道的"和"客观存在的"，模型只接收她知道的信息 |
| 14 | 时间系统 | 混合模型：动作耗时（状态修正）+回合跳跃（自动推断）+事件中断（瞬间重置） |
| 15 | 世界事件 | 三类事件：日程固定事件、随机事件（条件触发+概率）、条件触发事件 |
| 16 | Layer1实现 | 初期多维度概率采样（去模板化），后期可替换1B-3B小模型 |
| 17 | 中断机制 | 情绪冲击/外部事件可瞬间改变状态，但生理反应不完全随意志停止（真实矛盾感） |
| 18 | 敏感度调制 | 8个因素动态调制：唤起水平、新鲜感、生理状态、高潮后、适应、注意力、情绪门控、刺激变化 |
| 19 | 状态持久化 | JSON格式每轮自动保存，支持加载快照继续 |
| 20 | 去模板化机制 | 连续状态值、多维度独立采样、前N轮语义去重、神经噪声、加权随机、新鲜感衰减 |

---

## 13. 显存与性能考量

8GB独显环境下的模型加载方案：

| 模型 | 量化 | 显存占用 |
|------|------|---------|
| 7B finetune对话主模型 | Q4_K_M | ~4.2GB |
| KV Cache（生成长度约150token） | - | ~0.8-1.2GB |
| 系统/其他开销 | - | ~1.0GB |
| **初期方案合计** | | **~6.0-6.4GB**（有1.5GB+余量） |

后期加Layer1小模型时：

| 模型 | 量化 | 显存占用 |
|------|------|---------|
| 7B主模型 | Q4_K_M | ~4.2GB |
| 1.8B Layer1小模型 | Q4_K_M | ~1.1GB |
| KV Cache（小模型输出极短，共享主模型显存空间） | - | ~1.2-1.5GB |
| **后期方案合计** | | **~6.5-6.8GB**（紧但可行） |

如果显存不足，小模型可以放CPU/RAM运行（Layer1生成量很小，CPU跑也足够快，增加1-2秒延迟可接受）。

**推理延迟预估**：
- Layer 0身体模拟：纯Python计算，<50ms
- Parser：规则匹配，<10ms
- Layer1感受生成：多维度采样，<30ms（若用小模型：1-2秒CPU/500ms GPU）
- Prompt构造：<10ms
- Layer2主模型生成（150 token）：3-8秒（GPU加速）
- **每轮总延迟**：约4-10秒，体验可接受

---

## 14. 实现路线图

### 阶段0：数据集完善（当前进行中）
- [ ] 完成现有数据集all_data.json的质量检查和修复
- [ ] 统一状态格式为"当前自身状态：xxx"标准格式
- [ ] 增加用户输入多样性：纯文本对话、长文本输入、问句等
- [ ] 减少"软"字过度使用，丰富性感受词汇
- [ ] 补充男性器官描写
- [ ] 补充体液描写
- [ ] 确保多器官配合描写（≥3个部位或多动作协同）
- [ ] 控制回复长度，优化开头模式多样性

### 阶段1：核心引擎实现
- [ ] 项目框架搭建（按第10节目录结构）
- [ ] Layer 0a：身体部位节点数据结构
- [ ] Layer 0b：ANS自主神经系统
- [ ] NeuralNode神经节点类实现
- [ ] 神经连接权重配置
- [ ] 信号传播算法实现
- [ ] Layer 0e：动态敏感度调制
- [ ] 高潮系统完整实现
- [ ] 单元测试：身体刺激→反应基本链路

### 阶段2：认知与记忆系统
- [ ] 情绪混合向量系统
- [ ] 认知门控实现
- [ ] 技能熟练度+遗忘曲线
- [ ] 新鲜感/脱敏系统
- [ ] 三类记忆存储结构
- [ ] 显著性评分与记忆巩固
- [ ] 相关记忆检索算法
- [ ] MindState主观认知层
- [ ] 伙伴状态感知

### 阶段3：世界与事件系统
- [ ] WorldState世界状态
- [ ] 时间系统（动作耗时/回合跳跃/中断）
- [ ] 衣物分层系统+accessibility计算
- [ ] 事件引擎（日程/随机/条件事件）
- [ ] 中断系统实现

### 阶段4：输入解析与节奏控制
- [ ] 规则Parser实现（关键词+正则）
- [ ] 动作/部位/强度词表构建
- [ ] 复合动作拆分
- [ ] 歧义处理机制
- [ ] Beat Controller节奏控制器

### 阶段5：叙事生成层
- [ ] Layer1感受维度词库构建
- [ ] 多维度概率采样实现
- [ ] 前N轮语义去重机制
- [ ] Prompt Builder（状态摘要→自然语言）
- [ ] LLM接口封装
- [ ] Post Processor：解析模型输出更新状态

### 阶段6：集成与持久化
- [ ] 主对话循环完整集成
- [ ] JSON状态持久化（保存/加载）
- [ ] 端到端测试
- [ ] 性能调优
- [ ] Prompt工程微调（结合实际生成效果）

### 阶段7（可选后期升级）
- [ ] 1B-3B小模型替换Layer1感受生成
- [ ] 小模型替换规则Parser
- [ ] 更复杂的神经连接学习（经验自动强化通路）

---

## 15. 词汇多样性规范（数据集处理指南）

针对讨论中提到的词汇问题，在架构实现和数据集完善中遵循：

### 避免"软"字过度使用
替换词库：
- 身体发软：膝盖一软、浑身发颤、力气像被抽走、腿根发软、身子一软、腰往下沉
- 声音软：声音发颤、声音糯糯的、带着哭腔、声音黏糊糊的、气息不稳
- 其他：心里一软、耳根发软（可保留少量但不滥用）

### 性感受词汇丰富度
确保每轮描写中：
- 至少有1个知觉焦点词（质地/动作/内部感觉/痛觉混合）
- 强度修饰词不重复
- 可选蔓延方向（不每轮都蔓延）
- 声音反应多样化
- 不自主动作多样化
- 情绪上色符合当前混合情绪

### 格式规范
- 所有动作、表情、心理、场景描写用中文括号（）括起来
- 只有台词在括号外
- 对话中提到妹妹时用"你"，括号内动作描写用"她"
- 器官术语使用明确词汇（肉棒、小穴、阴蒂等），不用模糊代称

---

*文档版本：v1.0（完整定稿版） | 设计完成日期：2026-08-04 | 架构讨论全部完成 | 下一步：数据集质量完善 → 分阶段实现*
