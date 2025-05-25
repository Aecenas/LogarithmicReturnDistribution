# 对数收益率分布分析

![Version](https://img.shields.io/badge/version-1.0.0-blue)

## 项目概述

本项目用于分析ETF产品的对数收益率分布特征，并比较正态分布和学生t分布的拟合效果。主要功能包括：
- 从Tushare获取ETF历史数据
- 计算对数收益率
- 拟合学生t分布参数
- 可视化实际分布与理论分布对比
- 基于肥尾分布的期权定价模型

## 功能特性

- 支持多只ETF产品分析
- 自动计算年化波动率
- 生成分布对比图表
- 提供基于学生t分布的欧式期权定价函数
- 提供QuantLib的BS模型期权定价函数

## 快速开始

1. 安装依赖:
```bash
pip3 install -r requirements.txt
```
2. 配置Tushare API Token:
修改main.py中的 ts.pro_api("your_api_token")

3. 运行分析:
```bash
python3 main.py
```

## 文件结构

```tree
LogarithmicReturnDistribution/
├── test_figure/
├── LICENSE
├── README.md
├── VERSION
├── main.py
└── requirements.txt
```

## 许可证

MIT License - 详见 LICENSE 文件

## 维护者

- Ain