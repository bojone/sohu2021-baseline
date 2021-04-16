# sohu2021-baseline
2021搜狐校园文本匹配算法大赛baseline

## 简介

分享了一个搜狐文本匹配的baseline，主要是通过条件LayerNorm来增加模型的多样性，以实现同一模型处理不同类型的数据、形成不同输出的目的。

线下验证集F1约0.74，线上测试集F1约0.73。预训练模型是[RoFormer](https://github.com/ZhuiyiTechnology/roformer)，也欢迎对比其他预训练模型的效果。

测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.5，如果在其他环境组合下报错，请根据错误信息自行调整代码。

详情请看：https://kexue.fm/archives/8337

## 交流

QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
