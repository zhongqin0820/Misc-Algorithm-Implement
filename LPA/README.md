# 创建日期
2018/07/23

# 参考资料
- [标签传播算法（Label Propagation）及Python实现](https://blog.csdn.net/zouxy09/article/details/49105265)

# 优化
对算法的并行化，一般分为两种：数据并行和模型并行。

迭代算法，除了判断收敛外，我们还可以让每迭代几步，就用测试label测试一次结果，看模型的整体训练性能如何。特别是判断训练是否过拟合的时候非常有效。因此，代码中包含了这部分内容。
