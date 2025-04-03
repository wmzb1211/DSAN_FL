# 基于新型差分隐私的联邦学习系统设计与实现
​	提出新型的DSAN_FL算法，该算法出动态敏感度-自适应噪声联邦学习框架（DSAN_FL），其核心创新在于三阶协同优化：首先，设计梯度敏感度驱动的动态DP机制，通过实时追踪梯度L2范数分布（计算均值𝜇_𝑔与标准差𝜎_𝑔），动态调整噪声乘数𝜎(𝑡) = $𝜎_{𝑏𝑎𝑠𝑒} ·𝑒𝑥𝑝(−𝜆·\frac{𝜇_𝑔}{𝜎_𝑔}  )$。在模型初期梯度波动大时降低噪声强度（减少有效信息损失），后期梯度稳 定时增强噪声（防止隐私预算过早耗尽）；其次，开发残差补偿型梯度稀疏化， 客户端采用Top-K 选择（保留 10%梯度）后，将丢弃的梯度残差通过动量累积 $𝑟(𝑡) = 0.9𝑟(𝑡 −1) +0.1(𝑔(𝑡)− 𝑆𝑝𝑎𝑟𝑠𝑒(𝑔(𝑡))$补偿至后续训练轮次。最后，构建去中心化安全聚合架构，服务器基于谱聚类算 法（相似度阈值>0.85）将客户端动态分组后实施组内噪声增强聚合。

# TODO
- MLP mnist 50epochs epsilon=8 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- MLP mnist 50epochs epsilon=8 noiseMultiplier=0.2 <span style="color:red;">✗</span>
- MLP mnist 50epochs epsilon=8 noiseMultiplier=0.3 <span style="color:red;">✗</span>
- MLP mnist 50epochs epsilon=4 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- MLP mnist 50epochs epsilon=2 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- MLP mnist 50epochs epsilon=1 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN mnist 50epochs epsilon=8 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN mnist 50epochs epsilon=8 noiseMultiplier=0.2 <span style="color:red;">✗</span>
- CNN mnist 50epochs epsilon=8 noiseMultiplier=0.3 <span style="color:red;">✗</span>
- CNN mnist 50epochs epsilon=4 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN mnist 50epochs epsilon=2 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN mnist 50epochs epsilon=1 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=8 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=8 noiseMultiplier=0.2 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=8 noiseMultiplier=0.3 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=4 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=2 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=1 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=8 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=8 noiseMultiplier=0.2 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=8 noiseMultiplier=0.3 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=4 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=2 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- CNN cifar 100epochs epsilon=1 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=8 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=8 noiseMultiplier=0.2 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=8 noiseMultiplier=0.3 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=4 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=2 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=1 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=8 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=8 noiseMultiplier=0.2 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=8 noiseMultiplier=0.3 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=4 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=2 noiseMultiplier=0.1 <span style="color:red;">✗</span>
- ResNet cifar 100epochs epsilon=1 noiseMultiplier=0.1 <span style="color:red;">✗</span>











