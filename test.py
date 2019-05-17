from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Solution:
    def StrToInt(self, s):
        # write code here
        if not s or len(s) == 0:
            return 0
        flag = 1
        res = 0
        j = 1 if(s[0] == "+" or  s[0] == "-") else 0
        if s[0] == "-":
            flag = 0
        for i in range(j, len(s)):
            res = res*10 + (ord(s[i]) - ord('0'))
        if not flag:
            res = -res
        return res

def set_threshold(pre_reg, threshold):
    res = []
    for i in range(len(pre_reg)):
        if pre_reg[i] > threshold:
           res.append(1)
        else:
            res.append(0)
    return res

a = [0.677478, 0.675171, 0.545794, 0.563495]
b = np.array(a)
print(len(b))
print(b[0])
res = set_threshold(a, 0.6)
print(res)
# print(set_threshold(np.array(a), 0.6))