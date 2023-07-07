def count_substrings(s, k):
    count = 0  # 记录满足条件的子串数量

    for i in range(len(s) - k + 1):
        substring = s[i:i+k]  # 获取当前位置开始的长度为k的子串
        char_counts = {}  # 用于记录每个字母出现的次数

        # 统计子串中每个字母出现的次数
        for char in substring:
            char_counts[char] = char_counts.get(char, 0) + 1

        # 判断是否满足所有字母出现的次数为偶数次
        if all(count % 2 == 0 for count in char_counts.values()):
            count += 1

    return count


# 测试代码
s = "aabbc"
k = 2
result = count_substrings(s, k)
print(f"满足条件的长度为{k}的连续子串数量：{result}")
