# main.py
import sys
import re
import math
from collections import Counter
import string


def preprocess_text(text):
    """预处理文本：去除标点符号、转换为小写、分词"""
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation + '，。！？；：“”‘’【】（）《》'))
    # 转换为小写
    text = text.lower()
    # 简单的中文分词（按字符分割）
    words = []
    current_word = ''

    for char in text:
        if char.strip():  # 非空白字符
            # 中文字符直接作为一个词
            if '\u4e00' <= char <= '\u9fff':
                if current_word:
                    words.append(current_word)
                    current_word = ''
                words.append(char)
            # 英文字母和数字组成单词
            elif char.isalnum():
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ''
        else:
            if current_word:
                words.append(current_word)
                current_word = ''

    if current_word:
        words.append(current_word)

    return words


def get_tfidf_vectors(text1, text2):
    """计算两个文本的TF-IDF向量"""
    # 预处理文本
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # 构建词汇表
    all_words = set(words1 + words2)

    # 计算词频(TF)
    tf1 = Counter(words1)
    tf2 = Counter(words2)

    # 计算文档频率(DF)
    doc_freq = {}
    for word in all_words:
        count = 0
        if word in tf1:
            count += 1
        if word in tf2:
            count += 1
        doc_freq[word] = count

    # 计算TF-IDF向量
    vector1 = []
    vector2 = []

    for word in all_words:
        # TF-IDF = TF * log((N + 1)/(DF + 1)) + 1
        idf = math.log((2 + 1) / (doc_freq.get(word, 1) + 1)) + 1

        tf_idf1 = (tf1.get(word, 0) / len(words1)) * idf if words1 else 0
        tf_idf2 = (tf2.get(word, 0) / len(words2)) * idf if words2 else 0

        vector1.append(tf_idf1)
        vector2.append(tf_idf2)

    return vector1, vector2


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错：{e}")
        sys.exit(1)


def calculate_similarity(original_text, copied_text):
    """计算文本相似度"""
    if not original_text or not copied_text:
        return 0.0

    vec1, vec2 = get_tfidf_vectors(original_text, copied_text)
    similarity = cosine_similarity(vec1, vec2)

    return max(0.0, min(1.0, similarity))  # 确保在0-1范围内


def write_result(result, output_path):
    """将结果写入文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{result:.2f}")
    except Exception as e:
        print(f"写入文件时出错：{e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件] [抄袭版论文文件] [答案文件]")
        sys.exit(1)

    original_path = sys.argv[1]
    copied_path = sys.argv[2]
    output_path = sys.argv[3]

    # 读取文件
    original_text = read_file(original_path)
    copied_text = read_file(copied_path)

    # 计算相似度
    similarity = calculate_similarity(original_text, copied_text)

    # 写入结果
    write_result(similarity, output_path)

    print(f"查重完成，重复率为: {similarity:.2%}")


if __name__ == "__main__":
    main()